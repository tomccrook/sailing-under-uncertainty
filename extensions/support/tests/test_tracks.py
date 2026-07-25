import csv
import hashlib
import io
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from sail_support.calibration import fit_timing, transitions
from sail_support.contracts import ImportConfig
from sail_support.fixtures import synthetic_csv
from sail_support.server import create_app
from sail_support.tracks import import_track, save_track


def test_flags_keep_rows_and_disconnect_every_bad_fix() -> None:
    track = import_track(synthetic_csv(1, 10, corrupt=True), "csv", session_id="test")
    assert len(track.points) == 170
    flags = {flag for point in track.points for flag in point.flags}
    assert {"duplicate_time", "clock_regression", "implausible_ground_speed", "gap_before"} <= flags
    assert sum(not point.accepted for point in track.points) == 3
    assert len({p.segment for p in track.points if p.accepted}) == 5
    for a, b in zip(track.points, track.points[1:], strict=False):
        if not a.accepted and b.accepted:
            assert b.derived_speed_over_ground_m_s is None
    assert track.points[0].east_m == pytest.approx(0, abs=1e-6)
    assert track.points[0].north_m == pytest.approx(0, abs=1e-6)
    assert track.points[1].derived_speed_over_ground_m_s == pytest.approx(3, abs=1e-6)


def test_raw_and_versioned_tables_round_trip(tmp_path: Path) -> None:
    raw = synthetic_csv(1, 10)
    track = import_track(raw, "csv", session_id="test", imported_utc=datetime.now(UTC))
    save_track(track, raw, tmp_path / "import")
    assert (tmp_path / "import/source.csv").read_bytes() == raw
    assert pq.read_table(tmp_path / "import/points.parquet").num_rows == len(track.points)
    assert track.source_sha256 == hashlib.sha256(raw).hexdigest()
    with pytest.raises(FileExistsError):
        save_track(track, raw, tmp_path / "import")
    with pytest.raises(ValueError, match="hash"):
        save_track(track, b"changed", tmp_path / "new")


def test_gpx_course_is_not_hull_heading_and_segments_stay_separate() -> None:
    raw = b"""<gpx version="1.0" creator="fixture" xmlns="http://www.topografix.com/GPX/1/0">
    <trk><trkseg><trkpt lat="50.7" lon="-1.3"><time>2026-01-01T00:00:00Z</time>
    <course>45</course></trkpt></trkseg><trkseg><trkpt lat="50.7" lon="-1.3">
    <time>2026-01-01T00:00:01Z</time></trkpt></trkseg></trk></gpx>"""
    track = import_track(raw, "gpx", session_id="gpx")
    assert all(p.heading_deg is None for p in track.points)
    assert [p.segment for p in track.points] == [0, 1]
    with pytest.raises(ValueError, match="entity"):
        import_track(b'<!DOCTYPE gpx [<!ENTITY x "hello">]><gpx/>', "gpx", session_id="bad")


@pytest.mark.parametrize(
    "timestamp,latitude",
    [
        ("2026-01-01T12:00:00", "50"),
        ("2026-01-01T12:00:00Z", "nan"),
        ("2026-01-01T12:00:00Z", "91"),
    ],
)
def test_bad_clock_and_coordinates_retained(timestamp: str, latitude: str) -> None:
    raw = f"timestamp_utc,latitude_deg,longitude_deg\n{timestamp},{latitude},0\n".encode()
    track = import_track(raw, "csv", session_id="bad")
    assert not track.points[0].accepted
    assert track.points[0].flags


def test_byte_and_row_limits() -> None:
    with pytest.raises(ValueError, match="byte limit"):
        import_track(b"x" * 100, "csv", session_id="big", config=ImportConfig(max_bytes=50))
    with pytest.raises(ValueError, match="row limit"):
        import_track(
            synthetic_csv(1, 10), "csv", session_id="big", config=ImportConfig(max_rows=50)
        )


def test_chronological_day_split_is_held_out_and_fixture_fit_is_known() -> None:
    tracks = tuple(
        import_track(
            synthetic_csv(day, duration, corrupt=True),
            "csv",
            session_id=f"day-{day}",
            evidence="synthetic_fixture",
        )
        for day, duration in enumerate((10, 12, 14, 16, 18, 20), 1)
    )
    report = fit_timing(tracks)
    assert len(report.transitions) == 12
    assert report.fitted_mean_s == 13
    assert report.holdout_mae_s == 6
    assert max(report.training_days) < min(report.holdout_days)
    assert "minimum tack time" in report.unsupported
    changed = tracks[:-2] + tuple(
        import_track(
            synthetic_csv(day, 25), "csv", session_id=f"day-{day}", evidence="synthetic_fixture"
        )
        for day in (5, 6)
    )
    assert fit_timing(changed).fitted_mean_s == report.fitted_mean_s
    assert fit_timing(tracks).model_dump() == report.model_dump()
    with pytest.raises(ValueError, match="Duplicate"):
        fit_timing(tracks + (tracks[0],))
    with pytest.raises(ValueError, match="six"):
        fit_timing(tracks[:4])


def test_gps_only_and_gybes_do_not_calibrate_upwind_transitions() -> None:
    rows = list(csv.DictReader(io.StringIO(synthetic_csv(1, 10).decode())))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        row["wind_from_deg"] = "180"  # These are now downwind gybes.
        writer.writerow(row)
    track = import_track(buffer.getvalue().encode(), "csv", session_id="gybe")
    assert not transitions(track)
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=["timestamp_utc", "latitude_deg", "longitude_deg"], extrasaction="ignore"
    )
    writer.writeheader()
    writer.writerows(rows)
    assert not transitions(import_track(buffer.getvalue().encode(), "csv", session_id="gps-only"))


def test_local_import_is_private_bounded_and_origin_checked(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "private"))
    assert client.get("/health").json()["cloud_submission"] == "disabled"
    response = client.post(
        "/tracks/import?format=csv",
        content=synthetic_csv(1, 10),
        headers={"Origin": "https://untrusted.example"},
    )
    assert response.status_code == 403
    assert not (tmp_path / "private").exists()
    response = client.post(
        "/tracks/import?format=csv",
        content=synthetic_csv(1, 10),
        headers={"Origin": "http://127.0.0.1:5187"},
    )
    assert response.status_code == 200
    assert response.json()["track"]["evidence"] == "unvalidated_measurement"
    identifier = response.json()["import_id"]
    assert (tmp_path / "private" / identifier / "source.csv").exists()
    assert client.get(f"/tracks/{identifier}").status_code == 200
    assert client.get("/tracks/unsafe").status_code == 400
    assert client.post("/tracks/import?format=csv", content=b"x" * 5_000_001).status_code == 413
    assert client.post("/tracks/import?format=csv", content=b"bad").status_code == 422
    assert client.post("/tracks/import?format=gpx", content=b"<broken").status_code == 422
