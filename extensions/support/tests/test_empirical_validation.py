"""Independent held-out and measurement-identification integrity tests."""

import gzip
import io
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from scripts import empirical_validation as empirical


def dbf(payload: bytes) -> bytes:
    header = bytearray(32)
    header[0] = 3
    struct.pack_into("<IHH", header, 4, 1, 65, 5)
    field = bytearray(32)
    field[:4] = b"name"
    field[11] = ord("C")
    field[16] = 4
    return bytes(header + field + b"\r" + b" " + payload + b"\x1a")


def test_dbf_bounds_and_record_ids() -> None:
    row = list(empirical.read_dbf(io.BytesIO(dbf(b"boat"))))
    assert row == [{"_record": "0", "_deleted": "False", "name": "boat"}]
    with pytest.raises(ValueError, match="truncated"):
        list(empirical.read_dbf(io.BytesIO(dbf(b"boa")[:-1])))
    modified = bytearray(dbf(b"boat"))
    struct.pack_into("<I", modified, 4, 1_000_000)
    with pytest.raises(ValueError, match="bounds"):
        list(empirical.read_dbf(io.BytesIO(modified)))


def test_missing_wind_does_not_bridge_gap_or_split() -> None:
    text = """2025 03 01 23 50 0 5
2025 03 02 00 00 0 5
2025 03 02 00 10 999 99
2025 03 02 00 20 90 5
2025 03 02 00 30 90 6
"""
    pairs, counts = empirical.wind_pairs(gzip.compress(text.encode()), empirical.EmpiricalPlan())
    assert len(pairs) == 1 and pairs[0]["split"] == "heldout"
    assert pairs[0]["x"] == [-5.0, pytest.approx(0.0, abs=1e-10)]
    assert counts["missing_or_invalid_wind"] == 1
    assert counts["gap_or_day_boundary"] == 1


def wind_fixture(shift: float = 0) -> bytes:
    from datetime import UTC, datetime, timedelta

    values = []
    for day in (datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 3, 2, tzinfo=UTC)):
        for i in range(144):
            t = day + timedelta(minutes=10 * i)
            speed = 5 + np.sin(i / 10) + (shift if t.month == 3 else 0)
            values.append(f"{t:%Y %m %d %H %M} {int(180 + 10 * np.sin(i / 13))} {speed}")
    return gzip.compress("\n".join(values).encode())


def test_heldout_targets_cannot_change_forecaster_fit(tmp_path: Path) -> None:
    fits = []
    scores = []
    for i, shift in enumerate((0.0, 3.0)):
        source = tmp_path / f"source{i}"
        source.mkdir()
        (source / empirical.WIND_FILE).write_bytes(wind_fixture(shift))
        output = tmp_path / f"out{i}"
        output.mkdir()
        result = empirical.wind_validation(
            source, output, empirical.EmpiricalPlan(bootstrap_count=200)
        )
        fits.append(json.loads((output / "wind-fit.json").read_text()))
        scores.append(result["predictors"]["AR1"]["vector_rmse_m_s"])
        assert result["heldout_pairs"] == 143
        assert result["heldout_days"] == 1
    assert fits[0] == fits[1]
    assert scores[0] != scores[1]


def test_gps_sog_is_not_water_speed_or_heading(tmp_path: Path) -> None:
    rows = []
    for i in range(3):
        rows.append(
            {
                "session": "001",
                "source_day": "2021-01-01",
                "source_record": i,
                "timestamp_utc_assumed": f"2021-01-01T00:00:{i * 10:02d}+00:00",
                "latitude_deg": 50.0,
                "longitude_deg": 0.0001 * i,
                "source_sog_knots": 1.0,
                "exclusion_reasons": [],
            }
        )
    result = empirical.track_validation(
        rows, {"gps_sessions": {"001": "heldout"}}, tmp_path, empirical.EmpiricalPlan()
    )
    track = json.loads(next((tmp_path / "tracks").glob("*/track.json")).read_text())
    assert result["raw_records"] == 3
    assert result["accepted_rows"] == 3
    assert result["projection_gate_passed"]
    assert not result["independent_instrument_comparison"]
    assert all(p["heading_deg"] is None and p["speed_water_m_s"] is None for p in track["points"])
    assert result["projection_relative_speed_error"]["heldout"]["pairs"] == 2


def test_reconstructed_gps_rows_remain_accounted_for(tmp_path: Path) -> None:
    rows = [{"exclusion_reasons": ["non_primary_or_reconstructed_provenance"], "source_record": 0}]
    result = empirical.track_validation(
        rows, {"gps_sessions": {}}, tmp_path, empirical.EmpiricalPlan()
    )
    assert result["raw_records"] == result["provenance_excluded_rows"] == 1
    assert result["processed_rows"] == 0
    assert not result["projection_gate_passed"]


def test_provenance_exclusion_breaks_nearby_gps_path(tmp_path: Path) -> None:
    records = []
    for i in range(3):
        records.append(
            {
                "session": "001",
                "source_day": "2021-01-01",
                "source_record": i,
                "timestamp_utc_assumed": f"2021-01-01T00:00:{i * 10:02d}+00:00",
                "latitude_deg": 50.0,
                "longitude_deg": 0.0001 * i,
                "source_sog_knots": 1.0,
                "exclusion_reasons": ["non_primary_or_reconstructed_provenance"] if i == 1 else [],
            }
        )
    result = empirical.track_validation(
        records, {"gps_sessions": {"001": "heldout"}}, tmp_path, empirical.EmpiricalPlan()
    )
    track = json.loads(next((tmp_path / "tracks").glob("*/track.json")).read_text())
    assert result["raw_records"] == result["processed_rows"] + result["provenance_excluded_rows"]
    assert track["points"][0]["segment"] != track["points"][1]["segment"]
    assert track["points"][1]["derived_speed_over_ground_m_s"] is None


def test_method_identity_uses_readable_repository_relative_paths() -> None:
    identities = empirical.method_hashes()
    assert "scripts/empirical_validation.py" in identities
    assert "extensions/support/src/sail_support/tracks.py" in identities
    assert all(not Path(name).is_absolute() and Path(name).exists() for name in identities)
    assert all(len(value) == 64 for value in identities.values())


def test_constant_clock_offset_preserves_local_groups_without_mutating_source() -> None:
    from scripts.recorded_clock_sensitivity import shift_records

    original = [
        {"timestamp_utc_assumed": "2024-07-01T00:20:00+00:00", "source_day": "2024-07-01"},
        {"timestamp_utc_assumed": "", "source_day": "unknown"},
    ]
    shifted = shift_records(original)
    assert shifted[0]["timestamp_utc_assumed"] == "2024-06-30T23:20:00+00:00"
    assert shifted[0]["source_day"] == original[0]["source_day"]
    assert shifted[1] == original[1]
    assert original[0]["timestamp_utc_assumed"] == "2024-07-01T00:20:00+00:00"
