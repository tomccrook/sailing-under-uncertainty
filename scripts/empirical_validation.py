"""Recorded GPS quality and chronologically held-out in-situ wind validation.

Run in the existing support environment. No boat polar or tack parameter is fitted.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import struct
import subprocess
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from typing import IO, Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field
from pyproj import Geod
from sail_support.contracts import ImportConfig
from sail_support.tracks import import_track, save_track

SOURCE_DIR = Path("artifacts/measurements/public-sources")
GPS_ARCHIVE = "Jarrett_tracks_log_v3.zip"
WIND_FILE = "44013h2025.txt.gz"


def fetch(source: Path) -> dict[str, str]:
    """Download bounded, pinned public inputs; preserve any existing source bytes."""
    source.mkdir(parents=True, exist_ok=True)
    urls = {
        GPS_ARCHIVE: (
            "https://zenodo.org/records/17340505/files/Jarrett_tracks_log_v3.zip?download=1",
            20_000_000,
        ),
        "jarrett-latest.json": ("https://zenodo.org/api/records/17340505", 200_000),
        WIND_FILE: (
            "https://www.ndbc.noaa.gov/data/historical/stdmet/44013h2025.txt.gz",
            3_000_000,
        ),
        "ndbc-measurements.html": ("https://www.ndbc.noaa.gov/faq/measdes.shtml", 200_000),
        "ndbc-station.html": (
            "https://www.ndbc.noaa.gov/station_page.php?station=44013",
            200_000,
        ),
    }
    for name, (url, limit) in urls.items():
        target = source / name
        if target.exists():
            continue
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(limit + 1)
        if not data or len(data) > limit:
            raise ValueError(f"public source exceeds download bound: {name}")
        with target.open("xb") as handle:
            handle.write(data)
    return source_hashes(source)


class EmpiricalPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    version: Literal["recorded-component-validation-v1"] = "recorded-component-validation-v1"
    gps_train_fraction: float = Field(default=2 / 3, gt=0, lt=1)
    gps_nominal_interval_s: float = Field(default=10, gt=0)
    gps_gap_s: float = Field(default=30, gt=0)
    gps_max_speed_m_s: float = Field(default=15, gt=0)
    projection_relative_p99_tolerance: float = Field(default=0.01, gt=0)
    wind_train_start: str = "2025-01-01T00:00:00+00:00"
    wind_test_start: str = "2025-03-02T00:00:00+00:00"
    wind_test_end: str = "2025-04-01T00:00:00+00:00"
    wind_interval_s: int = Field(default=600, ge=60)
    bootstrap_count: int = Field(default=4000, ge=200, le=10000)
    analysis_seed: Literal[889910] = 889910


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_dbf(stream: IO[bytes]) -> Iterator[dict[str, str]]:
    """Restricted dBASE point-attribute reader, retaining record IDs/deletion flags.

    Fixed-width fields remain strings. Bounds/lengths are validated before each
    record; no executing files or extracting arbitrary archive paths.
    """
    header = stream.read(32)
    if len(header) != 32 or header[0] != 3:
        raise ValueError("unsupported/truncated dBASE header")
    count, header_size, record_size = struct.unpack_from("<IHH", header, 4)
    if count > 120000 or header_size < 33 or record_size < 2 or record_size > 20000:
        raise ValueError("dBASE bounds exceed adapter limits")
    descriptors = stream.read(header_size - 32)
    if len(descriptors) != header_size - 32 or descriptors[-1] != 13 or (header_size - 33) % 32:
        raise ValueError("invalid dBASE descriptors")
    fields = []
    for offset in range(0, len(descriptors) - 1, 32):
        entry = descriptors[offset : offset + 32]
        name = entry[:11].split(b"\0")[0].decode("ascii")
        if chr(entry[11]) not in ("C", "D", "F", "N") or not entry[16]:
            raise ValueError("unsupported dBASE field")
        fields.append((name, entry[16]))
    if sum(size for _, size in fields) != record_size - 1 or len({n for n, _ in fields}) != len(
        fields
    ):
        raise ValueError("inconsistent dBASE record layout")
    for i in range(count):
        data = stream.read(record_size)
        if len(data) != record_size or data[0] not in (32, 42):
            raise ValueError("truncated/invalid dBASE record")
        row = {"_record": str(i), "_deleted": str(data[0] == 42)}
        position = 1
        for name, size in fields:
            row[name] = data[position : position + size].decode("utf-8").strip()
            position += size
        yield row
    if stream.read() not in (b"", b"\x1a"):
        raise ValueError("unexpected dBASE trailing records")


def gps_records(archive: Path) -> list[dict[str, Any]]:
    rows = []
    with zipfile.ZipFile(archive) as z:
        # Explicit allowlist; CRC checks occur while reading the complete member.
        name = "Zenodo_shps/Jarrett_GPS_tracks_all_sails.dbf"
        if z.getinfo(name).file_size > 220_000_000:
            raise ValueError("GPS attribute member exceeds cap")
        with z.open(name) as stream:
            for raw in read_dbf(stream):
                reasons = []
                if raw["_deleted"] == "True":
                    reasons.append("deleted_source_record")
                if raw["Source"] or "lost" in raw["sail"].lower():
                    reasons.append("non_primary_or_reconstructed_provenance")
                # Published GPS DateTime_1 is a corrected local-time character field.
                try:
                    local = datetime.fromisoformat(raw["DateTime_1"])
                    if local.tzinfo is not None:
                        raise ValueError("unexpected source zone")
                    timestamp = local.replace(tzinfo=timezone(timedelta(hours=1))).astimezone(UTC)
                    local_day = local.date().isoformat()
                except ValueError:
                    timestamp = None
                    local_day = "unknown"
                    reasons.append("unparseable_local_clock")

                def number(key: str, record: dict[str, str] = raw) -> float | None:
                    try:
                        value = float(record[key])
                        return value if math.isfinite(value) else None
                    except ValueError:
                        return None

                rows.append(
                    {
                        "source_record": int(raw["_record"]),
                        "session": raw["sail"],
                        "source_local_time": raw["DateTime_1"],
                        "source_day": local_day,
                        "timestamp_utc_assumed": timestamp.isoformat() if timestamp else "",
                        "latitude_deg": number("latDD"),
                        "longitude_deg": number("longDD"),
                        "source_sog_knots": number("Speed"),
                        "source_distance_m": number("Distance"),
                        "source_duration_s": number("Duration"),
                        "source_cog_deg": number("Bearing"),
                        "source_provenance": raw["Source"],
                        "source_certainty": raw["certainty"],
                        "exclusion_reasons": reasons,
                    }
                )
    return rows


def source_hashes(source: Path) -> dict[str, str]:
    return {
        name: digest(source / name)
        for name in (
            GPS_ARCHIVE,
            WIND_FILE,
            "jarrett-latest.json",
            "ndbc-measurements.html",
            "ndbc-station.html",
        )
    }


def method_hashes() -> dict[str, str]:
    return {
        "scripts/empirical_validation.py": digest(Path(__file__)),
        **{
            p.as_posix(): digest(p)
            for p in sorted(Path("extensions/support/src/sail_support").glob("*.py"))
        },
        "extensions/support/uv.lock": digest(Path("extensions/support/uv.lock")),
    }


def prepare(source: Path, output: Path, plan: EmpiricalPlan) -> dict[str, Any]:
    if (
        not datetime.fromisoformat(plan.wind_train_start)
        < datetime.fromisoformat(plan.wind_test_start)
        < datetime.fromisoformat(plan.wind_test_end)
    ):
        raise ValueError("invalid chronological wind split")
    metadata = json.loads((source / "jarrett-latest.json").read_text())
    file = next(f for f in metadata["files"] if f["key"] == GPS_ARCHIVE)
    if file["checksum"] != "md5:" + hashlib.md5((source / GPS_ARCHIVE).read_bytes()).hexdigest():
        raise ValueError("published archive checksum mismatch")
    if metadata["metadata"]["license"]["id"] != "cc-by-sa-4.0":
        raise ValueError("unexpected dataset licence")
    records = gps_records(source / GPS_ARCHIVE)
    starts: dict[str, str] = {}
    for row in records:
        if row["source_day"] != "unknown":
            starts[row["session"]] = min(
                starts.get(row["session"], row["source_day"]), row["source_day"]
            )
    sessions = sorted(starts, key=lambda s: (starts[s], s))
    n = math.ceil(len(sessions) * plan.gps_train_fraction)
    split = {s: "train" if i < n else "heldout" for i, s in enumerate(sessions)}
    output.mkdir(parents=True, exist_ok=False)
    pq.write_table(pa.Table.from_pylist(records), output / "gps-source-records.parquet")
    write(
        output / "source-snapshot.json",
        {name: Path(name).read_text() for name in method_hashes()},
    )
    registration = {
        "stage": "component_validation_registration",
        "sources": source_hashes(source),
        "source_path": str(source.resolve()),
        "methods": method_hashes(),
        "plan": plan.model_dump(mode="json"),
        "gps_sessions": split,
        "gps_session_start_dates": starts,
        "gps_records": len(records),
        "gps_record_sha256": digest(output / "gps-source-records.parquet"),
        "source_snapshot_sha256": digest(output / "source-snapshot.json"),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_tree": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "timestamp_assumption": (
            "GPS local CET interpreted at fixed UTC+01:00; UTC+02 sensitivity is a "
            "constant shift and leaves relative-speed/projection metrics invariant. "
            "Exact historical UTC is unverified; no forecast/meteorological "
            "alignment."
        ),
        "boat_components": (
            "Polar, through-water speed and tack dynamics unidentifiable; no fit or "
            "change to simulator."
        ),
        "wind_scope": (
            "NOAA buoy 44013 vector wind, 10-minute transitions; in-situ observation "
            "prediction, not historical forecast availability or racecourse wind."
        ),
        "source_licence": (
            "Jarrett CC BY-SA 4.0; NOAA US government observational data, preserve attribution."
        ),
        "primary_wind_comparison": (
            "fitted vector AR(1) minus persistence vector RMSE on held-out days; day bootstrap"
        ),
        "projection_gate": (
            "held-out p99 relative projected/geodesic velocity difference <=1%; same "
            "GPS, numerical consistency not independent instrument accuracy"
        ),
        "screening": (
            "Public schema, timestamps/provenance and licence inspected before "
            "registration; no held-out diagnostic outcome used for fitting."
        ),
    }
    write(output / "registration.json", registration)
    (output / "registration.sha256").write_text(digest(output / "registration.json") + "\n")
    return registration


def verify(output: Path) -> tuple[dict[str, Any], EmpiricalPlan, Path]:
    r = json.loads((output / "registration.json").read_text())
    source = Path(r["source_path"])
    if (
        digest(output / "registration.json") != (output / "registration.sha256").read_text().strip()
        or r["sources"] != source_hashes(source)
        or r["methods"] != method_hashes()
        or r["gps_record_sha256"] != digest(output / "gps-source-records.parquet")
        or r["source_snapshot_sha256"] != digest(output / "source-snapshot.json")
    ):
        raise ValueError("registered data/methods changed")
    return r, EmpiricalPlan.model_validate(r["plan"]), source


def track_validation(
    records: list[dict[str, Any]], registration: dict[str, Any], output: Path, plan: EmpiricalPlan
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    excluded: Counter[str] = Counter()
    breaks: dict[tuple[str, str], int] = defaultdict(int)
    for row in records:
        group = (row.get("session", "unknown"), row.get("source_day", "unknown"))
        if row["exclusion_reasons"]:
            excluded.update(row["exclusion_reasons"])
            breaks[group] += 1
        else:
            groups[group].append({**row, "provenance_segment": breaks[group]})
    summary = []
    geod = Geod(ellps="WGS84")
    projection_errors: dict[str, list[float]] = {"train": [], "heldout": []}
    device_errors: dict[str, list[float]] = {"train": [], "heldout": []}
    all_points = []
    for (session, day), sample in sorted(groups.items()):
        # Keep original row order: the importer detects regressions, rather than sorting them away.
        for chunk_index, begin in enumerate(range(0, len(sample), 10000)):
            chunk = sample[begin : begin + 10000]
            stream = io.StringIO()
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "timestamp_utc",
                    "latitude_deg",
                    "longitude_deg",
                    "device_speed_over_ground_m_s",
                    "_segment",
                ]
            )
            for row in chunk:
                sog = row["source_sog_knots"]
                writer.writerow(
                    [
                        row["timestamp_utc_assumed"],
                        row["latitude_deg"],
                        row["longitude_deg"],
                        sog * 0.5144444444444445 if sog is not None and sog >= 0 else "",
                        row["provenance_segment"],
                    ]
                )
            raw = stream.getvalue().encode()
            identity = f"{session}-{day}-{chunk_index}"
            track = import_track(
                raw,
                "csv",
                session_id=identity,
                config=ImportConfig(
                    nominal_interval_s=plan.gps_nominal_interval_s,
                    gap_s=plan.gps_gap_s,
                    max_ground_speed_m_s=plan.gps_max_speed_m_s,
                ),
                device_metadata={
                    "source": "Greer Jarrett, DOI 10.5281/zenodo.17340505",
                    "boat_scope": "traditional Nordic clinker voyages; rowing/motoring may occur",
                    "clock": "assumed CET +01; no historical forecast alignment",
                    "instrument": "GPS, not hull heading/water speed/true wind/current",
                },
            )
            save_track(track, raw, output / "tracks" / identity)
            split = registration["gps_sessions"][session]
            flags = Counter(flag for p in track.points for flag in p.flags)
            summary.append(
                {
                    "session": session,
                    "source_day": day,
                    "chunk": chunk_index,
                    "split": split,
                    "rows": len(track.points),
                    "accepted": sum(p.accepted for p in track.points),
                    "rejected": sum(not p.accepted for p in track.points),
                    "flags": dict(flags),
                    "segments": len({p.segment for p in track.points if p.accepted}),
                }
            )
            previous = None
            for point, source_row in zip(track.points, chunk, strict=True):
                all_points.append(
                    {
                        **point.model_dump(mode="json"),
                        "session": session,
                        "source_day": day,
                        "split": split,
                        "source_record": source_row["source_record"],
                    }
                )
                if (
                    point.accepted
                    and previous is not None
                    and point.segment == previous.segment
                    and point.derived_speed_over_ground_m_s is not None
                ):
                    assert point.timestamp_utc is not None and previous.timestamp_utc is not None
                    dt = (point.timestamp_utc - previous.timestamp_utc).total_seconds()
                    _, _, distance = geod.inv(
                        previous.longitude_deg,
                        previous.latitude_deg,
                        point.longitude_deg,
                        point.latitude_deg,
                    )
                    speed = distance / dt
                    if speed >= 0.1:
                        projection_errors[split].append(
                            abs(point.derived_speed_over_ground_m_s - speed) / speed
                        )
                    if point.device_speed_over_ground_m_s is not None:
                        device_errors[split].append(
                            point.derived_speed_over_ground_m_s - point.device_speed_over_ground_m_s
                        )
                previous = point if point.accepted else None
    pq.write_table(pa.Table.from_pylist(all_points), output / "gps-processed-points.parquet")
    write(output / "gps-quality-by-day.json", summary)

    def stats(values: list[float]) -> dict[str, Any]:
        return {
            "pairs": len(values),
            "median": float(np.median(values)) if values else None,
            "p99": float(np.quantile(values, 0.99)) if values else None,
            "rmse": math.sqrt(float(np.mean(np.square(values)))) if values else None,
        }

    projection = {s: stats(x) for s, x in projection_errors.items()}
    p99 = projection["heldout"]["p99"]
    return {
        "raw_records": len(records),
        "provenance_excluded_rows": sum(bool(r["exclusion_reasons"]) for r in records),
        "provenance_exclusion_reasons": dict(excluded),
        "processed_rows": len(all_points),
        "accepted_rows": sum(p["accepted"] for p in all_points),
        "rejected_rows": sum(not p["accepted"] for p in all_points),
        "session_splits": registration["gps_sessions"],
        "processed_session_days": len(groups),
        "projection_relative_speed_error": projection,
        "projection_gate_passed": p99 is not None and p99 <= plan.projection_relative_p99_tolerance,
        "projected_minus_source_sog_m_s": {s: stats(x) for s, x in device_errors.items()},
        "independent_instrument_comparison": False,
        "limitations": [
            "Source SOG is GPS-derived too; this is not independent speed truth.",
            "CET clock assumption; absolute UTC remains unverified.",
            "Daily frame/chunk resets break paths explicitly; excluded provenance, "
            "gaps, regressions and rejected rows survive.",
            "No polar/tack fit; apparent-wind notes and sparse heading cannot "
            "identify through-water dynamics.",
        ],
    }


def wind_pairs(raw: bytes, plan: EmpiricalPlan) -> tuple[list[dict[str, Any]], dict[str, int]]:
    text = gzip.decompress(raw).decode("ascii")
    pairs = []
    counts: Counter[str] = Counter()
    previous = None
    start = datetime.fromisoformat(plan.wind_train_start)
    split = datetime.fromisoformat(plan.wind_test_start)
    end = datetime.fromisoformat(plan.wind_test_end)
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        values = line.split()
        year, month, day, hour, minute = map(int, values[:5])
        t = datetime(year, month, day, hour, minute, tzinfo=UTC)
        if not start <= t < end:
            continue
        counts["window_rows"] += 1
        direction, speed = float(values[5]), float(values[6])
        if direction == 999 or speed == 99 or not 0 <= direction <= 360 or speed < 0:
            counts["missing_or_invalid_wind"] += 1
            previous = None
            continue
        vector = np.array(
            [-speed * math.sin(math.radians(direction)), -speed * math.cos(math.radians(direction))]
        )
        if previous is not None:
            old_t, old = previous
            if (t - old_t).total_seconds() == plan.wind_interval_s and t.date() == old_t.date():
                pairs.append(
                    {
                        "timestamp_utc": t.isoformat(),
                        "day": t.date().isoformat(),
                        "split": "train" if t < split else "heldout",
                        "x": old.tolist(),
                        "y": vector.tolist(),
                    }
                )
            else:
                counts["gap_or_day_boundary"] += 1
        previous = t, vector
    return pairs, dict(counts)


def wind_validation(source: Path, output: Path, plan: EmpiricalPlan) -> dict[str, Any]:
    pairs, quality = wind_pairs((source / WIND_FILE).read_bytes(), plan)
    train = [r for r in pairs if r["split"] == "train"]
    test = [r for r in pairs if r["split"] == "heldout"]
    if len(train) < 100 or len(test) < 100:
        raise ValueError("insufficient recorded wind pairs")
    x = np.array([r["x"] for r in train])
    y = np.array([r["y"] for r in train])
    phi = float(
        np.clip(
            np.sum((x - x.mean(axis=0)) * (y - y.mean(axis=0))) / np.sum((x - x.mean(axis=0)) ** 2),
            0,
            0.9999,
        )
    )
    intercept = y.mean(axis=0) - phi * x.mean(axis=0)
    climatology = y.mean(axis=0)
    fit = {
        "phi_600s": phi,
        "intercept_m_s": intercept.tolist(),
        "climatology_air_m_s": climatology.tolist(),
        "equivalent_scalar_kappa_s_inv": -math.log(phi) / plan.wind_interval_s if phi > 0 else None,
        "scope": (
            "fitted station-level 10-minute vector predictor, not the boat filter or "
            "a 1-second racecourse OU calibration"
        ),
    }
    predictors: dict[str, Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]]] = {
        "AR1": lambda v: intercept + phi * v,
        "persistence": lambda v: v,
        "climatology": lambda v: np.tile(climatology, (len(v), 1)),
    }
    tx = np.array([r["x"] for r in test])
    ty = np.array([r["y"] for r in test])
    scores = {}
    details = []
    day_rss: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for name, predict in predictors.items():
        residual = y - predict(x)
        covariance = np.cov(residual.T) + np.eye(2) * 1e-6
        pred = predict(tx)
        error = ty - pred
        squared = np.sum(error**2, axis=1)
        mahal = np.einsum("ij,jk,ik->i", error, np.linalg.inv(covariance), error)
        nll = math.log(2 * math.pi) + 0.5 * np.linalg.slogdet(covariance)[1] + 0.5 * mahal
        speeds = np.linalg.norm(pred, axis=1) - np.linalg.norm(ty, axis=1)
        observed_direction = np.arctan2(-ty[:, 0], -ty[:, 1])
        forecast_direction = np.arctan2(-pred[:, 0], -pred[:, 1])
        angle = (forecast_direction - observed_direction + math.pi) % (2 * math.pi) - math.pi
        directional = np.linalg.norm(ty, axis=1) >= 0.5
        scores[name] = {
            "heldout_pairs": len(test),
            "vector_rmse_m_s": math.sqrt(float(squared.mean())),
            "speed_rmse_m_s": math.sqrt(float(np.mean(speeds**2))),
            "direction_rmse_deg_at_speed_ge_0_5": math.degrees(
                math.sqrt(float(np.mean(angle[directional] ** 2)))
            )
            if directional.any()
            else None,
            "coverage95": float(np.mean(mahal <= 5.991464547)),
            "gaussian_nll_mean": float(nll.mean()),
            "train_residual_covariance_m2_s2": covariance.tolist(),
        }
        for i, row in enumerate(test):
            day_rss[row["day"]][name].append(float(squared[i]))
            details.append(
                {
                    "timestamp_utc": row["timestamp_utc"],
                    "day": row["day"],
                    "predictor": name,
                    "prediction_air_m_s": pred[i].tolist(),
                    "observed_air_m_s": ty[i].tolist(),
                    "squared_error": float(squared[i]),
                    "covered95": bool(mahal[i] <= 5.991464547),
                    "gaussian_nll": float(nll[i]),
                }
            )
    days = sorted(day_rss)
    rng = np.random.default_rng(plan.analysis_seed)
    draw = rng.integers(0, len(days), size=(plan.bootstrap_count, len(days)))
    sizes = np.array([len(day_rss[d]["AR1"]) for d in days])
    delta = []
    sums = {p: np.array([sum(day_rss[d][p]) for d in days]) for p in predictors}
    for ids in draw:
        delta.append(
            math.sqrt(sums["AR1"][ids].sum() / sizes[ids].sum())
            - math.sqrt(sums["persistence"][ids].sum() / sizes[ids].sum())
        )
    pq.write_table(pa.Table.from_pylist(pairs), output / "wind-pairs.parquet")
    pq.write_table(pa.Table.from_pylist(details), output / "wind-heldout-predictions.parquet")
    write(output / "wind-fit.json", fit)
    return {
        "station": "44013",
        "measurement": (
            "8-minute averaged buoy wind, sampled at 10-minute timestamps; FROM true "
            "north, m/s, UTC"
        ),
        "quality": quality,
        "train_pairs": len(train),
        "heldout_pairs": len(test),
        "train_days": len({r["day"] for r in train}),
        "heldout_days": len(days),
        "fit": fit,
        "predictors": scores,
        "AR1_minus_persistence_vector_rmse_m_s": scores["AR1"]["vector_rmse_m_s"]
        - scores["persistence"]["vector_rmse_m_s"],
        "difference95": np.quantile(delta, [0.025, 0.975]).tolist(),
        "bootstrap_count": plan.bootstrap_count,
        "bootstrap_unit": (
            "whole held-out calendar days, all transitions retained within resampled day"
        ),
        "limitations": [
            "Same buoy instrument supplies targets; not hidden instantaneous true wind.",
            "Retrospective observation series, never decision-time historical forecasts.",
            "Serial dependence across days may reduce effective sample size; 30-day "
            "interval is conditional on this season/site.",
            "No local racecourse spatial, minute-scale wind or boat-policy effectiveness claim.",
        ],
    }


def run(output: Path) -> dict[str, Any]:
    registration, plan, source = verify(output)
    destination = output / "evaluation"
    destination.mkdir(exist_ok=False)
    started = time.perf_counter()
    try:
        records = pq.read_table(output / "gps-source-records.parquet").to_pylist()
        gps = track_validation(records, registration, destination, plan)
        wind = wind_validation(source, destination, plan)
        verify(output)
        result = {
            "stage": "recorded_component_validation",
            "registration_sha256": digest(output / "registration.json"),
            "gps": gps,
            "wind": wind,
            "wall_s": time.perf_counter() - started,
            "dependencies": {
                p: version(p) for p in ("numpy", "pyarrow", "pydantic", "pyproj", "gpxpy")
            },
            "attribution": {
                "sailing": (
                    "Greer Jarrett, From the masthead to the map, DOI "
                    "10.5281/zenodo.17340505, CC BY-SA 4.0; GPS adaptation into SI CSV and "
                    "metric frames."
                ),
                "wind": (
                    "NOAA National Data Buoy Center, station 44013 historical standard "
                    "meteorological data, 2025."
                ),
            },
            "boat_validation_status": (
                "Not empirically calibrated: polar, tack loss, current and through-water "
                "speed remain unidentifiable in these public records."
            ),
        }
        write(output / "report.json", result)
        write(
            output / "manifest.json",
            {
                "status": "succeeded",
                "artifacts": {
                    p.relative_to(output).as_posix(): digest(p)
                    for p in sorted(output.rglob("*"))
                    if p.is_file() and p != output / "manifest.json"
                },
            },
        )
        return result
    except BaseException:
        import traceback

        (destination / "error.txt").write_text(traceback.format_exc())
        write(
            output / "manifest.json",
            {
                "status": "failed",
                "artifacts": {
                    p.relative_to(output).as_posix(): digest(p)
                    for p in sorted(output.rglob("*"))
                    if p.is_file() and p != output / "manifest.json"
                },
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "prepare", "run", "verify"))
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/measurements/recorded-validation")
    )
    args = parser.parse_args()
    if args.command == "fetch":
        value = fetch(args.source)
    elif args.command == "prepare":
        value = prepare(args.source, args.output, EmpiricalPlan())
    elif args.command == "run":
        value = run(args.output)
    else:
        registration, _, _ = verify(args.output)
        value = {
            "registration_sha256": digest(args.output / "registration.json"),
            "records": registration["gps_records"],
        }
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
