"""Loss-aware GPX/CSV ingestion: UTC clocks, WGS84 projection, disconnected gaps."""

import csv
import hashlib
import io
import json
import math
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import gpxpy
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import CRS, Transformer

from sail_support.contracts import Frame, ImportConfig, Track, TrackPoint, utc

INSTRUMENTS = (
    "heading_deg",
    "wind_from_deg",
    "wind_speed_m_s",
    "current_east_m_s",
    "current_north_m_s",
    "speed_water_m_s",
    "device_speed_over_ground_m_s",
)
DEFAULT_IMPORT_CONFIG = ImportConfig()


def _rows(raw: bytes, source_format: Literal["csv", "gpx"]) -> list[dict[str, str]]:
    text = raw.decode("utf-8-sig")
    if source_format == "csv":
        reader = csv.DictReader(io.StringIO(text))
        required = {"timestamp_utc", "latitude_deg", "longitude_deg"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"CSV requires columns {sorted(required)}; SI units are explicit")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Duplicate CSV columns")
        result = []
        for row in reader:
            if None in row:
                raise ValueError("CSV row contains more fields than its header")
            result.append({key: value or "" for key, value in row.items()})
        return result
    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        raise ValueError("GPX document type and entity declarations are prohibited")
    try:
        document = gpxpy.parse(text)
    except gpxpy.gpx.GPXException as error:
        raise ValueError(f"Invalid GPX: {error}") from error
    result = []
    for segment_index, segment in enumerate(
        segment for track in document.tracks for segment in track.segments
    ):
        for point in segment.points:
            row = {
                "timestamp_utc": point.time.isoformat() if point.time else "",
                "latitude_deg": str(point.latitude),
                "longitude_deg": str(point.longitude),
                "_segment": str(segment_index),
            }
            # GPX course is GPS course over ground, not instrumented hull heading.
            for extension in point.extensions:
                for node in extension.iter():
                    name = node.tag.rsplit("}", 1)[-1]
                    if name in INSTRUMENTS and node.text is not None:
                        row[name] = node.text.strip()
            result.append(row)
    if not result:
        raise ValueError("GPX contains no track points")
    return result


def _number(row: dict[str, str], name: str, flags: list[str]) -> float | None:
    if not row.get(name):
        return None
    try:
        value = float(row[name])
        if not math.isfinite(value):
            raise ValueError("nonfinite")
        if name in {"heading_deg", "wind_from_deg"} and not 0 <= value < 360:
            raise ValueError("angle outside declared range")
        if "speed" in name and value < 0:
            raise ValueError("negative speed")
        return value
    except ValueError:
        flags.append(f"invalid_{name}")
        return None


def import_track(
    raw: bytes,
    source_format: Literal["csv", "gpx"],
    *,
    session_id: str,
    evidence: Literal["synthetic_fixture", "unvalidated_measurement"] = "unvalidated_measurement",
    config: ImportConfig = DEFAULT_IMPORT_CONFIG,
    imported_utc: datetime | None = None,
    device_metadata: dict[str, str] | None = None,
) -> Track:
    if not raw or len(raw) > config.max_bytes:
        raise ValueError("Input is empty or exceeds the byte limit")
    rows = _rows(raw, source_format)
    if not rows or len(rows) > config.max_rows:
        raise ValueError("Input is empty or exceeds the row limit")
    points: list[TrackPoint] = []
    frame: Frame | None = None
    transform: Transformer | None = None
    last: TrackPoint | None = None
    contiguous = False
    segment_id = -1
    last_source_segment: str | None = None
    measured: set[str] = {"UTC timestamp", "GPS latitude/longitude"}
    for index, row in enumerate(rows):
        flags: list[str] = []
        try:
            timestamp = utc(row["timestamp_utc"])
        except ValueError:
            timestamp = None
            flags.append("invalid_timestamp_utc")
        lat = _number(row, "latitude_deg", flags)
        lon = _number(row, "longitude_deg", flags)
        if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            flags.append("invalid_coordinates")
        instruments = {name: _number(row, name, flags) for name in INSTRUMENTS}
        measured.update(name for name, value in instruments.items() if value is not None)
        if any(instruments[name] is None for name in INSTRUMENTS[:6]):
            flags.append("incomplete_instruments")
        valid = timestamp is not None and "invalid_coordinates" not in flags
        east: float | None = None
        north: float | None = None
        speed: float | None = None
        if valid and lat is not None and lon is not None:
            if frame is None:
                frame = Frame(latitude_deg=lat, longitude_deg=lon)
                transform = Transformer.from_crs(
                    "EPSG:4326",
                    CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m"),
                    always_xy=True,
                )
            assert transform is not None
            east, north = transform.transform(lon, lat)
            if not math.isfinite(east) or not math.isfinite(north):
                flags.append("invalid_projection")
                east = north = None
                valid = False
            elif math.hypot(east, north) > config.max_radius_m:
                flags.append("outside_local_frame")
                valid = False
            if last is not None and timestamp is not None and last.timestamp_utc is not None:
                dt = (timestamp - last.timestamp_utc).total_seconds()
                if dt <= 0:
                    flags.append("duplicate_time" if dt == 0 else "clock_regression")
                    valid = False
                elif east is not None and north is not None:
                    assert last.east_m is not None and last.north_m is not None
                    displacement = math.hypot(east - last.east_m, north - last.north_m)
                    if displacement / dt > config.max_ground_speed_m_s:
                        flags.append("implausible_ground_speed")
                        valid = False
                    if dt > config.gap_s:
                        flags.append("gap_before")
                        contiguous = False
                    elif abs(dt - config.nominal_interval_s) > config.nominal_interval_s * 0.25:
                        flags.append("irregular_sampling")
                    if valid and contiguous and row.get("_segment") == last_source_segment:
                        speed = displacement / dt
        if valid:
            if row.get("_segment") != last_source_segment:
                contiguous = False
                speed = None
            if not contiguous:
                segment_id += 1
            point = TrackPoint(
                row=index,
                timestamp_utc=timestamp,
                latitude_deg=lat,
                longitude_deg=lon,
                east_m=east,
                north_m=north,
                segment=segment_id,
                accepted=True,
                flags=tuple(flags),
                derived_speed_over_ground_m_s=speed,
                heading_deg=instruments["heading_deg"],
                wind_from_deg=instruments["wind_from_deg"],
                wind_speed_m_s=instruments["wind_speed_m_s"],
                current_east_m_s=instruments["current_east_m_s"],
                current_north_m_s=instruments["current_north_m_s"],
                speed_water_m_s=instruments["speed_water_m_s"],
                device_speed_over_ground_m_s=instruments["device_speed_over_ground_m_s"],
            )
            last = point
            last_source_segment = row.get("_segment")
            contiguous = True
        else:
            contiguous = False
            point = TrackPoint(
                row=index,
                timestamp_utc=timestamp,
                latitude_deg=lat,
                longitude_deg=lon,
                east_m=east,
                north_m=north,
                flags=tuple(flags),
                heading_deg=instruments["heading_deg"],
                wind_from_deg=instruments["wind_from_deg"],
                wind_speed_m_s=instruments["wind_speed_m_s"],
                current_east_m_s=instruments["current_east_m_s"],
                current_north_m_s=instruments["current_north_m_s"],
                speed_water_m_s=instruments["speed_water_m_s"],
                device_speed_over_ground_m_s=instruments["device_speed_over_ground_m_s"],
            )
        points.append(point)
    return Track(
        source_sha256=hashlib.sha256(raw).hexdigest(),
        source_format=source_format,
        session_id=session_id,
        evidence=evidence,
        imported_utc=imported_utc or datetime.now(UTC),
        config=config,
        frame=frame,
        points=tuple(points),
        measured=tuple(sorted(measured)),
        device_metadata=device_metadata or {},
    )


def save_track(track: Track, raw: bytes, output: Path) -> Path:
    """Exclusive artifact directory: raw plus JSON/Parquet, hashes and tool versions."""
    if hashlib.sha256(raw).hexdigest() != track.source_sha256:
        raise ValueError("Raw source hash differs from processed track")
    output.mkdir(parents=True, exist_ok=False)
    (output / f"source.{track.source_format}").write_bytes(raw)
    (output / "track.json").write_text(track.model_dump_json(indent=2) + "\n")
    rows = [point.model_dump(mode="json") for point in track.points]
    pq.write_table(pa.Table.from_pylist(rows), output / "points.parquet")
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    manifest = {
        "status": "succeeded",
        "source_sha256": track.source_sha256,
        "source_format": track.source_format,
        "evidence": track.evidence,
        "session_id": track.session_id,
        "imported_utc": track.imported_utc.isoformat(),
        "config": track.config.model_dump(),
        "files": hashes,
        "accepted": sum(point.accepted for point in track.points),
        "rejected": sum(not point.accepted for point in track.points),
        "python": platform.python_version(),
        "dependencies": {
            name: version(name) for name in ("pyproj", "gpxpy", "pyarrow", "pydantic")
        },
        "support_source": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "dependency_lock_sha256": hashlib.sha256(
            (Path(__file__).parents[2] / "uv.lock").read_bytes()
        ).hexdigest()
        if (Path(__file__).parents[2] / "uv.lock").exists()
        else None,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output
