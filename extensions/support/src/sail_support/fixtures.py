"""Explicit synthetic instruments, independent-day timing and quality corruptions."""

import csv
import io
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pyproj import CRS, Transformer

from sail_support.calibration import fit_timing
from sail_support.tracks import import_track, save_track
from sail_support.weather import WeatherRequest, parse_weather


def synthetic_csv(day: int, duration_s: int, *, corrupt: bool = False) -> bytes:
    inverse = Transformer.from_crs(
        CRS.from_proj4("+proj=aeqd +lat_0=50.7 +lon_0=-1.3 +datum=WGS84 +units=m"),
        "EPSG:4326",
        always_xy=True,
    )
    output = io.StringIO()
    columns = [
        "timestamp_utc",
        "latitude_deg",
        "longitude_deg",
        "heading_deg",
        "wind_from_deg",
        "wind_speed_m_s",
        "current_east_m_s",
        "current_north_m_s",
        "speed_water_m_s",
    ]
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    origin = datetime(2026, 1, day, 12, tzinfo=UTC)
    east = north = 0.0
    for second in range(181):
        heading = 50.0
        if 40 < second <= 40 + duration_s:
            heading = 50 - 100 * (second - 40) / duration_s
        elif 40 + duration_s < second <= 110:
            heading = -50
        elif 110 < second <= 110 + duration_s:
            heading = -50 + 100 * (second - 110) / duration_s
        speed = 2 if (40 < second < 40 + duration_s or 110 < second < 110 + duration_s) else 3
        if second:
            east += speed * math.sin(math.radians(heading))
            north += speed * math.cos(math.radians(heading))
        lon, lat = inverse.transform(east, north)
        row = dict(
            zip(
                columns,
                [
                    (origin + timedelta(seconds=second)).isoformat(),
                    lat,
                    lon,
                    heading % 360,
                    0,
                    6,
                    0,
                    0,
                    speed,
                ],
                strict=True,
            )
        )
        if corrupt and second == 5:
            writer.writerow(row)  # Duplicate is retained and disconnects the path.
        if corrupt and second == 8:
            row["timestamp_utc"] = (origin + timedelta(seconds=6)).isoformat()
        if corrupt and second == 15:
            row["latitude_deg"] = 50.75  # Implausible GPS jump, not a physical manoeuvre.
        if corrupt and 160 < second < 173:
            continue
        writer.writerow(row)
    return output.getvalue().encode()


def make_measurement_fixture(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    tracks = []
    for day, duration in enumerate((10, 12, 14, 16, 18, 20), start=1):
        raw = synthetic_csv(day, duration, corrupt=True)
        track = import_track(
            raw, "csv", session_id=f"synthetic-day-{day}", evidence="synthetic_fixture"
        )
        save_track(track, raw, output / track.session_id)
        tracks.append(track)
    report = fit_timing(tuple(tracks))
    (output / "timing.json").write_text(report.model_dump_json(indent=2) + "\n")
    run = datetime(2026, 1, 1, tzinfo=UTC)
    weather_raw = json.dumps(
        {
            "latitude": 50.75,
            "longitude": -1.25,
            "timezone": "GMT",
            "utc_offset_seconds": 0,
            "hourly_units": {"wind_speed_10m": "m/s", "wind_direction_10m": "°"},
            "hourly": {
                "time": [(run + timedelta(hours=hour)).isoformat() for hour in range(8, 16)],
                "wind_speed_10m": [6, 6.2, 5.7, 6, 5.8, 6.3, None, 6],
                "wind_direction_10m": [0, 3, 8, 5, 0, 10, None, 15],
            },
        }
    ).encode()
    weather = parse_weather(
        weather_raw,
        WeatherRequest(
            kind="single_run",
            latitude_deg=50.7,
            longitude_deg=-1.3,
            model="ecmwf_ifs",
            run_initialisation_utc=run,
            declared_spatial_resolution_m=9000,
        ),
        source_evidence="synthetic_fixture",
        retrieved_utc=datetime.now(UTC),
    )
    (output / "weather-source.json").write_bytes(weather_raw)
    (output / "weather.json").write_text(weather.model_dump_json(indent=2) + "\n")
    return output
