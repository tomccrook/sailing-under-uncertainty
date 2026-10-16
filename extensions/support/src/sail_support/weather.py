"""Open-Meteo archive adapter with a publication-time information boundary."""

import hashlib
import json
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal
from urllib.parse import urlencode
from urllib.request import urlopen

from pydantic import Field, field_validator, model_validator

from sail_support.contracts import Contract, utc

Kind = Literal["reanalysis", "historical_forecast", "previous_runs", "single_run"]
ENDPOINTS = {
    "reanalysis": "https://archive-api.open-meteo.com/v1/archive",
    "historical_forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast",
    "previous_runs": "https://previous-runs-api.open-meteo.com/v1/forecast",
    "single_run": "https://single-runs-api.open-meteo.com/v1/forecast",
}


class WeatherRequest(Contract):
    kind: Kind
    latitude_deg: float = Field(ge=-90, le=90)
    longitude_deg: float = Field(ge=-180, le=180)
    model: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9_]+$")
    model_version: str = "not supplied by API; requested model identifier only"
    declared_spatial_resolution_m: float = Field(gt=0)
    declared_temporal_resolution_s: float = Field(default=3600, gt=0)
    run_initialisation_utc: datetime | None = None
    known_available_utc: datetime | None = None
    availability_evidence: str | None = None
    assumed_publication_lag_s: float = Field(default=8 * 3600, ge=0)
    start_date: date | None = None
    end_date: date | None = None
    previous_day: int = Field(default=1, ge=1, le=7)

    @field_validator("run_initialisation_utc", "known_available_utc")
    @classmethod
    def aware(cls, value: datetime | None) -> datetime | None:
        return utc(value) if value is not None else None

    @model_validator(mode="after")
    def timing(self) -> "WeatherRequest":
        if self.kind == "single_run":
            if self.run_initialisation_utc is None:
                raise ValueError("Single runs require initialisation time")
            if self.run_initialisation_utc.second or self.run_initialisation_utc.microsecond:
                raise ValueError("Run initialisation must have minute precision")
            if self.known_available_utc is not None:
                if not self.availability_evidence:
                    raise ValueError("Known availability requires provenance")
                if self.known_available_utc < self.run_initialisation_utc:
                    raise ValueError("A run cannot be available before initialisation")
        elif self.run_initialisation_utc or self.known_available_utc:
            raise ValueError("Context products cannot acquire single-run availability metadata")
        if (self.start_date is None) != (self.end_date is None):
            raise ValueError("Date bounds must be supplied together")
        if self.start_date is not None and self.end_date is not None:
            if not 0 <= (self.end_date - self.start_date).days <= 7:
                raise ValueError("A fetch is limited to eight inclusive calendar days")
        return self


class WindPoint(Contract):
    valid_utc: datetime
    wind_speed_m_s: float | None = Field(ge=0)
    wind_from_deg: float | None = Field(ge=0, lt=360)
    flags: tuple[str, ...] = ()

    @field_validator("valid_utc")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        return utc(value)


class WeatherPacket(Contract):
    schema_version: Literal["weather-v1"] = "weather-v1"
    source_evidence: Literal["synthetic_fixture", "downloaded_weather"]
    request: WeatherRequest
    source_url: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieved_utc: datetime
    available_utc: datetime | None
    availability: Literal["known", "assumed_lag", "context_only"]
    returned_latitude_deg: float = Field(ge=-90, le=90)
    returned_longitude_deg: float = Field(ge=-180, le=180)
    points: tuple[WindPoint, ...]
    acknowledgement: str = "Open-Meteo; upstream model identified in request; data CC BY 4.0"
    limits: str = "Regional/hourly broad context; no measured racecourse field or covariance"

    @field_validator("retrieved_utc", "available_utc")
    @classmethod
    def aware(cls, value: datetime | None) -> datetime | None:
        return utc(value) if value is not None else None

    @model_validator(mode="after")
    def invariant(self) -> "WeatherPacket":
        request = self.request
        if request.kind == "single_run":
            assert request.run_initialisation_utc is not None
            expected = request.known_available_utc or (
                request.run_initialisation_utc
                + timedelta(seconds=request.assumed_publication_lag_s)
            )
            expected_kind = "known" if request.known_available_utc else "assumed_lag"
            if self.available_utc != expected or self.availability != expected_kind:
                raise ValueError("Packet availability does not match declared publication evidence")
        elif self.available_utc is not None or self.availability != "context_only":
            raise ValueError("Retrospective/stitched/fixed-lead products are context only")
        if any(
            a.valid_utc >= b.valid_utc for a, b in zip(self.points, self.points[1:], strict=False)
        ):
            raise ValueError("Forecast validity times must strictly increase")
        return self


def request_url(request: WeatherRequest) -> str:
    suffix = f"_previous_day{request.previous_day}" if request.kind == "previous_runs" else ""
    params = {
        "latitude": str(request.latitude_deg),
        "longitude": str(request.longitude_deg),
        "models": request.model,
        "hourly": f"wind_speed_10m{suffix},wind_direction_10m{suffix}",
        "wind_speed_unit": "ms",
        "timezone": "GMT",
    }
    if request.run_initialisation_utc is not None:
        params["run"] = request.run_initialisation_utc.strftime("%Y-%m-%dT%H:%M")
    if request.start_date is not None and request.end_date is not None:
        params.update(start_date=str(request.start_date), end_date=str(request.end_date))
    return ENDPOINTS[request.kind] + "?" + urlencode(params)


def parse_weather(
    raw: bytes,
    request: WeatherRequest,
    *,
    source_evidence: Literal["synthetic_fixture", "downloaded_weather"],
    retrieved_utc: datetime,
) -> WeatherPacket:
    if not raw or len(raw) > 5_000_000:
        raise ValueError("Weather payload is empty or exceeds the byte limit")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("error"):
        raise ValueError("Weather API returned an error or unsupported object")
    if payload.get("utc_offset_seconds") != 0 or payload.get("timezone") not in {"GMT", "UTC"}:
        raise ValueError("Weather response must explicitly declare UTC")
    suffix = f"_previous_day{request.previous_day}" if request.kind == "previous_runs" else ""
    speed_key, angle_key = f"wind_speed_10m{suffix}", f"wind_direction_10m{suffix}"
    units = payload["hourly_units"]
    factor = {"m/s": 1, "km/h": 1 / 3.6, "kn": 0.5144444444444445}.get(units.get(speed_key))
    if factor is None or units.get(angle_key) != "°":
        raise ValueError("Unrecognised wind units")
    hourly = payload["hourly"]
    times, speeds, angles = hourly["time"], hourly[speed_key], hourly[angle_key]
    if not times or len(times) > 10_000 or not len(times) == len(speeds) == len(angles):
        raise ValueError("Weather arrays have invalid lengths")
    points = []
    for time, speed, angle in zip(times, speeds, angles, strict=True):
        parsed = datetime.fromisoformat(time.replace("Z", "+00:00"))
        # Unzoned strings are permitted only under the response's explicit UTC declaration.
        valid = utc(parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed)
        flags = []
        if speed is None or angle is None:
            flags.append("missing_wind")
        if request.run_initialisation_utc is not None and valid < request.run_initialisation_utc:
            flags.append("pre_initialisation")
        speed_si = None if speed is None else float(speed) * factor
        direction = None if angle is None else float(angle)
        if direction == 360:
            direction = 0.0
        if (speed_si is not None and not math.isfinite(speed_si)) or (
            direction is not None and not math.isfinite(direction)
        ):
            raise ValueError("Nonfinite weather values")
        points.append(
            WindPoint(
                valid_utc=valid,
                wind_speed_m_s=speed_si,
                wind_from_deg=direction,
                flags=tuple(flags),
            )
        )
    available = None
    availability: Literal["known", "assumed_lag", "context_only"] = "context_only"
    if request.run_initialisation_utc is not None:
        available = request.known_available_utc or (
            request.run_initialisation_utc + timedelta(seconds=request.assumed_publication_lag_s)
        )
        availability = "known" if request.known_available_utc else "assumed_lag"
    return WeatherPacket(
        source_evidence=source_evidence,
        request=request,
        source_url=request_url(request),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        retrieved_utc=utc(retrieved_utc),
        available_utc=available,
        availability=availability,
        returned_latitude_deg=payload["latitude"],
        returned_longitude_deg=payload["longitude"],
        points=tuple(points),
    )


def available_forecast(packet: WeatherPacket, decision_utc: datetime) -> tuple[WindPoint, ...]:
    """A strict snapshot; caller must retain the known/assumed availability label."""
    decision = utc(decision_utc)
    if packet.request.kind != "single_run":
        raise ValueError("Context products cannot be used as a decision-time forecast")
    assert packet.available_utc is not None
    if packet.available_utc > decision:
        raise ValueError("Forecast run was not yet available at the decision")
    return tuple(
        point for point in packet.points if point.valid_utc >= decision and not point.flags
    )


def fetch_weather(request: WeatherRequest, output: Path) -> WeatherPacket:
    """One bounded request, no silent retry or synthetic fallback; preserves source bytes."""
    output.mkdir(parents=True, exist_ok=False)
    url = request_url(request)
    (output / "request.json").write_text(request.model_dump_json(indent=2) + "\n")
    try:
        with urlopen(url, timeout=15) as response:  # noqa: S310 - fixed allowlisted HTTPS endpoints
            raw = response.read(5_000_001)
        (output / "source.json").write_bytes(raw)
        packet = parse_weather(
            raw, request, source_evidence="downloaded_weather", retrieved_utc=datetime.now(UTC)
        )
        (output / "weather.json").write_text(packet.model_dump_json(indent=2) + "\n")
        manifest = {
            "status": "succeeded",
            "source_url": url,
            "files": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(output.iterdir())
                if path.is_file()
            },
        }
    except Exception as error:
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "source_url": url,
                    "error": f"{type(error).__name__}: {error}",
                },
                indent=2,
            )
            + "\n"
        )
        raise
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return packet
