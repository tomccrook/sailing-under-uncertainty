"""Independent, finite-valued measurement contracts; no simulator truth fields."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def utc(value: str | datetime) -> datetime:
    parsed = (
        datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timestamp must carry an explicit UTC offset")
    return parsed.astimezone(UTC)


class ImportConfig(Contract):
    nominal_interval_s: float = Field(default=1, gt=0)
    gap_s: float = Field(default=10, gt=0)
    max_ground_speed_m_s: float = Field(default=15, gt=0)
    max_radius_m: float = Field(default=100_000, gt=0, le=100_000)
    max_bytes: int = Field(default=5_000_000, gt=0, le=5_000_000)
    max_rows: int = Field(default=10_000, gt=0, le=10_000)


class TrackPoint(Contract):
    row: int = Field(ge=0)
    timestamp_utc: datetime | None
    latitude_deg: float | None
    longitude_deg: float | None
    east_m: float | None = None
    north_m: float | None = None
    segment: int | None = None
    accepted: bool = False
    flags: tuple[str, ...] = ()
    heading_deg: float | None = Field(default=None, ge=0, lt=360)
    wind_from_deg: float | None = Field(default=None, ge=0, lt=360)
    wind_speed_m_s: float | None = Field(default=None, ge=0)
    current_east_m_s: float | None = None
    current_north_m_s: float | None = None
    speed_water_m_s: float | None = Field(default=None, ge=0)
    device_speed_over_ground_m_s: float | None = Field(default=None, ge=0)
    derived_speed_over_ground_m_s: float | None = Field(default=None, ge=0)

    @field_validator("timestamp_utc")
    @classmethod
    def aware(cls, value: datetime | None) -> datetime | None:
        return utc(value) if value is not None else None


class Frame(Contract):
    latitude_deg: float
    longitude_deg: float
    projection: Literal["WGS84 azimuthal equidistant"] = "WGS84 azimuthal equidistant"
    axes: Literal["east_m,north_m"] = "east_m,north_m"


class Track(Contract):
    schema_version: Literal["track-v1"] = "track-v1"
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_format: Literal["csv", "gpx"]
    session_id: str = Field(min_length=1, max_length=100, pattern=r"^[\w.-]+$")
    evidence: Literal["synthetic_fixture", "unvalidated_measurement"]
    imported_utc: datetime
    config: ImportConfig
    device_metadata: dict[str, str] = Field(default_factory=dict)
    frame: Frame | None
    points: tuple[TrackPoint, ...]
    units: Literal["UTC,metres,seconds,m/s,degrees clockwise from north; wind FROM"] = (
        "UTC,metres,seconds,m/s,degrees clockwise from north; wind FROM"
    )
    measured: tuple[str, ...]
    derived: tuple[str, ...] = ("local coordinates", "GPS speed over ground")
    assumed: tuple[str, ...] = ("quality thresholds", "local projection", "device synchronisation")

    @field_validator("imported_utc")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        return utc(value)
