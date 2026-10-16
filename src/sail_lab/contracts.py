"""Policy-facing records and offline diagnostics are deliberately separate types."""

from typing import Literal, Self

from pydantic import Field, model_validator

from sail_lab.config import Contract, CourseConfig, Nonnegative, PolicyConfig, Positive, Vector

EpisodeStatus = Literal[
    "running",
    "finished",
    "deadline_exceeded",
    "boundary_violation",
    "invalid_numeric_state",
    "cancelled_experiment",
    "implementation_error",
]


class Action(Contract):
    target_heading_rad: float


class WindMeasurement(Contract):
    measured_s: Nonnegative
    available_s: Nonnegative
    air_m_s: Vector
    direction_noise_deg: Nonnegative
    speed_noise_m_s: Nonnegative

    @model_validator(mode="after")
    def causal(self) -> Self:
        if self.available_s < self.measured_s:
            raise ValueError("measurement cannot be available before acquisition")
        return self


class WindBelief(Contract):
    elapsed_s: Nonnegative
    mean_air_m_s: Vector
    covariance_m2_s2: tuple[Vector, Vector]
    model_version: Literal["vector-random-walk-v1"] = "vector-random-walk-v1"

    @model_validator(mode="after")
    def positive_semidefinite(self) -> Self:
        (a, b), (c, d) = self.covariance_m2_s2
        if abs(b - c) > 1e-8 or min(a, d) < 0 or a * d - b * c < -1e-8:
            raise ValueError("wind covariance must be symmetric positive semidefinite")
        return self


class ForecastPoint(Contract):
    valid_s: Nonnegative
    wind_from_rad: float
    speed_m_s: Nonnegative
    direction_std_rad: Positive
    speed_std_m_s: Positive


class ForecastPacket(Contract):
    issued_s: Nonnegative
    available_s: Nonnegative
    horizon_s: Positive
    spatial_resolution_m: Positive
    station_xy_m: Vector
    model: Literal["persistence", "mean_reversion"]
    distribution: Literal["local-angle-speed-gaussian-v1"] = "local-angle-speed-gaussian-v1"
    points: tuple[ForecastPoint, ...]

    @model_validator(mode="after")
    def causal(self) -> Self:
        if self.available_s < self.issued_s:
            raise ValueError("forecast availability cannot precede issue")
        if not self.points:
            raise ValueError("forecast requires points")
        for point in self.points:
            if not self.issued_s <= point.valid_s <= self.issued_s + self.horizon_s + 1e-8:
                raise ValueError("forecast valid time outside declared horizon")
        if any(
            b.valid_s <= a.valid_s for a, b in zip(self.points[:-1], self.points[1:], strict=True)
        ):
            raise ValueError("forecast valid times must be strictly increasing")
        return self


class Observation(Contract):
    elapsed_s: Nonnegative
    position_xy_m: Vector
    heading_rad: float
    speed_water_m_s: Nonnegative
    target_xy_m: Vector
    remaining_s: Nonnegative
    bounds_xy_m: tuple[float, float, float, float] = (-600, 600, -100, 1200)
    wind_air_m_s: Vector
    known_current_m_s: Vector
    measurement_time_s: Nonnegative
    measurement_valid: bool
    manoeuvre_active: bool
    previous_action_rad: float
    information_set: Literal["perfect_local_known_current", "noisy_local_known_current"] = (
        "perfect_local_known_current"
    )
    measurement_available_s: Nonnegative = 0.0
    latest_sensor_dropout: bool = False
    belief: WindBelief | None = None
    forecast: ForecastPacket | None = None
    executed_target_rad: float = 0.0
    manoeuvre_remaining_s: Nonnegative = 0.0

    @model_validator(mode="after")
    def causal(self) -> Self:
        if self.measurement_time_s > self.elapsed_s + 1e-8:
            raise ValueError("measurement timestamp is in the future")
        if self.measurement_time_s > self.measurement_available_s + 1e-8:
            raise ValueError("measurement availability cannot precede acquisition")
        if self.measurement_available_s > self.elapsed_s + 1e-8:
            raise ValueError("measurement not yet available")
        if self.forecast is not None and self.forecast.available_s > self.elapsed_s + 1e-8:
            raise ValueError("forecast not yet available")
        if self.belief is not None and abs(self.belief.elapsed_s - self.elapsed_s) > 1e-8:
            raise ValueError("belief timestamp must match observation")
        return self


class EpisodeSummary(Contract):
    episode_id: str = "fixture-0"
    policy_id: str = "polar_heuristic"
    status: EpisodeStatus
    elapsed_s: Nonnegative
    distance_m: Nonnegative
    tack_count: int = Field(ge=0)
    manoeuvre_s: Nonnegative
    decisions: int = Field(ge=0)
    policy_compute_s: Nonnegative
    simulation_compute_s: Nonnegative


class ReplaySample(Contract):
    elapsed_s: Nonnegative
    position_xy_m: Vector
    heading_rad: float
    speed_water_m_s: Nonnegative
    tack_count: int = Field(ge=0)
    manoeuvre_active: bool
    requested_heading_rad: float
    executed_target_rad: float
    observation: Observation
    diagnostic_wind_air_m_s: Vector
    diagnostic_current_m_s: Vector


class ReplayBundle(Contract):
    schema_version: Literal[1] = 1
    run_id: str
    evidence_level: Literal["synthetic"] = "synthetic"
    config_hash: str
    policy_label: str | None = None
    policy_parameters: PolicyConfig | None = None
    world_family: Literal["constant", "oscillating", "ou", "regime", "spatial"] = "constant"
    stage: Literal["fixture", "development", "locked"] = "fixture"
    course: CourseConfig
    summary: EpisodeSummary
    samples: list[ReplaySample]

    @model_validator(mode="after")
    def consistent_recording(self) -> Self:
        if not self.samples or self.samples[0].elapsed_s != 0:
            raise ValueError("replay must start at elapsed time zero")
        for previous, following in zip(self.samples[:-1], self.samples[1:], strict=True):
            if following.elapsed_s <= previous.elapsed_s:
                raise ValueError("replay sample times must be strictly increasing")
        last = self.samples[-1]
        if abs(last.elapsed_s - self.summary.elapsed_s) > 1e-8:
            raise ValueError("replay terminal time disagrees with summary")
        if last.tack_count != self.summary.tack_count:
            raise ValueError("replay terminal tack count disagrees with summary")
        for sample in self.samples:
            if sample.observation.measurement_time_s > sample.elapsed_s + 1e-8:
                raise ValueError("measurement timestamp is in the future")
        return self
