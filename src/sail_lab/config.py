"""Validated SI configuration. Unknown fields and nonfinite numbers are errors."""

import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Positive = Annotated[float, Field(gt=0)]
Nonnegative = Annotated[float, Field(ge=0)]
Vector = tuple[float, float]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class BoatConfig(Contract):
    model_id: Literal["synthetic-displacement-v1"] = "synthetic-displacement-v1"
    evidence_level: Literal["synthetic"] = "synthetic"
    class_slot: Literal["illustrative", "mirror_uncalibrated"] = "illustrative"
    no_go_deg: Annotated[float, Field(gt=0, lt=90)] = 35.0
    turn_rate_deg_s: Positive = 8.0
    acceleration_tau_s: Positive = 5.0
    minimum_tack_s: Positive = 8.0
    polar_scale: Positive = 1.0


class CourseConfig(Contract):
    coordinate_frame: Literal["local_east_north_m"] = "local_east_north_m"
    start_xy_m: Vector = (0.0, 0.0)
    initial_heading_deg: float = 50.0
    target_xy_m: Vector = (0.0, 1000.0)
    finish_radius_m: Positive = 10.0
    bounds_xy_m: tuple[float, float, float, float] = (-600.0, 600.0, -100.0, 1200.0)
    deadline_s: Positive = 1800.0

    @model_validator(mode="after")
    def geometry_is_valid(self) -> Self:
        left, right, bottom, top = self.bounds_xy_m
        if not left < right or not bottom < top:
            raise ValueError("bounds must be (xmin, xmax, ymin, ymax), strictly increasing")
        for x, y in (self.start_xy_m, self.target_xy_m):
            if not left < x < right or not bottom < y < top:
                raise ValueError("start and target must be strictly inside bounds")
        if math.dist(self.start_xy_m, self.target_xy_m) <= self.finish_radius_m:
            raise ValueError("start must be outside finish region")
        x, y = self.target_xy_m
        if min(x - left, right - x, y - bottom, top - y) <= self.finish_radius_m:
            raise ValueError("finish region must be strictly inside bounds")
        return self


class ClockConfig(Contract):
    physics_dt_s: Positive = 1.0
    policy_interval_s: Positive = 10.0
    sensor_interval_s: Positive = 2.0
    forecast_interval_s: Positive = 30.0

    @model_validator(mode="after")
    def clock_ratios(self) -> Self:
        for interval in (self.policy_interval_s, self.sensor_interval_s, self.forecast_interval_s):
            if not math.isclose(interval / self.physics_dt_s, round(interval / self.physics_dt_s)):
                raise ValueError("all clocks must be integral multiples of physics_dt_s")
        return self


class WorldConfig(Contract):
    model_version: Literal["frozen-wind-v1"] = "frozen-wind-v1"
    family: Literal["constant", "oscillating", "ou", "regime", "spatial"] = "constant"
    wind_from_deg: float = 0.0
    wind_speed_m_s: Nonnegative = 6.0
    current_ground_m_s: Vector = (0.0, 0.0)
    grid_dt_s: Positive = 2.0
    direction_kappa_s_inv: Positive = 0.02
    direction_sigma_deg_sqrt_s: Nonnegative = 1.2
    log_speed_kappa_s_inv: Positive = 0.03
    log_speed_sigma_sqrt_s: Nonnegative = 0.025
    max_direction_deviation_deg: Annotated[float, Field(gt=0, le=80)] = 60.0
    max_speed_m_s: Positive = 12.0
    oscillation_amplitude_deg: Nonnegative = 15.0
    oscillation_period_s: Positive = 240.0
    regime_shift_deg: Nonnegative = 20.0
    regime_mean_duration_s: Positive = 240.0
    spatial_amplitude_deg: Nonnegative = 12.0
    spatial_length_m: Positive = 600.0

    @model_validator(mode="after")
    def positive_stochastic_speed(self) -> Self:
        if self.family != "constant" and self.wind_speed_m_s <= 0:
            raise ValueError("nonconstant families require a positive reference wind speed")
        return self


class SensorConfig(Contract):
    information_set: Literal["perfect_local_known_current", "noisy_local_known_current"] = (
        "perfect_local_known_current"
    )
    allow_diagnostic_truth: Literal[False] = False
    direction_noise_deg: Nonnegative = 0.0
    speed_noise_m_s: Nonnegative = 0.0
    direction_bias_deg: float = 0.0
    speed_bias_m_s: float = 0.0
    latency_s: Nonnegative = 0.0
    dropout_probability: Annotated[float, Field(ge=0, le=1)] = 0.0
    max_measurement_age_s: Positive = 30.0

    @model_validator(mode="after")
    def information_label(self) -> Self:
        if self.information_set == "perfect_local_known_current" and any(
            (
                self.direction_noise_deg,
                self.speed_noise_m_s,
                self.direction_bias_deg,
                self.speed_bias_m_s,
                self.latency_s,
                self.dropout_probability,
            )
        ):
            raise ValueError("perturbed instruments must use the noisy information label")
        return self


class BeliefConfig(Contract):
    model_version: Literal["vector-random-walk-v1"] = "vector-random-walk-v1"
    prior_air_m_s: Vector = (0.0, -6.0)
    prior_std_m_s: Positive = 2.0
    process_variance_m2_s3: Nonnegative = 0.015
    instrument_variance_floor_m2_s2: Positive = 1e-4


class ForecastConfig(Contract):
    enabled: bool = False
    model: Literal["persistence", "mean_reversion"] = "mean_reversion"
    station_xy_m: Vector = (0.0, 500.0)
    spatial_resolution_m: Positive = 1000.0
    availability_delay_s: Nonnegative = 4.0
    horizon_s: Positive = 120.0
    point_interval_s: Positive = 10.0
    reference_from_deg: float = 0.0
    reference_speed_m_s: Nonnegative = 6.0
    direction_kappa_s_inv: Positive = 0.02
    direction_sigma_deg_sqrt_s: Nonnegative = 1.2
    speed_kappa_s_inv: Positive = 0.03
    speed_sigma_m_s_sqrt_s: Nonnegative = 0.15
    station_direction_noise_deg: Nonnegative = 2.0
    station_speed_noise_m_s: Nonnegative = 0.15
    direction_bias_deg: float = 0.0
    speed_bias_m_s: float = 0.0
    uncertainty_scale: Positive = 1.0

    @model_validator(mode="after")
    def horizon_clock(self) -> Self:
        if not math.isclose(
            self.horizon_s / self.point_interval_s, round(self.horizon_s / self.point_interval_s)
        ):
            raise ValueError("forecast horizon must be a multiple of point interval")
        return self


class PolicyConfig(Contract):
    name: Literal["polar_heuristic", "deterministic_mpc", "scenario_mpc"] = "polar_heuristic"
    wind_input: Literal["raw", "belief"] = "raw"
    deterministic_sensor_model: Literal["expected", "sampled"] = "expected"
    planning_horizon_s: Positive = 60.0
    scenario_count: Annotated[int, Field(ge=2, le=128)] = 8
    risk_lambda: Annotated[float, Field(ge=0, le=1)] = 0.25
    cvar_alpha: Annotated[float, Field(gt=0, lt=1)] = 0.95
    failure_cost_s: Positive = 2400.0
    initial_side: Literal[-1, 1] = 1
    commitment_s: Nonnegative = 20.0
    layline_margin_s: Nonnegative = 9.0
    direct_approach_m: Nonnegative = 100.0
    boundary_guard_s: Nonnegative = 0.0


class ExperimentConfig(Contract):
    schema_version: Literal[1] = 1
    name: str = "deterministic-smoke"
    stage: Literal["fixture", "development", "locked"] = "fixture"
    root_seed: Annotated[int, Field(ge=0)] = 20261003
    device: Literal["cpu"] = "cpu"
    retain_failed_episodes: Literal[True] = True
    cloud_submission_enabled: Literal[False] = False
    boat: BoatConfig = BoatConfig()
    course: CourseConfig = CourseConfig()
    clock: ClockConfig = ClockConfig()
    world: WorldConfig = WorldConfig()
    sensor: SensorConfig = SensorConfig()
    belief: BeliefConfig = BeliefConfig()
    forecast: ForecastConfig = ForecastConfig()
    policy: PolicyConfig = PolicyConfig()

    @model_validator(mode="after")
    def objective_is_consistent(self) -> Self:
        if self.policy.failure_cost_s <= self.course.deadline_s:
            raise ValueError("failure cost must exceed the episode deadline")
        if (
            self.policy.name != "polar_heuristic"
            and self.policy.planning_horizon_s < self.clock.policy_interval_s
        ):
            raise ValueError("planning horizon must cover at least one decision interval")
        return self

    def content_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest()


def load_config(path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(yaml.safe_load(path.read_text()))
