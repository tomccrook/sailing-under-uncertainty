"""New synthetic processes, kept outside the previously frozen numerical package."""

import hashlib
import math
from typing import Literal

import numpy as np
from pydantic import Field

from sail_lab.config import Contract, ExperimentConfig, Vector
from sail_lab.geometry import wind_air
from sail_lab.streams import stream_seed
from sail_lab.world import FrozenWorld, WindState

NEW_FAMILIES = ("gust_front", "persistent_trend")


class NewWorldConfig(Contract):
    version: Literal["new-processes-v1"] = "new-processes-v1"
    family: Literal["gust_front", "persistent_trend"]
    grid_dt_s: float = Field(default=2, gt=0)
    gust_arrival_mean_s: float = Field(default=140, gt=0)
    gust_decay_s: float = Field(default=45, gt=0)
    gust_direction_sd_deg: float = Field(default=16, ge=0)
    gust_log_speed_sd: float = Field(default=0.12, ge=0)
    trend_refresh_s: float = Field(default=90, gt=0)
    trend_rate_sd_deg_s: float = Field(default=0.18, ge=0)
    trend_restoring_s: float = Field(default=150, gt=0)
    direction_limit_deg: float = Field(default=50, gt=0, le=80)


class NewWorld:
    """Frozen compound-Poisson shot noise or persistent random-slope direction.

    The processes are spatially uniform. Arrays are offline truth; a policy only
    receives the standard observation. Query order and physics dt cannot change them.
    """

    def __init__(self, config: ExperimentConfig, process: NewWorldConfig):
        self.config, self.process = config, process
        self.duration_s = config.course.deadline_s
        dt = process.grid_dt_s
        self.times = np.arange(math.ceil(self.duration_s / dt) + 1, dtype=float) * dt
        # Distinct from old OU's child indices; direction/speed/event streams separated.
        streams = stream_seed(config.root_seed, "world").spawn(12)
        event_rng = np.random.default_rng(streams[8])
        direction_rng = np.random.default_rng(streams[9])
        speed_rng = np.random.default_rng(streams[10])
        self.direction = np.zeros(len(self.times))
        log_speed = np.zeros(len(self.times))
        self.events: list[float] = []
        if process.family == "gust_front":
            # Zero-start transient deliberately differs from stationary OU training.
            t = float(event_rng.exponential(process.gust_arrival_mean_s))
            while t <= self.times[-1]:
                self.events.append(t)
                response = np.where(
                    self.times >= t,
                    np.exp(-np.maximum(0, self.times - t) / process.gust_decay_s),
                    0,
                )
                self.direction += direction_rng.normal(0, process.gust_direction_sd_deg) * response
                log_speed += speed_rng.normal(0, process.gust_log_speed_sd) * response
                t += float(event_rng.exponential(process.gust_arrival_mean_s))
        else:
            # Piecewise-constant random angular rates with deterministic restoring force.
            rates = direction_rng.normal(
                0,
                process.trend_rate_sd_deg_s,
                size=math.ceil(self.times[-1] / process.trend_refresh_s) + 1,
            )
            self.events = list(np.arange(len(rates), dtype=float) * process.trend_refresh_s)
            decay = math.exp(-dt / process.trend_restoring_s)
            for i in range(1, len(self.times)):
                rate = rates[int(self.times[i - 1] / process.trend_refresh_s)]
                self.direction[i] = self.direction[
                    i - 1
                ] * decay + rate * process.trend_restoring_s * (1 - decay)
            # Bounded slow speed cycle with an independently drawn phase, not log OU.
            phase = float(speed_rng.uniform(0, 2 * math.pi))
            log_speed = 0.1 * np.sin(2 * math.pi * self.times / 300 + phase)
        self.direction = np.clip(
            self.direction, -process.direction_limit_deg, process.direction_limit_deg
        )
        self.speed = np.clip(
            config.world.wind_speed_m_s * np.exp(log_speed), 0, config.world.max_speed_m_s
        )
        digest = hashlib.sha256(
            (config.world.model_dump_json() + process.model_dump_json()).encode()
        )
        for array in (self.times, self.direction, self.speed):
            array.setflags(write=False)
            digest.update(array.tobytes())
        self.digest = digest.hexdigest()

    def at(self, position: Vector, elapsed_s: float) -> WindState:
        if not math.isfinite(elapsed_s) or not 0 <= elapsed_s <= self.duration_s + 1e-8:
            raise ValueError("world query outside frozen domain")
        if not all(math.isfinite(x) for x in position):
            raise ValueError("nonfinite position")
        return WindState(
            air_m_s=wind_air(
                math.radians(
                    self.config.world.wind_from_deg
                    + float(np.interp(elapsed_s, self.times, self.direction))
                ),
                float(np.interp(elapsed_s, self.times, self.speed)),
            ),
            current_m_s=self.config.world.current_ground_m_s,
        )


def make_world(
    config: ExperimentConfig, process: dict[str, object] | None
) -> FrozenWorld | NewWorld:
    if process is None:
        return FrozenWorld(config.world, config.root_seed, config.course.deadline_s)
    return NewWorld(config, NewWorldConfig.model_validate(process))
