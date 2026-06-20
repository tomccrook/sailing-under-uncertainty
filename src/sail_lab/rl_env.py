"""Gym boundary, fixed public features and an estimate-independent action contract.

The eight commands use only raw instruments and geometry. Belief features never
change their meaning, so present-only PPO cannot receive a filter via its actions.
No diagnostic wind, scenario ID, RNG state or outcome goes into Gym info.
"""

import math
from collections import deque
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from pydantic import Field

from sail_lab.config import Contract, ExperimentConfig
from sail_lab.contracts import Action, Observation
from sail_lab.dynamics.simulator import Simulator
from sail_lab.geometry import bearing, scale, subtract, wrap

FeatureMode = Literal["present", "belief", "history"]
Array = NDArray[np.float32]
ACTION_VERSION = "raw-eight-headings-v1"


def headings(obs: Observation) -> tuple[float, ...]:
    source = bearing(scale(subtract(obs.wind_air_m_s, obs.known_current_m_s), -1))
    return tuple(
        wrap(source + side * math.radians(a)) for side in (-1, 1) for a in (45, 55, 70)
    ) + (
        obs.executed_target_rad,
        bearing(subtract(obs.target_xy_m, obs.position_xy_m)),
    )


PRESENT_NAMES = (
    "target_dx",
    "target_dy",
    "left_distance",
    "right_distance",
    "bottom_distance",
    "top_distance",
    "heading_sin",
    "heading_cos",
    "water_speed",
    "remaining_time",
    "raw_air_e",
    "raw_air_n",
    "current_e",
    "current_n",
    "measurement_age",
    "measurement_valid",
    "dropout",
    "previous_sin",
    "previous_cos",
    "executed_sin",
    "executed_cos",
    "manoeuvre_active",
    "manoeuvre_remaining",
    "forecast_available",
    "forecast_age",
    "forecast_60_sin",
    "forecast_60_cos",
    "forecast_60_speed",
    "forecast_60_direction_std",
    "forecast_60_speed_std",
)
BELIEF_NAMES = ("belief_e", "belief_n", "cov_ee", "cov_en", "cov_nn", "belief_available")


class FeatureSpec(Contract):
    version: Literal["public-fixed-scales-v1"] = "public-fixed-scales-v1"
    mode: FeatureMode = "present"
    history_length: int = Field(default=4, ge=2, le=16)
    distance_scale_m: float = Field(default=1000, ge=1000, le=1000)
    speed_scale_m_s: float = Field(default=6, ge=6, le=6)
    deadline_scale_s: float = Field(default=1800, ge=1800, le=1800)
    clip_absolute: float = Field(default=10, ge=10, le=10)
    action_version: Literal["raw-eight-headings-v1"] = "raw-eight-headings-v1"

    @property
    def names(self) -> tuple[str, ...]:
        if self.mode == "history":
            return tuple(
                f"lag_{lag}/{name}"
                for lag in reversed(range(self.history_length))
                for name in (*PRESENT_NAMES, "frame_available")
            )
        return PRESENT_NAMES + (BELIEF_NAMES if self.mode == "belief" else ())


def present_features(obs: Observation) -> list[float]:
    x, y = obs.position_xy_m
    dx, dy = subtract(obs.target_xy_m, obs.position_xy_m)
    left, right, bottom, top = obs.bounds_xy_m
    values = [
        dx / 1000,
        dy / 1000,
        (x - left) / 1000,
        (right - x) / 1000,
        (y - bottom) / 1000,
        (top - y) / 1000,
        math.sin(obs.heading_rad),
        math.cos(obs.heading_rad),
        obs.speed_water_m_s / 6,
        obs.remaining_s / 1800,
        obs.wind_air_m_s[0] / 6,
        obs.wind_air_m_s[1] / 6,
        obs.known_current_m_s[0] / 6,
        obs.known_current_m_s[1] / 6,
        (obs.elapsed_s - obs.measurement_time_s) / 30,
        float(obs.measurement_valid),
        float(obs.latest_sensor_dropout),
        math.sin(obs.previous_action_rad),
        math.cos(obs.previous_action_rad),
        math.sin(obs.executed_target_rad),
        math.cos(obs.executed_target_rad),
        float(obs.manoeuvre_active),
        obs.manoeuvre_remaining_s / 30,
    ]
    packet = obs.forecast
    if packet is None:
        values.extend([0.0] * 7)
    else:
        # Public forecast, not the future realised wind. Nearest declared point;
        # last prediction is held beyond the packet horizon, matching M3.
        point = min(packet.points, key=lambda p: abs(p.valid_s - obs.elapsed_s - 60))
        values.extend(
            [
                1,
                (obs.elapsed_s - packet.issued_s) / 120,
                math.sin(point.wind_from_rad),
                math.cos(point.wind_from_rad),
                point.speed_m_s / 6,
                point.direction_std_rad,
                point.speed_std_m_s / 6,
            ]
        )
    return values


class FeatureEncoder:
    def __init__(self, spec: FeatureSpec):
        self.spec = spec
        self.frames: deque[list[float]] = deque(maxlen=spec.history_length)

    def reset(self) -> None:
        self.frames.clear()

    def encode(self, obs: Observation) -> Array:
        values = present_features(obs)
        if self.spec.mode == "belief":
            belief = obs.belief
            values.extend(
                [
                    belief.mean_air_m_s[0] / 6,
                    belief.mean_air_m_s[1] / 6,
                    belief.covariance_m2_s2[0][0] / 4,
                    belief.covariance_m2_s2[0][1] / 4,
                    belief.covariance_m2_s2[1][1] / 4,
                    1,
                ]
                if belief is not None
                else [0.0] * 6
            )
        elif self.spec.mode == "history":
            self.frames.append([*values, 1.0])
            padding = [[0.0] * (len(PRESENT_NAMES) + 1)] * (
                self.spec.history_length - len(self.frames)
            )
            values = [v for frame in [*padding, *self.frames] for v in frame]
        array = np.asarray(values, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("nonfinite public feature")
        return np.clip(array, -self.spec.clip_absolute, self.spec.clip_absolute)


def potential(obs: Observation, terminal: bool, shaping: bool) -> float:
    return -math.dist(obs.position_xy_m, obs.target_xy_m) / 200 if shaping and not terminal else 0.0


DEFAULT_FEATURES = FeatureSpec()


class SailingEnv(gym.Env[Array, int]):
    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        scenarios: tuple[ExperimentConfig, ...],
        spec: FeatureSpec = DEFAULT_FEATURES,
        shaping: bool = True,
    ):
        if not scenarios:
            raise ValueError("an environment needs an explicit nonempty scenario manifest")
        self.scenarios, self.feature_spec, self.shaping = scenarios, spec, shaping
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(-10, 10, (len(spec.names),), np.float32)
        self.encoder = FeatureEncoder(spec)
        self.sim: Simulator | None = None
        self.completed: list[dict[str, Any]] = []  # recorder only; never Gym info
        self.total_reward = 0.0
        self.steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Array, dict[str, Any]]:
        super().reset(seed=seed)
        self.sim = Simulator(self.scenarios[int(self.np_random.integers(len(self.scenarios)))])
        self.encoder.reset()
        self.total_reward, self.steps = 0.0, 0
        return self.encoder.encode(self.sim.observation()), {}

    def step(self, action: int) -> tuple[Array, float, bool, bool, dict[str, Any]]:
        if self.sim is None or self.sim.state.status != "running":
            raise RuntimeError("reset required before stepping")
        if not self.action_space.contains(action):
            raise ValueError("action outside declared space")
        sim = self.sim
        before = sim.observation()
        sim.request(Action(target_heading_rad=headings(before)[int(action)]))
        end = sim.state.elapsed + sim.config.clock.policy_interval_s
        while sim.state.status == "running" and sim.state.elapsed < end - 1e-8:
            sim.step(min(sim.config.clock.physics_dt_s, end - sim.state.elapsed))
        terminal = sim.state.status != "running"
        after = sim.observation()
        reward = -(after.elapsed_s - before.elapsed_s) / 100
        if terminal and sim.state.status != "finished":
            reward -= (sim.config.policy.failure_cost_s - after.elapsed_s) / 100
        # gamma=1: sums to -total_cost/100 - initial potential, even on failure.
        reward += potential(after, terminal, self.shaping) - potential(before, False, self.shaping)
        self.total_reward += reward
        self.steps += 1
        if terminal:
            self.completed.append(
                {
                    "scenario_id": f"{sim.config.world.family}-{sim.config.root_seed}",
                    "config_hash": sim.config.content_hash(),
                    "status": sim.state.status,
                    "elapsed_s": sim.state.elapsed,
                    "return": self.total_reward,
                    "decisions": self.steps,
                }
            )
        # Deadline is a task outcome with the full terminal cost, not an external
        # time-limit truncation to which SB3 would add a continuation value.
        return self.encoder.encode(after), reward, terminal, False, {}
