"""Frozen exogenous wind, independent of boat route and query order.

OU is exact at fixed world grid knots. Piecewise-linear interpolation defines
the numerical world between knots; it is not a continuous OU path. Physics dt
does not regenerate the grid. Direction uses bounded unwrapped local deviation;
speed is capped log-OU with its latent mean corrected for Jensen's inequality.
"""

import hashlib
import math
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from sail_lab.config import Contract, Vector, WorldConfig
from sail_lab.geometry import wind_air
from sail_lab.streams import stream_seed


class WindState(Contract):
    air_m_s: Vector
    current_m_s: Vector


class WindField(Protocol):
    def at(self, position: Vector, elapsed_s: float) -> WindState: ...


def ou_path(
    rng: np.random.Generator,
    count: int,
    dt: float,
    mean: float,
    kappa: float,
    sigma: float,
) -> NDArray[np.float64]:
    """Stationary initialisation and exact scalar OU transitions (units supplied by caller)."""
    if (
        not all(math.isfinite(value) for value in (dt, mean, kappa, sigma))
        or count < 1
        or dt <= 0
        or kappa <= 0
        or sigma < 0
    ):
        raise ValueError("invalid OU path parameters")
    decay = math.exp(-kappa * dt)
    stationary_sd = sigma / math.sqrt(2 * kappa)
    innovation_sd = stationary_sd * math.sqrt(-math.expm1(-2 * kappa * dt))
    values = np.empty(count, dtype=np.float64)
    values[0] = mean + stationary_sd * rng.normal()
    noise = rng.normal(size=count - 1)
    for i in range(1, count):
        values[i] = mean + (values[i - 1] - mean) * decay + innovation_sd * noise[i - 1]
    return values


class FrozenWorld:
    def __init__(self, config: WorldConfig, root_seed: int, duration_s: float):
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("finite positive world duration required")
        self.config = config
        self.duration_s = duration_s
        count = math.ceil(duration_s / config.grid_dt_s) + 1
        self.times = np.arange(count, dtype=np.float64) * config.grid_dt_s
        seeds = stream_seed(root_seed, "world").spawn(7)
        rngs = [np.random.default_rng(seed) for seed in seeds]
        self.direction = np.zeros(count, dtype=np.float64)
        self.speed = np.full(count, config.wind_speed_m_s, dtype=np.float64)
        self.spatial = np.zeros((3, count), dtype=np.float64)
        self.phases = np.zeros(3, dtype=np.float64)
        self.regimes = np.zeros(count, dtype=np.float64)
        self.oscillation_phase = float(rngs[2].uniform(0, 2 * math.pi))
        if config.family in ("ou", "regime", "spatial"):
            self.direction = ou_path(
                rngs[0],
                count,
                config.grid_dt_s,
                0,
                config.direction_kappa_s_inv,
                config.direction_sigma_deg_sqrt_s,
            )
            variance = config.log_speed_sigma_sqrt_s**2 / (2 * config.log_speed_kappa_s_inv)
            log_speed = ou_path(
                rngs[1],
                count,
                config.grid_dt_s,
                math.log(config.wind_speed_m_s) - variance / 2,
                config.log_speed_kappa_s_inv,
                config.log_speed_sigma_sqrt_s,
            )
            self.speed = np.minimum(np.exp(log_speed), config.max_speed_m_s)
        if config.family == "oscillating":
            self.direction = config.oscillation_amplitude_deg * np.sin(
                2 * math.pi * self.times / config.oscillation_period_s + self.oscillation_phase
            )
        if config.family == "regime":
            sign = int(rngs[2].choice([-1, 1]))
            next_shift = float(rngs[2].exponential(config.regime_mean_duration_s))
            for i, t in enumerate(self.times):
                while t >= next_shift:
                    sign *= -1
                    next_shift += float(rngs[2].exponential(config.regime_mean_duration_s))
                self.regimes[i] = sign * config.regime_shift_deg
        if config.family == "spatial":
            for j in range(3):
                self.spatial[j] = ou_path(
                    rngs[3 + j],
                    count,
                    config.grid_dt_s,
                    0,
                    config.direction_kappa_s_inv,
                    math.sqrt(2 * config.direction_kappa_s_inv),
                )
            self.phases = rngs[6].uniform(0, 2 * math.pi, 3)
        for array in (
            self.times,
            self.direction,
            self.speed,
            self.spatial,
            self.phases,
            self.regimes,
        ):
            array.setflags(write=False)
        digest = hashlib.sha256(config.model_dump_json().encode())
        for array in (
            self.times,
            self.direction,
            self.speed,
            self.spatial,
            self.phases,
            self.regimes,
        ):
            digest.update(array.tobytes())
        self.digest = digest.hexdigest()

    def save(self, path: Path) -> None:
        """Diagnostic world artefact; policies must never load this realised future."""
        np.savez_compressed(
            path,
            times=self.times,
            direction=self.direction,
            speed=self.speed,
            spatial=self.spatial,
            phases=self.phases,
            regimes=self.regimes,
            config_json=self.config.model_dump_json(),
            digest=self.digest,
        )

    def at(self, position: Vector, elapsed_s: float) -> WindState:
        if not math.isfinite(elapsed_s) or not 0 <= elapsed_s <= self.duration_s + 1e-8:
            raise ValueError("world query outside frozen time domain")
        if not all(math.isfinite(value) for value in position):
            raise ValueError("world query position must be finite")
        cfg = self.config
        deviation = float(np.interp(elapsed_s, self.times, self.direction))
        if cfg.family == "regime":
            # Regime labels are right-continuous at grid knots, preserving abrupt shifts.
            tick = min(int(elapsed_s / cfg.grid_dt_s), len(self.regimes) - 1)
            deviation += float(self.regimes[tick])
        if cfg.family == "spatial":
            x, y = position
            coordinates = (x, y, (x + y) / math.sqrt(2))
            for j, coordinate in enumerate(coordinates):
                coefficient = float(np.interp(elapsed_s, self.times, self.spatial[j]))
                deviation += (
                    cfg.spatial_amplitude_deg
                    / math.sqrt(3)
                    * coefficient
                    * math.sin(
                        2 * math.pi * coordinate / cfg.spatial_length_m + float(self.phases[j])
                    )
                )
        deviation = float(
            np.clip(deviation, -cfg.max_direction_deviation_deg, cfg.max_direction_deviation_deg)
        )
        speed = float(np.interp(elapsed_s, self.times, self.speed))
        return WindState(
            air_m_s=wind_air(math.radians(cfg.wind_from_deg + deviation), speed),
            current_m_s=cfg.current_ground_m_s,
        )
