"""Synthetic station nowcasts; the publisher observes only the issue-time world.

Forecast means are persistence/mean-reverting conditional predictions, not
samples of realised future weather. Bias and under/over-confidence are explicit.
"""

import math

from sail_lab.config import ForecastConfig
from sail_lab.contracts import ForecastPacket, ForecastPoint
from sail_lab.geometry import bearing, scale, wrap
from sail_lab.streams import keyed_rng
from sail_lab.world import WindField


def issue_forecast(
    config: ForecastConfig, world: WindField, root_seed: int, issued_s: float, issue_tick: int
) -> ForecastPacket:
    rng = keyed_rng(root_seed, "forecasts", issue_tick)
    air = world.at(config.station_xy_m, issued_s).air_m_s
    observed_direction = (
        bearing(scale(air, -1)) + math.radians(config.station_direction_noise_deg) * rng.normal()
    )
    observed_speed = max(0.0, math.hypot(*air) + config.station_speed_noise_m_s * rng.normal())
    points = []
    count = round(config.horizon_s / config.point_interval_s) + 1
    reference = math.radians(config.reference_from_deg)
    for i in range(count):
        lead = i * config.point_interval_s
        if config.model == "mean_reversion":
            direction_decay = math.exp(-config.direction_kappa_s_inv * lead)
            speed_decay = math.exp(-config.speed_kappa_s_inv * lead)
            direction = reference + wrap(observed_direction - reference) * direction_decay
            speed = (
                config.reference_speed_m_s
                + (observed_speed - config.reference_speed_m_s) * speed_decay
            )
            direction_var = (
                config.station_direction_noise_deg** 2 * direction_decay** 2
                + config.direction_sigma_deg_sqrt_s** 2
                / (2 * config.direction_kappa_s_inv)
                * (1 - direction_decay**2)
            )
            speed_var = (
                config.station_speed_noise_m_s** 2 * speed_decay** 2
                + config.speed_sigma_m_s_sqrt_s** 2
                / (2 * config.speed_kappa_s_inv)
                * (1 - speed_decay**2)
            )
        else:
            direction, speed = observed_direction, observed_speed
            direction_var = (
                config.station_direction_noise_deg**2 + config.direction_sigma_deg_sqrt_s**2 * lead
            )
            speed_var = config.station_speed_noise_m_s**2 + config.speed_sigma_m_s_sqrt_s**2 * lead
        points.append(
            ForecastPoint(
                valid_s=issued_s + lead,
                wind_from_rad=wrap(direction + math.radians(config.direction_bias_deg)),
                speed_m_s=max(0, speed + config.speed_bias_m_s),
                direction_std_rad=max(
                    1e-6, math.radians(math.sqrt(direction_var)) * config.uncertainty_scale
                ),
                speed_std_m_s=max(1e-6, math.sqrt(speed_var) * config.uncertainty_scale),
            )
        )
    return ForecastPacket(
        issued_s=issued_s,
        available_s=issued_s + config.availability_delay_s,
        horizon_s=config.horizon_s,
        station_xy_m=config.station_xy_m,
        spatial_resolution_m=config.spatial_resolution_m,
        model=config.model,
        points=tuple(points),
    )
