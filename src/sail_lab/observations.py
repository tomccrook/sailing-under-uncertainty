"""Acquisition, delayed delivery, masks, and estimation outside the policy."""

import math
from collections import deque

from sail_lab.beliefs import VectorWindFilter
from sail_lab.config import ExperimentConfig, Vector
from sail_lab.contracts import ForecastPacket, Observation, WindMeasurement
from sail_lab.forecasts import issue_forecast
from sail_lab.geometry import bearing, scale, wind_air
from sail_lab.streams import keyed_rng
from sail_lab.world import WindField


class ObservationSystem:
    def __init__(self, config: ExperimentConfig, world: WindField):
        self.config, self.world = config, world
        self.filter = VectorWindFilter(config.belief)
        self.sensor_tick = 0
        self.forecast_tick = 0
        self.pending: deque[WindMeasurement] = deque()
        self.pending_masks: deque[tuple[float, bool]] = deque()
        self.pending_forecasts: deque[ForecastPacket] = deque()
        self.measurement: WindMeasurement | None = None
        self.forecast: ForecastPacket | None = None
        self.latest_dropout = False

    def next_event_s(self) -> float:
        sensor = self.sensor_tick * self.config.clock.sensor_interval_s
        forecast = (
            self.forecast_tick * self.config.clock.forecast_interval_s
            if self.config.forecast.enabled
            else math.inf
        )
        return min(sensor, forecast)

    def advance(self, elapsed_s: float, position: Vector) -> None:
        cfg = self.config
        sample_time = self.sensor_tick * cfg.clock.sensor_interval_s
        if sample_time <= elapsed_s + 1e-8:
            if abs(sample_time - elapsed_s) > 1e-8:
                raise ValueError("sensor clock skipped; advance at acquisition boundaries")
            rng = keyed_rng(cfg.root_seed, "sensors", self.sensor_tick)
            dropped = bool(rng.random() < cfg.sensor.dropout_probability)
            self.pending_masks.append((sample_time + cfg.sensor.latency_s, dropped))
            direction_error = cfg.sensor.direction_noise_deg * rng.normal()
            speed_error = cfg.sensor.speed_noise_m_s * rng.normal()
            if not dropped:
                air = self.world.at(position, sample_time).air_m_s
                source = bearing(scale(air, -1)) + math.radians(
                    direction_error + cfg.sensor.direction_bias_deg
                )
                speed = max(0.0, math.hypot(*air) + speed_error + cfg.sensor.speed_bias_m_s)
                self.pending.append(
                    WindMeasurement(
                        measured_s=sample_time,
                        available_s=sample_time + cfg.sensor.latency_s,
                        air_m_s=wind_air(source, speed),
                        direction_noise_deg=cfg.sensor.direction_noise_deg,
                        speed_noise_m_s=cfg.sensor.speed_noise_m_s,
                    )
                )
            self.sensor_tick += 1
        issue_time = self.forecast_tick * cfg.clock.forecast_interval_s
        if cfg.forecast.enabled and issue_time <= elapsed_s + 1e-8:
            if abs(issue_time - elapsed_s) > 1e-8:
                raise ValueError("forecast issue clock skipped")
            self.pending_forecasts.append(
                issue_forecast(
                    cfg.forecast, self.world, cfg.root_seed, issue_time, self.forecast_tick
                )
            )
            self.forecast_tick += 1
        while self.pending and self.pending[0].available_s <= elapsed_s + 1e-8:
            self.measurement = self.pending.popleft()
            self.filter.predict(self.measurement.available_s)
            self.filter.update(self.measurement)
        self.filter.predict(elapsed_s)
        while self.pending_masks and self.pending_masks[0][0] <= elapsed_s + 1e-8:
            _, self.latest_dropout = self.pending_masks.popleft()
        while self.pending_forecasts and self.pending_forecasts[0].available_s <= elapsed_s + 1e-8:
            self.forecast = self.pending_forecasts.popleft()

    def snapshot(
        self,
        elapsed_s: float,
        position: Vector,
        heading: float,
        speed: float,
        manoeuvre_remaining_s: float,
        requested: float,
        executed: float,
    ) -> Observation:
        measurement = self.measurement
        return Observation(
            elapsed_s=elapsed_s,
            position_xy_m=position,
            heading_rad=heading,
            speed_water_m_s=speed,
            target_xy_m=self.config.course.target_xy_m,
            bounds_xy_m=self.config.course.bounds_xy_m,
            remaining_s=max(0, self.config.course.deadline_s - elapsed_s),
            wind_air_m_s=measurement.air_m_s if measurement else self.config.belief.prior_air_m_s,
            known_current_m_s=self.config.world.current_ground_m_s,
            measurement_time_s=measurement.measured_s if measurement else 0,
            measurement_available_s=measurement.available_s if measurement else 0,
            measurement_valid=measurement is not None
            and elapsed_s - measurement.measured_s <= self.config.sensor.max_measurement_age_s,
            latest_sensor_dropout=self.latest_dropout,
            manoeuvre_active=manoeuvre_remaining_s > 1e-9,
            manoeuvre_remaining_s=manoeuvre_remaining_s,
            previous_action_rad=requested,
            executed_target_rad=executed,
            information_set=self.config.sensor.information_set,
            belief=self.filter.snapshot(),
            forecast=self.forecast,
        )
