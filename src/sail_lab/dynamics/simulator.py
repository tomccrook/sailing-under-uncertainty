"""Deterministic physics. Policies never receive this mutable state object."""

import math
from dataclasses import dataclass

from sail_lab.config import ExperimentConfig, Vector
from sail_lab.contracts import Action, EpisodeStatus, Observation, ReplaySample
from sail_lab.dynamics.polar import SyntheticPolar
from sail_lab.geometry import (
    add,
    bearing,
    boundary_exit_fraction,
    heading_vector,
    scale,
    segment_circle_fraction,
    subtract,
    wrap,
)
from sail_lab.observations import ObservationSystem
from sail_lab.world import FrozenWorld, WindField


class InvalidNumericState(RuntimeError):
    """An implementation/numerical fault, never a normal sailing failure."""


@dataclass
class BoatState:
    position: Vector
    heading: float
    speed: float = 0.0
    elapsed: float = 0.0
    tacks: int = 0
    distance: float = 0.0
    manoeuvre_time: float = 0.0
    tack_remaining: float = 0.0
    tack_origin_side: int = 0
    last_side: int = 0
    turn_target: float = 0.0
    requested: float = 0.0
    tack_rate: float = 0.0
    tack_source_rad: float = 0.0
    status: EpisodeStatus = "running"


class Simulator:
    def __init__(
        self,
        config: ExperimentConfig,
        initial_heading: float | None = None,
        world: WindField | None = None,
        instruments_enabled: bool = True,
    ):
        self.config = config
        self.instruments_enabled = instruments_enabled
        self.polar = SyntheticPolar(config.boat)
        self.world = (
            world
            if world is not None
            else FrozenWorld(config.world, config.root_seed, config.course.deadline_s)
        )
        self.refresh_wind(config.course.start_xy_m, 0)
        heading = (
            math.radians(config.course.initial_heading_deg)
            if initial_heading is None
            else initial_heading
        )
        self.state = BoatState(config.course.start_xy_m, wrap(heading))
        self.state.turn_target = self.state.requested = self.state.heading
        self.state.last_side = self.sailing_side(self.state.heading)
        self.instruments = (
            ObservationSystem(config, self.world)
            if instruments_enabled
            and (
                config.sensor.information_set == "noisy_local_known_current"
                or config.world.family != "constant"
                or config.forecast.enabled
            )
            else None
        )
        if self.instruments is not None:
            self.instruments.advance(0, self.state.position)

    def refresh_wind(self, position: Vector, elapsed_s: float) -> None:
        wind = self.world.at(position, elapsed_s)
        self.air, self.current = wind.air_m_s, wind.current_m_s
        self.water_air = subtract(self.air, self.current)
        self.water_wind_speed = math.hypot(*self.water_air)
        self.water_source = bearing(scale(self.water_air, -1))

    def sailing_side(self, heading: float) -> int:
        relative = wrap(heading - self.water_source)
        if math.radians(self.config.boat.no_go_deg) <= abs(relative) < math.pi / 2:
            return 1 if relative > 0 else -1
        return 0

    def observation(self) -> Observation:
        if not self.instruments_enabled:
            raise RuntimeError("hypothetical physics has no live instruments")
        s = self.state
        if self.instruments is not None:
            return self.instruments.snapshot(
                s.elapsed,
                s.position,
                s.heading,
                s.speed,
                s.tack_remaining,
                s.requested,
                s.turn_target,
            )
        self.refresh_wind(s.position, s.elapsed)
        return Observation(
            elapsed_s=s.elapsed,
            position_xy_m=s.position,
            heading_rad=s.heading,
            speed_water_m_s=s.speed,
            target_xy_m=self.config.course.target_xy_m,
            bounds_xy_m=self.config.course.bounds_xy_m,
            remaining_s=max(0.0, self.config.course.deadline_s - s.elapsed),
            wind_air_m_s=self.air,
            known_current_m_s=self.current,
            measurement_time_s=(
                math.floor((s.elapsed + 1e-9) / self.config.clock.sensor_interval_s)
                * self.config.clock.sensor_interval_s
            ),
            measurement_valid=True,
            measurement_available_s=(
                math.floor((s.elapsed + 1e-9) / self.config.clock.sensor_interval_s)
                * self.config.clock.sensor_interval_s
            ),
            manoeuvre_active=s.tack_remaining > 1e-9,
            previous_action_rad=s.requested,
            executed_target_rad=s.turn_target,
            manoeuvre_remaining_s=s.tack_remaining,
        )

    def request(self, action: Action) -> None:
        s = self.state
        if s.status != "running":
            raise RuntimeError("cannot command a terminated episode")
        self.refresh_wind(s.position, s.elapsed)
        s.requested = wrap(action.target_heading_rad)
        if s.tack_remaining > 1e-9:
            return  # Latched manoeuvre cannot overlap or be cancelled by command chatter.
        s.turn_target = s.requested
        side = self.sailing_side(s.turn_target)
        # A downwind gybe or natural wind-side change is not counted as a commanded tack.
        path_mid = wrap(s.heading + wrap(s.turn_target - s.heading) / 2 - self.water_source)
        origin_side = self.sailing_side(s.heading)
        if side and origin_side and side != origin_side and abs(path_mid) < math.pi / 2:
            duration = max(
                self.config.boat.minimum_tack_s,
                abs(wrap(s.turn_target - s.heading))
                / math.radians(self.config.boat.turn_rate_deg_s),
            )
            s.tack_origin_side = origin_side
            s.tack_source_rad = self.water_source
            s.tack_remaining = duration
            s.tack_rate = abs(wrap(s.turn_target - s.heading)) / duration

    def step(self, dt: float | None = None) -> None:
        s = self.state
        if s.status != "running":
            raise RuntimeError("cannot step a terminated episode")
        duration = self.config.clock.physics_dt_s if dt is None else dt
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("step duration must be finite and positive")
        duration = min(duration, self.config.course.deadline_s - s.elapsed)
        # Split at manoeuvre completion so minimum duration is independent of dt.
        split_at = s.tack_remaining if s.tack_remaining > 1e-9 else math.inf
        if self.instruments is not None:
            split_at = min(split_at, self.instruments.next_event_s() - s.elapsed)
        if 1e-9 < split_at < duration - 1e-9:
            first = split_at
            self.step(first)
            if s.status == "running":
                self.step(duration - first)
            return
        old_heading, old_speed = s.heading, s.speed
        self.refresh_wind(s.position, s.elapsed + duration / 2)
        rate = (
            s.tack_rate
            if s.tack_remaining > 1e-9
            else math.radians(self.config.boat.turn_rate_deg_s)
        )
        error = wrap(s.turn_target - old_heading)
        turn = math.copysign(min(abs(error), rate * duration), error)
        midpoint = wrap(old_heading + turn / 2)
        beta = abs(wrap(midpoint - self.water_source))
        target_speed = self.polar.speed(self.water_wind_speed, beta)
        tau = self.config.boat.acceleration_tau_s
        decay = math.exp(-duration / tau)
        water_distance = target_speed * duration + (old_speed - target_speed) * tau * (1 - decay)
        displacement = add(
            scale(heading_vector(midpoint), water_distance), scale(self.current, duration)
        )
        end = add(s.position, displacement)
        hit = segment_circle_fraction(
            s.position, end, self.config.course.target_xy_m, self.config.course.finish_radius_m
        )
        exit_at = boundary_exit_fraction(s.position, end, self.config.course.bounds_xy_m)
        events: list[tuple[float, EpisodeStatus]] = []
        if hit is not None:
            events.append((hit, "finished"))
        if exit_at is not None:
            events.append((exit_at, "boundary_violation"))
        fraction, status = min(events, key=lambda event: event[0]) if events else (1.0, "running")
        actual_dt = duration * fraction
        s.position = add(s.position, scale(displacement, fraction))
        s.heading = wrap(old_heading + turn * fraction)
        s.speed = target_speed + (old_speed - target_speed) * math.exp(-actual_dt / tau)
        s.elapsed += actual_dt
        s.distance += math.hypot(*displacement) * fraction
        if s.tack_remaining > 1e-9:
            s.manoeuvre_time += actual_dt
            s.tack_remaining = max(0.0, s.tack_remaining - actual_dt)
            if s.tack_remaining <= 1e-9:
                relative = wrap(s.heading - s.tack_source_rad)
                new_side = (
                    (1 if relative > 0 else -1)
                    if abs(relative) >= math.radians(self.config.boat.no_go_deg)
                    and abs(relative) < math.pi / 2
                    else 0
                )
                if new_side and new_side != s.tack_origin_side:
                    s.tacks += 1
                    s.last_side = new_side
        elif self.sailing_side(s.heading):
            s.last_side = self.sailing_side(s.heading)
        if not all(math.isfinite(v) for v in (*s.position, s.heading, s.speed, s.elapsed)):
            s.status = "invalid_numeric_state"
            raise InvalidNumericState("nonfinite physics state")
        s.status = status
        if s.status == "running" and s.elapsed >= self.config.course.deadline_s - 1e-9:
            s.status = "deadline_exceeded"
        if self.instruments is not None:
            self.instruments.advance(s.elapsed, s.position)

    def sample(self) -> ReplaySample:
        s = self.state
        self.refresh_wind(s.position, s.elapsed)
        return ReplaySample(
            elapsed_s=s.elapsed,
            position_xy_m=s.position,
            heading_rad=s.heading,
            speed_water_m_s=s.speed,
            tack_count=s.tacks,
            manoeuvre_active=s.tack_remaining > 1e-9,
            requested_heading_rad=s.requested,
            executed_target_rad=s.turn_target,
            observation=self.observation(),
            diagnostic_wind_air_m_s=self.air,
            diagnostic_current_m_s=self.current,
        )
