"""B0 exact ideal fixture and B1 polar-informed constant-wind layline policy."""

import math
from typing import Protocol

import numpy as np

from sail_lab.config import BoatConfig, PolicyConfig
from sail_lab.contracts import Action, Observation
from sail_lab.dynamics.polar import SyntheticPolar
from sail_lab.geometry import add, bearing, heading_vector, scale, subtract, wrap


class Policy(Protocol):
    def reset(self) -> None: ...
    def act(self, observation: Observation) -> Action: ...


def ideal_reference(distance_m: float = 1000.0, speed_m_s: float = 3.0) -> dict[str, float]:
    if distance_m <= 0 or speed_m_s <= 0:
        raise ValueError("reference distance and speed must be positive")
    length = distance_m / math.cos(math.pi / 4)
    return {"distance_m": length, "arrival_s": length / speed_m_s, "max_lateral_m": distance_m / 2}


def two_leg_reference(
    distance_m: float, speed_m_s: float, angle_rad: float, finish_radius_m: float = 0.0
) -> dict[str, float]:
    """Lossless north-wind two-heading solution, reaching a circle on the second leg.

    With headings +/-a and constant speed, total path length is d/cos(a)-r.
    The end is target minus r times the second-leg heading; first leg has length
    (d/cos(a)-2r)/2. This is a small-case reference, not a realistic optimum.
    """
    if not (0 < angle_rad < math.pi / 2 and speed_m_s > 0 and distance_m > 0):
        raise ValueError("positive distance/speed and upwind angle in (0,pi/2) required")
    if not 0 <= finish_radius_m < distance_m / 2:
        raise ValueError("finish radius must be nonnegative and smaller than half the leg")
    total = distance_m / math.cos(angle_rad) - finish_radius_m
    first = (total - finish_radius_m) / 2
    return {
        "distance_m": total,
        "arrival_s": total / speed_m_s,
        "first_leg_s": first / speed_m_s,
        "max_lateral_m": first * math.sin(angle_rad),
    }


def ideal_trajectory(dt_s: float, side: int = 1) -> list[tuple[float, float, float]]:
    """Event-split numerical integration of the point-target fixture, independent of physics."""
    if dt_s <= 0 or side not in (-1, 1):
        raise ValueError("positive dt and side +/-1 required")
    total = ideal_reference()["arrival_s"]
    t, x, y = 0.0, 0.0, 0.0
    rows = [(t, x, y)]
    for end, direction in ((total / 2, side), (total, -side)):
        while t < end - 1e-10:
            dt = min(dt_s, end - t)
            dx, dy = heading_vector(direction * math.pi / 4)
            x, y = x + dx * 3 * dt, y + dy * 3 * dt
            t += dt
            rows.append((t, x, y))
    return rows


def observed_air(observation: Observation, wind_input: str) -> tuple[float, float]:
    if observation.belief is not None and wind_input == "belief":
        return observation.belief.mean_air_m_s
    return observation.wind_air_m_s


class PolarHeuristic:
    """Use observed wind, polar VMG and two-leg geometry with commitment.

    Fixture parameters are provisional; stochastic baseline tuning occurs at M3.
    """

    def __init__(self, boat: BoatConfig, config: PolicyConfig):
        self.polar = SyntheticPolar(boat)
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.side: int = self.config.initial_side
        self.last_switch = -math.inf

    def act(self, observation: Observation) -> Action:
        obs = observation
        wind_water = subtract(observed_air(obs, self.config.wind_input), obs.known_current_m_s)
        source = bearing(scale(wind_water, -1))
        wind_speed = math.hypot(*wind_water)
        angle = self.polar.best_upwind_angle(wind_speed)
        displacement = subtract(obs.target_xy_m, obs.position_xy_m)
        direct = bearing(displacement)
        relative_direct = wrap(direct - source)
        direct_side = 1 if relative_direct > 0 else -1
        if self.config.boundary_guard_s > 0 and not obs.manoeuvre_active:
            velocity = add(
                scale(
                    heading_vector(source + self.side * angle), self.polar.speed(wind_speed, angle)
                ),
                obs.known_current_m_s,
            )
            future = add(obs.position_xy_m, scale(velocity, self.config.boundary_guard_s))
            left, right, bottom, top = obs.bounds_xy_m
            if not left < future[0] < right or not bottom < future[1] < top:
                alternative = add(
                    scale(
                        heading_vector(source - self.side * angle),
                        self.polar.speed(wind_speed, angle),
                    ),
                    obs.known_current_m_s,
                )
                candidate = add(obs.position_xy_m, scale(alternative, self.config.boundary_guard_s))
                if left < candidate[0] < right and bottom < candidate[1] < top:
                    self.side *= -1
                    self.last_switch = obs.elapsed_s
                    return Action(target_heading_rad=wrap(source + self.side * angle))
        # Direct approach can bear away once the target is outside the optimum layline.
        close_sailable = math.hypot(*displacement) <= self.config.direct_approach_m and abs(
            relative_direct
        ) > math.radians(self.polar.config.no_go_deg)
        if (abs(relative_direct) >= angle or close_sailable) and direct_side == self.side:
            return Action(target_heading_rad=direct)
        speed = self.polar.speed(wind_speed, angle)
        velocities = [
            add(scale(heading_vector(source + side * angle), speed), obs.known_current_m_s)
            for side in (self.side, -self.side)
        ]
        matrix = np.asarray(velocities).T
        if abs(float(np.linalg.det(matrix))) > 1e-9:
            first_time = float(np.linalg.solve(matrix, np.asarray(displacement))[0])
            if (
                first_time <= self.config.layline_margin_s
                and not obs.manoeuvre_active
                and obs.elapsed_s - self.last_switch >= self.config.commitment_s
            ):
                self.side *= -1
                self.last_switch = obs.elapsed_s
        return Action(target_heading_rad=wrap(source + self.side * angle))
