"""B2/B3 finite-model MPC with causal approximate rollout recourse.

No evaluation world is accepted by this module. Conditional paths are built
from the public observation and declared planning model. The first action is
shared across samples; subsequent heuristic actions use delayed noisy local
instruments from the hypothetical path. Rollout recourse holds raw measurements
and does not run a future Kalman filter; this approximation is disclosed.
"""

import hashlib
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sail_lab.config import ExperimentConfig, Vector
from sail_lab.contracts import Action, Observation
from sail_lab.dynamics.polar import SyntheticPolar
from sail_lab.dynamics.simulator import BoatState, Simulator
from sail_lab.geometry import bearing, heading_vector, scale, subtract, wind_air, wrap
from sail_lab.policies import PolarHeuristic, observed_air
from sail_lab.streams import keyed_rng
from sail_lab.world import WindState


def upper_cvar(values: list[float] | NDArray[np.float64], alpha: float = 0.95) -> float:
    """Fractional-mass empirical upper-tail CVaR, including ties and small N."""
    if not 0 < alpha < 1:
        raise ValueError("CVaR alpha must be in (0,1)")
    ordered = np.sort(np.asarray(values, dtype=float))[::-1]
    if not len(ordered) or not np.isfinite(ordered).all():
        raise ValueError("CVaR needs finite nonempty outcomes")
    mass = len(ordered) * (1 - alpha)
    weights = np.clip(mass - np.arange(len(ordered)), 0, 1)
    return float(ordered @ weights / mass)


def candidate_headings(obs: Observation, config: ExperimentConfig) -> list[float]:
    water_air = subtract(observed_air(obs, config.policy.wind_input), obs.known_current_m_s)
    source = bearing(scale(water_air, -1))
    angle = SyntheticPolar(config.boat).best_upwind_angle(math.hypot(*water_air))
    angles = (
        max(math.radians(config.boat.no_go_deg + 1), angle - math.radians(5)),
        angle,
        min(math.pi / 2, angle + math.radians(10)),
    )
    headings = [wrap(source + side * a) for side in (-1, 1) for a in angles]
    headings.append(obs.executed_target_rad)
    direct = bearing(subtract(obs.target_xy_m, obs.position_xy_m))
    if abs(wrap(direct - source)) > math.radians(config.boat.no_go_deg):
        headings.append(direct)
    unique: list[float] = []
    for heading in headings:
        if not any(abs(wrap(heading - other)) < 1e-8 for other in unique):
            unique.append(heading)
    return unique


class ConditionalPath:
    def __init__(
        self, times: NDArray[np.float64], air: NDArray[np.float64], current: Vector, start_s: float
    ):
        self.times, self.air, self.current, self.start_s = times, air, current, start_s

    def at(self, position: Vector, elapsed_s: float) -> WindState:
        lead = max(0, elapsed_s - self.start_s)
        return WindState(
            air_m_s=(
                float(np.interp(lead, self.times, self.air[:, 0])),
                float(np.interp(lead, self.times, self.air[:, 1])),
            ),
            current_m_s=self.current,
        )


def conditional_paths(
    obs: Observation, config: ExperimentConfig, tick: int, deterministic: bool
) -> tuple[ConditionalPath, list[ConditionalPath]]:
    count = max(2, math.ceil(config.policy.planning_horizon_s / 2) + 1)
    times = np.linspace(0, config.policy.planning_horizon_s, count)
    local = observed_air(obs, config.policy.wind_input)
    mean = np.tile(local, (count, 1)).astype(float)
    packet = obs.forecast
    if packet is not None and packet.points[-1].valid_s >= obs.elapsed_s:
        valid = np.array([p.valid_s for p in packet.points])
        vectors = np.array([wind_air(p.wind_from_rad, p.speed_m_s) for p in packet.points])
        station = np.column_stack(
            [np.interp(obs.elapsed_s + times, valid, vectors[:, i]) for i in range(2)]
        )
        mean = station + (np.asarray(local) - station[0]) * np.exp(-0.02 * times[:, None])
    mean_path = ConditionalPath(times, mean, obs.known_current_m_s, obs.elapsed_s)
    if deterministic:
        return mean_path, [mean_path]
    direction = np.array([bearing(scale((float(v[0]), float(v[1])), -1)) for v in mean])
    speed = np.linalg.norm(mean, axis=1)
    covariance = (
        np.asarray(obs.belief.covariance_m2_s2)
        if obs.belief is not None
        else np.eye(2) * config.belief.prior_std_m_s**2
    )
    radial = np.asarray(local) / max(math.hypot(*local), 1e-6)
    tangent = np.array([-radial[1], radial[0]])
    initial_direction_std = math.sqrt(max(0, float(tangent @ covariance @ tangent))) / max(
        speed[0], 0.1
    )
    initial_speed_std = math.sqrt(max(0, float(radial @ covariance @ radial)))
    rng = keyed_rng(config.root_seed, "planner", tick)
    direction_sigma = math.radians(config.forecast.direction_sigma_deg_sqrt_s)
    speed_sigma = config.forecast.speed_sigma_m_s_sqrt_s
    if packet is not None:
        first, last = packet.points[0], packet.points[-1]

        def implied_diffusion(initial_std: float, final_std: float, kappa: float) -> float:
            decay = math.exp(-kappa * packet.horizon_s)
            return math.sqrt(
                max(0, final_std**2 - initial_std**2 * decay**2)
                * 2
                * kappa
                / max(1e-8, 1 - decay**2)
            )

        direction_sigma = implied_diffusion(
            first.direction_std_rad, last.direction_std_rad, config.forecast.direction_kappa_s_inv
        )
        speed_sigma = implied_diffusion(
            first.speed_std_m_s, last.speed_std_m_s, config.forecast.speed_kappa_s_inv
        )
    paths = []
    for _ in range(config.policy.scenario_count):
        direction_error = np.empty(count)
        speed_error = np.empty(count)
        direction_error[0] = rng.normal() * initial_direction_std
        speed_error[0] = rng.normal() * initial_speed_std
        for i in range(1, count):
            dt = times[i] - times[i - 1]
            for values, kappa, sigma in (
                (
                    direction_error,
                    config.forecast.direction_kappa_s_inv,
                    direction_sigma,
                ),
                (
                    speed_error,
                    config.forecast.speed_kappa_s_inv,
                    speed_sigma,
                ),
            ):
                decay = math.exp(-kappa * dt)
                values[i] = values[i - 1] * decay + rng.normal() * sigma * math.sqrt(
                    (1 - decay**2) / (2 * kappa)
                )
        air = np.array(
            [
                wind_air(float(d), max(0, float(s)))
                for d, s in zip(direction + direction_error, speed + speed_error, strict=True)
            ]
        )
        paths.append(ConditionalPath(times, air, obs.known_current_m_s, obs.elapsed_s))
    return mean_path, paths


class RecedingHorizonPlanner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.tick = 0
        self.last_decision: dict[str, Any] = {}

    def act(self, obs: Observation) -> Action:
        cfg = self.config
        mean, paths = conditional_paths(obs, cfg, self.tick, cfg.policy.name == "deterministic_mpc")
        candidates = candidate_headings(obs, cfg)
        records: list[dict[str, Any]] = []
        for heading in candidates:
            costs = [self.rollout(obs, path, mean, heading, i) for i, path in enumerate(paths)]
            expected = float(np.mean(costs))
            tail = upper_cvar(costs, cfg.policy.cvar_alpha)
            objective = (
                (1 - cfg.policy.risk_lambda) * expected + cfg.policy.risk_lambda * tail
                if cfg.policy.name == "scenario_mpc"
                else expected
            )
            records.append(
                {
                    "heading_rad": heading,
                    "sample_costs_s": costs,
                    "mean_s": expected,
                    "cvar_s": tail,
                    "objective_s": objective,
                }
            )
        selected = min(range(len(records)), key=lambda i: records[i]["objective_s"])
        digest = hashlib.sha256(b"".join(p.air.tobytes() for p in paths)).hexdigest()
        self.last_decision = {
            "candidates": records,
            "selected_index": selected,
            "conditional_paths_hash": digest,
            "planner_tick": self.tick,
            "sample_count": len(paths),
            "horizon_s": cfg.policy.planning_horizon_s,
            "shared_first_action": True,
            "recourse": (
                "delayed-mean-raw-heuristic-v1"
                if cfg.policy.name == "deterministic_mpc"
                and cfg.policy.deterministic_sensor_model == "expected"
                else "delayed-noisy-raw-heuristic-v1"
            ),
        }
        self.tick += 1
        return Action(target_heading_rad=candidates[selected])

    def rollout(
        self,
        obs: Observation,
        path: ConditionalPath,
        mean: ConditionalPath,
        heading: float,
        sample_index: int,
    ) -> float:
        cfg = self.config
        sim = Simulator(cfg, world=path, instruments_enabled=False)
        sim.state = BoatState(
            position=obs.position_xy_m,
            heading=obs.heading_rad,
            speed=obs.speed_water_m_s,
            elapsed=obs.elapsed_s,
            tack_remaining=obs.manoeuvre_remaining_s,
            turn_target=obs.executed_target_rad,
            requested=obs.previous_action_rad,
            tack_rate=(
                abs(wrap(obs.executed_target_rad - obs.heading_rad)) / obs.manoeuvre_remaining_s
                if obs.manoeuvre_active
                else 0
            ),
        )
        # Hidden tack anchors are not imported. They affect diagnostics, not future motion.
        recourse_config = cfg.policy.model_copy(
            update={
                "name": "polar_heuristic",
                "wind_input": "raw",
                "initial_side": 1
                if wrap(
                    obs.heading_rad - bearing(scale(observed_air(obs, cfg.policy.wind_input), -1))
                )
                >= 0
                else -1,
            }
        )
        recourse = PolarHeuristic(cfg.boat, recourse_config)
        recourse.last_switch = obs.elapsed_s  # approximate commitment restarted at each horizon
        sim.request(Action(target_heading_rad=heading))
        end = min(cfg.course.deadline_s, obs.elapsed_s + cfg.policy.planning_horizon_s)
        next_action = obs.elapsed_s + cfg.clock.policy_interval_s
        future_obs = obs
        while sim.state.status == "running" and sim.state.elapsed < end - 1e-8:
            sim.step(min(cfg.clock.physics_dt_s, end - sim.state.elapsed))
            if sim.state.status == "running" and sim.state.elapsed >= next_action - 1e-8:
                future_obs = self.recourse_observation(
                    obs, sim.state, path, sample_index, future_obs
                )
                sim.request(recourse.act(future_obs))
                next_action += cfg.clock.policy_interval_s
        s = sim.state
        if s.status == "finished":
            return s.elapsed
        if s.status != "running":
            return cfg.policy.failure_cost_s
        # Common terminal estimate: two-heading remaining-time geometry at forecast mean.
        air = subtract(mean.at(s.position, s.elapsed).air_m_s, obs.known_current_m_s)
        source = bearing(scale(air, -1))
        angle = sim.polar.best_upwind_angle(math.hypot(*air))
        velocity = sim.polar.speed(math.hypot(*air), angle)
        displacement = np.asarray(subtract(obs.target_xy_m, s.position))
        matrix = np.array(
            [
                np.asarray(heading_vector(source + side * angle)) * velocity
                + np.asarray(obs.known_current_m_s)
                for side in (-1, 1)
            ]
        ).T
        if abs(float(np.linalg.det(matrix))) < 1e-8:
            return cfg.policy.failure_cost_s
        durations = np.linalg.solve(matrix, displacement)
        if np.all(durations >= 0):
            remaining = float(durations.sum())
            if np.all(durations > 1e-8):
                remaining += max(
                    cfg.boat.minimum_tack_s, 2 * angle / math.radians(cfg.boat.turn_rate_deg_s)
                )
        else:
            direct = bearing(tuple(displacement))
            direct_speed = sim.polar.speed(math.hypot(*air), abs(wrap(direct - source)))
            ground = np.asarray(heading_vector(direct)) * direct_speed + np.asarray(
                obs.known_current_m_s
            )
            remaining = math.hypot(*displacement) / max(
                0.1, float(ground @ np.asarray(heading_vector(direct)))
            )
        return min(cfg.policy.failure_cost_s, s.elapsed + remaining + s.tack_remaining)

    def recourse_observation(
        self,
        initial: Observation,
        state: BoatState,
        path: ConditionalPath,
        sample_index: int,
        previous: Observation | None = None,
    ) -> Observation:
        cfg = self.config
        previous = initial if previous is None else previous
        air = previous.wind_air_m_s
        acquired, available, valid = (
            previous.measurement_time_s,
            previous.measurement_available_s,
            previous.measurement_valid,
        )
        first_tick = max(
            math.ceil(initial.elapsed_s / cfg.clock.sensor_interval_s),
            math.floor(
                (previous.elapsed_s - cfg.sensor.latency_s + 1e-8) / cfg.clock.sensor_interval_s
            )
            + 1,
        )
        last_tick = math.floor(
            (state.elapsed - cfg.sensor.latency_s + 1e-8) / cfg.clock.sensor_interval_s
        )
        for tick in range(first_tick, last_tick + 1):
            measured = tick * cfg.clock.sensor_interval_s
            rng = keyed_rng(
                cfg.root_seed, "optimiser", self.tick * 1_000_000 + sample_index * 10_000 + tick
            )
            draw_dropout = rng.random()
            draw_direction, draw_speed = rng.normal(), rng.normal()
            expected_instruments = (
                cfg.policy.name == "deterministic_mpc"
                and cfg.policy.deterministic_sensor_model == "expected"
            )
            dropped = not expected_instruments and draw_dropout < cfg.sensor.dropout_probability
            direction_error = (
                0
                if expected_instruments
                else math.radians(cfg.sensor.direction_noise_deg) * draw_direction
            )
            speed_error = 0 if expected_instruments else cfg.sensor.speed_noise_m_s * draw_speed
            if not dropped:
                truth = path.at(state.position, measured).air_m_s
                air = wind_air(
                    bearing(scale(truth, -1)) + direction_error,
                    max(0, math.hypot(*truth) + speed_error),
                )
                acquired, available, valid = measured, measured + cfg.sensor.latency_s, True
        return Observation(
            elapsed_s=state.elapsed,
            position_xy_m=state.position,
            heading_rad=state.heading,
            speed_water_m_s=state.speed,
            target_xy_m=initial.target_xy_m,
            bounds_xy_m=initial.bounds_xy_m,
            remaining_s=max(0, cfg.course.deadline_s - state.elapsed),
            wind_air_m_s=air,
            known_current_m_s=initial.known_current_m_s,
            measurement_time_s=acquired,
            measurement_available_s=available,
            measurement_valid=valid
            and state.elapsed - acquired <= cfg.sensor.max_measurement_age_s,
            manoeuvre_active=state.tack_remaining > 1e-8,
            manoeuvre_remaining_s=state.tack_remaining,
            previous_action_rad=state.requested,
            executed_target_rad=state.turn_target,
            information_set="noisy_local_known_current",
        )
