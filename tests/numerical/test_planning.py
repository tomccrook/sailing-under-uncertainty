import math

import numpy as np
import pytest

from sail_lab.config import ExperimentConfig
from sail_lab.dynamics.simulator import BoatState, Simulator
from sail_lab.experiments import simulate
from sail_lab.planning import (
    RecedingHorizonPlanner,
    candidate_headings,
    conditional_paths,
    upper_cvar,
)
from sail_lab.policies import PolarHeuristic


def planner_config(name="scenario_mpc"):
    return ExperimentConfig.model_validate(
        {
            "policy": {
                "name": name,
                "wind_input": "belief",
                "planning_horizon_s": 20,
                "scenario_count": 4,
            },
            "sensor": {
                "information_set": "noisy_local_known_current",
                "direction_noise_deg": 6,
                "speed_noise_m_s": 0.35,
                "latency_s": 4,
            },
            "forecast": {"enabled": True},
        }
    )


def test_fractional_cvar_and_failure_mass() -> None:
    assert upper_cvar([1, 2, 3, 4], 0.625) == pytest.approx((4 + 0.5 * 3) / 1.5)
    assert upper_cvar([1, 2, 3, 2400], 0.95) == pytest.approx(2400)
    assert upper_cvar([7] * 8, 0.95) == pytest.approx(7)
    with pytest.raises(ValueError):
        upper_cvar([math.nan])


def test_planner_does_not_construct_evaluation_world(monkeypatch) -> None:
    cfg = planner_config()
    obs = Simulator(cfg).observation()

    def forbidden(*args, **kwargs):
        raise AssertionError("planner requested hidden world")

    monkeypatch.setattr("sail_lab.dynamics.simulator.FrozenWorld", forbidden)
    planner = RecedingHorizonPlanner(cfg)
    action = planner.act(obs)
    assert action.target_heading_rad in candidate_headings(obs, cfg)
    decision = planner.last_decision
    assert decision["shared_first_action"] and decision["sample_count"] == 4
    chosen = decision["candidates"][decision["selected_index"]]
    assert chosen["objective_s"] == min(r["objective_s"] for r in decision["candidates"])
    assert all(len(r["sample_costs_s"]) == 4 for r in decision["candidates"])
    planner.reset()
    assert planner.act(obs) == action and planner.last_decision == decision


def test_conditional_sampling_is_distinct_from_frozen_truth_and_uses_packet_uncertainty():
    cfg = planner_config()
    sim = Simulator(cfg)
    for _ in range(10):
        sim.step()
    obs = sim.observation()
    mean, paths = conditional_paths(obs, cfg, 0, False)
    assert any(p.at((0, 0), 20).air_m_s != sim.world.at((0, 0), 20).air_m_s for p in paths)
    assert mean.air.shape == paths[0].air.shape
    assert not np.array_equal(paths[0].air, paths[1].air)
    assert obs.forecast is not None
    points = tuple(
        p.model_copy(
            update={
                "direction_std_rad": p.direction_std_rad * 4,
                "speed_std_m_s": p.speed_std_m_s * 4,
            }
        )
        for p in obs.forecast.points
    )
    wider = obs.model_copy(update={"forecast": obs.forecast.model_copy(update={"points": points})})
    _, alternative = conditional_paths(wider, cfg, 0, False)
    assert not np.array_equal(paths[0].air[-1], alternative[0].air[-1])
    # B1 belief is always feasible in the common current-observation candidate set.
    action = PolarHeuristic(cfg.boat, cfg.policy).act(obs)
    assert action.target_heading_rad in candidate_headings(obs, cfg)


def test_recourse_queries_only_deliverable_hypothetical_measurements():
    cfg = planner_config()
    initial = Simulator(cfg).observation()
    planner = RecedingHorizonPlanner(cfg)

    class Spy:
        def __init__(self):
            self.times = []

        def at(self, pos, t):
            self.times.append(t)
            assert t + 4 <= 10
            from sail_lab.world import WindState

            return WindState(air_m_s=(0, -6), current_m_s=(0, 0))

    spy = Spy()
    state = BoatState(position=(0, 0), heading=0, elapsed=10)
    obs = planner.recourse_observation(initial, state, spy, 0)
    assert spy.times == [0, 2, 4, 6]
    assert obs.measurement_time_s == 6 and obs.measurement_available_s == 10


def test_stochastic_episode_and_planner_reproduce_with_failures_retained():
    cfg = planner_config("deterministic_mpc").model_copy(
        update={"course": planner_config().course.model_copy(update={"deadline_s": 20})}
    )
    a, samples_a, decisions_a = simulate(cfg)
    b, samples_b, decisions_b = simulate(cfg)
    assert a.status == b.status == "deadline_exceeded"
    assert samples_a == samples_b
    assert [d["planning"] for d in decisions_a] == [d["planning"] for d in decisions_b]


def test_deterministic_recourse_uses_expected_instruments_and_retains_delay():
    cfg = planner_config("deterministic_mpc")
    initial = Simulator(cfg).observation()
    _, paths = conditional_paths(initial, cfg, 0, True)
    state = BoatState(position=(0, 0), heading=0, elapsed=10)
    a = RecedingHorizonPlanner(cfg).recourse_observation(initial, state, paths[0], 0)
    b = RecedingHorizonPlanner(cfg).recourse_observation(initial, state, paths[0], 8)
    assert a == b
    assert a.wind_air_m_s == pytest.approx(paths[0].at((0, 0), 6).air_m_s)
    assert a.measurement_time_s == 6 and a.measurement_available_s == 10
    altered = cfg.model_copy(
        update={"policy": cfg.policy.model_copy(update={"deterministic_sensor_model": "sampled"})}
    )
    c = RecedingHorizonPlanner(altered).recourse_observation(initial, state, paths[0], 0)
    assert c.wind_air_m_s != a.wind_air_m_s
