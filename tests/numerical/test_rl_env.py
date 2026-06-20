import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from sail_lab.classical_study import scenario_config
from sail_lab.config import ExperimentConfig, PolicyConfig
from sail_lab.contracts import WindBelief
from sail_lab.dynamics.simulator import Simulator
from sail_lab.rl_env import FeatureEncoder, FeatureSpec, SailingEnv, headings, potential


def test_gym_contract_and_reset_determinism() -> None:
    configs = tuple(scenario_config("ou", 100000 + i, PolicyConfig()) for i in range(3))
    for mode in ("present", "belief", "history"):
        env = SailingEnv(configs, FeatureSpec(mode=mode))
        check_env(env, skip_render_check=True)
        initial, info = env.reset(seed=14)
        assert info == {}
        following = env.step(0)
        replay, _ = env.reset(seed=14)
        np.testing.assert_array_equal(initial, replay)
        np.testing.assert_array_equal(following[0], env.step(0)[0])
        assert following[4] == {}


def test_belief_cannot_leak_via_present_history_or_actions() -> None:
    obs = Simulator(scenario_config("ou", 100001, PolicyConfig())).observation()
    changed = obs.model_copy(
        update={
            "belief": WindBelief(
                elapsed_s=0, mean_air_m_s=(6, 0), covariance_m2_s2=((8, 0), (0, 8))
            )
        }
    )
    assert headings(obs) == headings(changed)
    for mode in ("present", "history"):
        spec = FeatureSpec(mode=mode)
        np.testing.assert_array_equal(
            FeatureEncoder(spec).encode(obs), FeatureEncoder(spec).encode(changed)
        )
    assert not np.array_equal(
        FeatureEncoder(FeatureSpec(mode="belief")).encode(obs),
        FeatureEncoder(FeatureSpec(mode="belief")).encode(changed),
    )


def test_history_reset_removes_all_previous_episode_frames() -> None:
    obs = Simulator(ExperimentConfig()).observation()
    encoder = FeatureEncoder(FeatureSpec(mode="history"))
    first = encoder.encode(obs)
    encoder.encode(obs.model_copy(update={"elapsed_s": 10}))
    encoder.reset()
    np.testing.assert_array_equal(first, encoder.encode(obs))
    assert np.count_nonzero(first[:-31]) == 0
    assert first[-1] == 1


@pytest.mark.parametrize("shaping", [False, True])
@pytest.mark.parametrize("outcome", ["deadline", "boundary", "finish"])
def test_returns_equal_total_cost_and_no_terminal_bootstrap(shaping: bool, outcome: str) -> None:
    from sail_lab.policies import PolarHeuristic

    config = ExperimentConfig.model_validate(
        {
            "course": {"deadline_s": 20 if outcome == "deadline" else 1800},
        }
    )
    env = SailingEnv((config,), shaping=shaping)
    env.reset(seed=1)
    assert env.sim is not None
    start_potential = potential(env.sim.observation(), False, shaping)
    policy = PolarHeuristic(config.boat, config.policy)
    policy.reset()
    total = 0.0
    for _ in range(200):
        obs = env.sim.observation()
        intended = policy.act(obs).target_heading_rad if outcome == "finish" else obs.heading_rad
        action = min(
            range(8), key=lambda i: abs(math.remainder(headings(obs)[i] - intended, 2 * math.pi))
        )
        _, reward, done, truncated, info = env.step(action)
        total += reward
        assert info == {} and not truncated
        if done:
            break
    assert done
    assert (
        env.sim.state.status
        == {
            "finish": "finished",
            "boundary": "boundary_violation",
            "deadline": "deadline_exceeded",
        }[outcome]
    )
    cost = env.sim.state.elapsed if outcome == "finish" else 2400
    assert total == pytest.approx(-cost / 100 - start_potential)


def test_potential_telescope_has_no_loop_or_turn_bonus() -> None:
    obs = Simulator(ExperimentConfig()).observation()
    path = [
        obs.model_copy(update={"position_xy_m": p, "heading_rad": h})
        for p, h in [((0, 0), 0), ((100, 100), math.pi), ((0, 0), 0), ((0, 0), -math.pi)]
    ]
    assert sum(
        potential(b, False, True) - potential(a, False, True)
        for a, b in zip(path[:-1], path[1:], strict=True)
    ) == pytest.approx(0)


def test_invalid_actions_and_terminated_episode_fail_explicitly() -> None:
    env = SailingEnv((ExperimentConfig.model_validate({"course": {"deadline_s": 10}}),))
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(8)
    env.step(0)
    with pytest.raises(RuntimeError):
        env.step(0)
