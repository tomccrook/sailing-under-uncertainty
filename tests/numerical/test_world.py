import math

import numpy as np
import pytest

from sail_lab.config import ExperimentConfig, WorldConfig
from sail_lab.dynamics.simulator import Simulator
from sail_lab.geometry import bearing, scale, wrap
from sail_lab.streams import keyed_rng
from sail_lab.world import FrozenWorld, ou_path


def test_exact_ou_stationarity_and_transition() -> None:
    rng = np.random.default_rng(724)
    values = ou_path(rng, 20000, 50, 2, 0.3, 0.8)
    assert float(np.mean(values)) == pytest.approx(2, abs=0.03)
    assert float(np.var(values)) == pytest.approx(0.8**2 / (2 * 0.3), rel=0.04)
    constant = ou_path(rng, 10, 1, 3, 0.2, 0)
    assert np.all(constant == 3)
    # Independent paths sampled at a common elapsed time have the same marginal law.
    fine = np.array([ou_path(rng, 3, 1, 0, 0.3, 0.8)[-1] for _ in range(5000)])
    coarse = np.array([ou_path(rng, 2, 2, 0, 0.3, 0.8)[-1] for _ in range(5000)])
    assert float(np.mean(fine)) == pytest.approx(float(np.mean(coarse)), abs=0.05)
    assert float(np.var(fine)) == pytest.approx(float(np.var(coarse)), rel=0.08)


@pytest.mark.parametrize("family", ["constant", "oscillating", "ou", "regime", "spatial"])
def test_world_frozen_reproducible_bounded_and_query_order_independent(family: str) -> None:
    config = WorldConfig.model_validate({"family": family})
    world = FrozenWorld(config, 123, 500)
    repeated = FrozenWorld(config, 123, 500)
    assert world.digest == repeated.digest
    queries = [((0.0, 0.0), 0.0), ((300.0, 500.0), 13.0), ((-20.0, 700.0), 499.0)]
    values = [world.at(position, t) for position, t in queries]
    for _ in range(100):
        keyed_rng(123, "planner", 1).normal(size=100)
    assert [world.at(position, t) for position, t in reversed(queries)] == list(reversed(values))
    for value in values:
        assert 0 <= math.hypot(*value.air_m_s) <= 12
        deviation = math.degrees(wrap(bearing(scale(value.air_m_s, -1))))
        assert abs(deviation) <= 60 + 1e-8
    with pytest.raises(ValueError):
        world.direction[0] = 5
    with pytest.raises(ValueError, match="time domain"):
        world.at((0, 0), 501)


def test_spatial_continuity_and_temporal_interpolation() -> None:
    world = FrozenWorld(WorldConfig(family="spatial"), 911, 100)
    a = world.at((0, 0), 20).air_m_s
    b = world.at((1e-4, 1e-4), 20 + 1e-4).air_m_s
    assert math.dist(a, b) < 1e-4
    assert world.at((300, 0), 20) != world.at((0, 0), 20)
    ou = FrozenWorld(WorldConfig(family="ou"), 911, 100)
    expected = (ou.direction[10] + ou.direction[11]) / 2
    actual = math.degrees(bearing(scale(ou.at((0, 0), 21).air_m_s, -1)))
    assert actual == pytest.approx(expected)


def test_log_speed_mean_correction() -> None:
    config = WorldConfig(family="ou", log_speed_sigma_sqrt_s=0.12, max_speed_m_s=100, grid_dt_s=100)
    world = FrozenWorld(config, 618, 1_000_000)
    assert float(np.mean(world.speed)) == pytest.approx(6, rel=0.02)


def test_initial_heading_does_not_reveal_hidden_world_or_change_with_policy() -> None:
    config = ExperimentConfig.model_validate({"world": {"family": "ou"}})
    a = Simulator(config)
    altered = config.model_copy(
        update={"root_seed": 801, "policy": config.policy.model_copy(update={"initial_side": -1})}
    )
    b = Simulator(altered)
    assert a.air != b.air
    assert a.state.heading == b.state.heading == pytest.approx(math.radians(50))


def test_sensor_and_forecast_streams_are_counter_keyed() -> None:
    first = keyed_rng(9, "sensors", 25).normal(size=4)
    keyed_rng(9, "sensors", 10).normal(size=100)
    assert np.array_equal(first, keyed_rng(9, "sensors", 25).normal(size=4))
    assert not np.array_equal(first, keyed_rng(9, "forecasts", 25).normal(size=4))
