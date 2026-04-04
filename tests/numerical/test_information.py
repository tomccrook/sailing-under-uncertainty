import math

import numpy as np
import pytest
from pydantic import ValidationError

from sail_lab.beliefs import VectorWindFilter
from sail_lab.config import (
    BeliefConfig,
    ExperimentConfig,
    ForecastConfig,
    SensorConfig,
    WorldConfig,
)
from sail_lab.contracts import WindBelief, WindMeasurement
from sail_lab.dynamics.simulator import Simulator
from sail_lab.forecasts import issue_forecast
from sail_lab.geometry import bearing, scale, wind_air, wrap
from sail_lab.observations import ObservationSystem
from sail_lab.world import FrozenWorld, WindState


def noisy(**kwargs: object) -> ExperimentConfig:
    return ExperimentConfig(
        sensor=SensorConfig.model_validate(
            {"information_set": "noisy_local_known_current", **kwargs}
        )
    )


def snapshot(system: ObservationSystem, t: float):
    return system.snapshot(t, (0, 0), 0, 0, 0, 0, 0)


def test_delivery_masks_staleness_and_prior_are_causal() -> None:
    cfg = noisy(latency_s=3, dropout_probability=1)
    system = ObservationSystem(cfg, FrozenWorld(cfg.world, cfg.root_seed, 30))
    for t in range(4):
        system.advance(t, (0, 0))
        obs = snapshot(system, t)
        assert not obs.measurement_valid
        assert obs.wind_air_m_s == cfg.belief.prior_air_m_s
        assert obs.latest_sensor_dropout == (t >= 3)
    assert np.trace(system.filter.covariance) > 8
    cfg = noisy(latency_s=3, max_measurement_age_s=3)
    system = ObservationSystem(cfg, FrozenWorld(cfg.world, cfg.root_seed, 30))
    for t in range(5):
        system.advance(t, (0, 0))
    obs = snapshot(system, 4)
    assert obs.measurement_time_s == 0 and obs.measurement_available_s == 3
    assert not obs.measurement_valid


def test_forecast_queries_only_issue_time_and_rejects_early_visibility() -> None:
    class Spy:
        def __init__(self):
            self.calls = []

        def at(self, position, t):
            self.calls.append((position, t))
            assert t == 10
            return WindState(air_m_s=(0, -6), current_m_s=(0, 0))

    spy = Spy()
    packet = issue_forecast(ForecastConfig(enabled=True), spy, 81, 10, 1)
    assert spy.calls == [((0, 500), 10)]
    obs = Simulator(ExperimentConfig()).observation()
    with pytest.raises(ValidationError, match="not yet available"):
        type(obs).model_validate({**obs.model_dump(), "forecast": packet.model_dump()})
    with pytest.raises(ValidationError, match="precede issue"):
        type(packet).model_validate({**packet.model_dump(), "available_s": 9})


def test_acquisition_clock_independent_of_physics_clock() -> None:
    cfg = noisy(direction_noise_deg=6, speed_noise_m_s=0.3, latency_s=3.5)
    cfg = cfg.model_copy(
        update={"world": WorldConfig(family="ou"), "forecast": ForecastConfig(enabled=True)}
    )
    a = Simulator(cfg)
    b = Simulator(
        cfg.model_copy(update={"clock": cfg.clock.model_copy(update={"physics_dt_s": 0.5})})
    )
    for t in range(1, 21):
        a.step()
        b.step()
        b.step()
        # Uniform temporal world: boat path cannot alter sensor or forecast innovations.
        assert a.observation().wind_air_m_s == b.observation().wind_air_m_s
        assert a.observation().belief.mean_air_m_s == pytest.approx(
            b.observation().belief.mean_air_m_s, abs=1e-12
        )
        assert np.allclose(
            a.observation().belief.covariance_m2_s2,
            b.observation().belief.covariance_m2_s2,
            atol=1e-12,
        )
        assert a.observation().forecast == b.observation().forecast
        assert a.observation().measurement_time_s <= t
        assert a.observation().measurement_available_s <= t
    before = a.instruments.filter.snapshot()
    for _ in range(20):
        a.observation()
    assert a.instruments.filter.snapshot() == before
    assert "diagnostic" not in a.observation().model_dump_json()
    assert "world" not in a.observation().model_dump()


def test_station_forecast_is_identical_on_different_spatial_routes() -> None:
    cfg = noisy(direction_noise_deg=6)
    cfg = cfg.model_copy(
        update={"world": WorldConfig(family="spatial"), "forecast": ForecastConfig(enabled=True)}
    )
    world = FrozenWorld(cfg.world, 81, 100)
    a, b = ObservationSystem(cfg, world), ObservationSystem(cfg, world)
    for t in range(31):
        a.advance(t, (-300, 0))
        b.advance(t, (300, 500))
        assert a.forecast == b.forecast
    assert a.measurement.air_m_s != b.measurement.air_m_s


def test_vector_filter_wrap_duplicate_updates_and_delay_uncertainty() -> None:
    f = VectorWindFilter(BeliefConfig())
    for i, angle in enumerate((359, 1, 359, 1)):
        f.predict(i * 2)
        f.update(
            WindMeasurement(
                measured_s=i * 2,
                available_s=i * 2,
                air_m_s=wind_air(math.radians(angle), 6),
                direction_noise_deg=5,
                speed_noise_m_s=0.3,
            )
        )
    assert abs(wrap(bearing(scale(f.snapshot().mean_air_m_s, -1)))) < math.radians(2)
    assert np.linalg.eigvalsh(f.covariance).min() > 0
    before = f.snapshot()
    f.update(
        WindMeasurement(
            measured_s=6, available_s=6, air_m_s=(6, 0), direction_noise_deg=5, speed_noise_m_s=0.3
        )
    )
    assert f.snapshot() == before
    f.predict(30)
    assert np.trace(f.covariance) > np.trace(before.covariance_m2_s2)
    with pytest.raises(ValueError, match="unavailable"):
        f.update(
            WindMeasurement(
                measured_s=30,
                available_s=40,
                air_m_s=(6, 0),
                direction_noise_deg=5,
                speed_noise_m_s=0.3,
            )
        )
    with pytest.raises(ValidationError, match="semidefinite"):
        WindBelief(elapsed_s=0, mean_air_m_s=(0, -6), covariance_m2_s2=((1, 2), (2, 1)))
