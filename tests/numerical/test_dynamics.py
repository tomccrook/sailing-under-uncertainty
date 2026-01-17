import math

import pytest

from sail_lab.config import ExperimentConfig
from sail_lab.contracts import Action
from sail_lab.dynamics.simulator import Simulator
from sail_lab.geometry import heading_vector, scale, wrap


def make_sim(payload: dict[str, object] | None = None, heading: float | None = None) -> Simulator:
    return Simulator(ExperimentConfig.model_validate(payload or {}), heading)


def test_exact_speed_and_integrated_acceleration() -> None:
    sim = make_sim(heading=math.pi / 2)
    for _ in range(10):
        sim.step()
    target = 4.5
    assert sim.state.speed == pytest.approx(target * (1 - math.exp(-10 / 5)))
    assert sim.state.position[0] == pytest.approx(target * (10 - 5 * (1 - math.exp(-2))))


def test_zero_wind_current_drift_and_relative_water_wind() -> None:
    sim = make_sim({"world": {"wind_speed_m_s": 0}}, heading=0)
    sim.step()
    assert sim.state.position == (0, 0)
    assert sim.state.speed == 0
    # Current equal to air speed leaves zero water-relative wind: pure ground drift.
    sim = make_sim({"world": {"wind_speed_m_s": 0.4, "current_ground_m_s": [0, -0.4]}})
    sim.step(10)
    assert sim.state.position == pytest.approx((0, -4))
    assert sim.state.speed == 0
    sim = make_sim({"world": {"current_ground_m_s": [0.5, 0]}})
    assert sim.water_air == (-0.5, -6)
    assert sim.water_wind_speed == pytest.approx(math.hypot(0.5, 6))


def test_finite_turn_completed_tack_latches_commands() -> None:
    sim = make_sim()
    old = sim.state.heading
    sim.request(Action(target_heading_rad=-old))
    assert sim.state.tacks == 0
    assert sim.state.tack_remaining >= 8
    sim.step()
    assert abs(wrap(sim.state.heading - old)) <= math.radians(8) + 1e-12
    assert sim.state.tacks == 0
    sim.request(Action(target_heading_rad=old))
    assert sim.state.turn_target == -old
    while sim.state.tack_remaining > 1e-9:
        sim.step()
        sim.request(Action(target_heading_rad=old)) if sim.state.tack_remaining > 1e-9 else None
    assert sim.state.tacks == 1
    assert sim.state.heading == pytest.approx(-old)
    assert sim.state.manoeuvre_time == pytest.approx(math.degrees(2 * old) / 8)
    sim.request(Action(target_heading_rad=-old))
    sim.step()
    assert sim.state.tacks == 1


def test_small_head_to_wind_noise_does_not_count_tacks() -> None:
    sim = make_sim(heading=0)
    for i in range(20):
        sim.request(Action(target_heading_rad=(-1) ** i * 0.02))
        sim.step()
    assert sim.state.tacks == 0


def test_minimum_manoeuvre_duration() -> None:
    sim = make_sim({"boat": {"minimum_tack_s": 30}})
    sim.request(Action(target_heading_rad=-sim.state.heading))
    for _ in range(29):
        sim.step()
    assert sim.state.tacks == 0
    sim.step()
    assert sim.state.tacks == 1
    assert sim.state.manoeuvre_time == pytest.approx(30)


def test_segment_finish_no_skip_and_no_boundary_clipping() -> None:
    sim = make_sim(heading=math.pi / 2)
    sim.state.position = (-20, 1000)
    sim.state.speed = 4.5
    sim.step(10)
    assert sim.state.status == "finished"
    assert sim.state.position == pytest.approx((-10, 1000))
    assert sim.state.elapsed == pytest.approx(10 / 4.5)
    with pytest.raises(RuntimeError):
        sim.step()
    sim = make_sim(heading=math.pi / 2)
    sim.state.position = (599, 0)
    sim.state.speed = 4.5
    sim.step()
    assert sim.state.status == "boundary_violation"
    assert sim.state.elapsed == pytest.approx(1 / 4.5)
    assert sim.state.position[0] == pytest.approx(600)


def test_tack_loss_and_recovery_against_corresponding_polar() -> None:
    sim = make_sim()
    target_speed = sim.polar.speed(6, abs(sim.state.heading))
    sim.state.speed = target_speed
    sim.request(Action(target_heading_rad=-sim.state.heading))
    while sim.state.tack_remaining > 1e-9:
        sim.step(0.25)
    assert sim.state.speed < target_speed
    for _ in range(40):
        sim.step()
    assert sim.state.speed == pytest.approx(target_speed, rel=1e-3)


def test_deadline_without_wind() -> None:
    sim = make_sim({"world": {"wind_speed_m_s": 0}, "course": {"deadline_s": 3}})
    sim.step(10)
    assert sim.state.status == "deadline_exceeded"
    assert sim.state.elapsed == 3


def test_constant_heading_displacement_includes_current_once() -> None:
    sim = make_sim({"world": {"current_ground_m_s": [0.3, 0.1]}}, math.pi / 2)
    beta = abs(wrap(sim.state.heading - sim.water_source))
    speed = sim.polar.speed(sim.water_wind_speed, beta)
    sim.state.speed = speed
    sim.step(2)
    water_displacement = scale(heading_vector(math.pi / 2), speed * 2)
    assert sim.state.position == pytest.approx((water_displacement[0] + 0.6, 0.2))
