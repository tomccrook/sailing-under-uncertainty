import math

import numpy as np
import pytest

from sail_lab.config import BoatConfig
from sail_lab.dynamics.polar import SPEEDS_M_S, SyntheticPolar
from sail_lab.geometry import (
    heading_vector,
    knots_to_m_s,
    m_s_to_knots,
    segment_circle_fraction,
    wind_air,
    wrap,
)
from sail_lab.policies import ideal_reference, ideal_trajectory, two_leg_reference


@pytest.mark.parametrize(
    "source,expected",
    [
        (0, (0, -6)),
        (90, (-6, 0)),
        (180, (0, 6)),
        (270, (6, 0)),
    ],
)
def test_cardinal_wind(source: float, expected: tuple[float, float]) -> None:
    assert wind_air(math.radians(source), 6) == pytest.approx(expected, abs=1e-12)


def test_angles_and_units() -> None:
    assert heading_vector(math.pi / 2) == pytest.approx((1, 0))
    assert wrap(math.pi) == -math.pi
    assert wrap(-3 * math.pi) == -math.pi
    assert wrap(0.1 + 20 * math.pi) == pytest.approx(0.1)
    assert knots_to_m_s(1) == pytest.approx(1852 / 3600)
    assert m_s_to_knots(knots_to_m_s(12.5)) == pytest.approx(12.5)


def test_crossing_not_endpoint_only() -> None:
    assert segment_circle_fraction((-20, 0), (20, 0), (0, 0), 1) == pytest.approx(0.475)
    assert segment_circle_fraction((-20, 1), (20, 1), (0, 0), 1) == pytest.approx(0.5)
    assert segment_circle_fraction((-20, 2), (20, 2), (0, 0), 1) is None
    assert segment_circle_fraction((0, 0), (0, 0), (0, 0), 1) == 0


def test_polar_domain_continuity_boundedness_and_vmg() -> None:
    polar = SyntheticPolar(BoatConfig())
    assert polar.speed(6, math.radians(45)) == 2.8
    assert polar.speed(0, math.pi / 2) == 0
    assert polar.speed(6, math.radians(34)) == 0
    assert polar.speed(6, math.radians(35 + 1e-7)) < 1e-6
    assert polar.speed(100, math.pi / 2) == polar.speed(10, math.pi / 2)
    for wind in np.linspace(0, 20, 21):
        for angle in np.linspace(0, math.pi, 61):
            assert 0 <= polar.speed(float(wind), float(angle)) <= max(map(max, SPEEDS_M_S))
    optimum = polar.best_upwind_angle(6)
    assert math.degrees(optimum) == pytest.approx(50, abs=0.15)
    assert polar.speed(6, optimum) * math.cos(optimum) > polar.speed(
        6, math.radians(40)
    ) * math.cos(math.radians(40))
    with pytest.raises(ValueError):
        polar.speed(-1, math.pi / 2)


@pytest.mark.parametrize("dt", [0.1, 1, 7, 60])
def test_independent_ideal_fixture_and_reflection(dt: float) -> None:
    reference = ideal_reference()
    rows = ideal_trajectory(dt)
    mirrored = ideal_trajectory(dt, -1)
    assert rows[-1] == pytest.approx((reference["arrival_s"], 0, 1000), abs=1e-8)
    assert reference["arrival_s"] == pytest.approx(471.404520791)
    assert reference["distance_m"] == pytest.approx(math.sqrt(2) * 1000)
    assert max(row[1] for row in rows) == pytest.approx(500, abs=1e-8)
    for (t, x, y), mirrored_row in zip(rows, mirrored, strict=True):
        assert mirrored_row == pytest.approx((t, -x, y), abs=1e-8)


@pytest.mark.parametrize("radius", [0, 10, 50])
def test_corresponding_polar_two_leg_reference(radius: float) -> None:
    angle, speed = math.radians(50), 3.2
    reference = two_leg_reference(1000, speed, angle, radius)
    first = reference["first_leg_s"] * speed
    total = reference["distance_m"]
    end = (math.sin(angle) * (2 * first - total), math.cos(angle) * total)
    assert math.dist(end, (0, 1000)) == pytest.approx(radius, abs=1e-8)
    assert reference["arrival_s"] == pytest.approx((1000 / math.cos(angle) - radius) / speed)
