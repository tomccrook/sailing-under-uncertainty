"""East/north metres; heading radians clockwise from north; wind is FROM."""

import math

from sail_lab.config import Vector

KNOT_M_S = 1852.0 / 3600.0


def wrap(angle: float) -> float:
    """Return [-pi, pi), including +pi -> -pi."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def heading_vector(angle: float) -> Vector:
    return math.sin(angle), math.cos(angle)


def bearing(vector: Vector) -> float:
    return math.atan2(vector[0], vector[1])


def wind_air(source_rad: float, speed_m_s: float) -> Vector:
    x, y = heading_vector(source_rad)
    return -speed_m_s * x, -speed_m_s * y


def knots_to_m_s(knots: float) -> float:
    return knots * KNOT_M_S


def m_s_to_knots(speed: float) -> float:
    return speed / KNOT_M_S


def add(a: Vector, b: Vector) -> Vector:
    return a[0] + b[0], a[1] + b[1]


def subtract(a: Vector, b: Vector) -> Vector:
    return a[0] - b[0], a[1] - b[1]


def scale(v: Vector, factor: float) -> Vector:
    return v[0] * factor, v[1] * factor


def segment_circle_fraction(
    start: Vector, end: Vector, centre: Vector, radius: float
) -> float | None:
    """Earliest contact in [0,1], including tangency and a start already inside."""
    delta, offset = subtract(end, start), subtract(start, centre)
    c = offset[0] ** 2 + offset[1] ** 2 - radius**2
    if c <= 0:
        return 0.0
    a = delta[0] ** 2 + delta[1] ** 2
    if a == 0:
        return None
    b = 2 * (offset[0] * delta[0] + offset[1] * delta[1])
    discriminant = b**2 - 4 * a * c
    if discriminant < -1e-10:
        return None
    root = (-b - math.sqrt(max(0.0, discriminant))) / (2 * a)
    return root if 0 <= root <= 1 else None


def boundary_exit_fraction(
    start: Vector, end: Vector, bounds: tuple[float, float, float, float]
) -> float | None:
    """First exit from closed rectangle; never clip a boat back into bounds."""
    left, right, bottom, top = bounds
    if not (left <= start[0] <= right and bottom <= start[1] <= top):
        return 0.0
    fractions: list[float] = []
    for a, b, lo, hi in ((start[0], end[0], left, right), (start[1], end[1], bottom, top)):
        if b < lo:
            fractions.append((lo - a) / (b - a))
        elif b > hi:
            fractions.append((hi - a) / (b - a))
    return min(fractions) if fractions else None
