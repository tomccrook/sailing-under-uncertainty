"""Original synthetic table, not calibrated for any boat class.

Bilinear bounded interpolation; linear low-wind ramp to zero; high wind clamps.
Angles below no-go are zero, with a continuous ramp to the first sailing knot.
The table is embedded so installed packages use the identical model as the CLI.
"""

import math

import numpy as np
from numpy.typing import NDArray

from sail_lab.config import BoatConfig

WIND_SPEEDS = (0.0, 3.0, 6.0, 10.0)
ANGLES_DEG = (35.0, 40.0, 45.0, 50.0, 60.0, 90.0, 120.0, 150.0, 180.0)
SPEEDS_M_S = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 1.4, 1.9, 2.2, 2.6, 3.0, 2.8, 2.3, 1.8),
    (0.0, 2.0, 2.8, 3.2, 3.8, 4.5, 4.2, 3.5, 2.7),
    (0.0, 2.5, 3.4, 3.9, 4.7, 5.6, 5.2, 4.3, 3.3),
)


class SyntheticPolar:
    def __init__(self, config: BoatConfig):
        self.config = config
        # Shifting the first knot changes the no-go domain without discontinuity.
        self.angles: NDArray[np.float64] = np.asarray(ANGLES_DEG)
        if config.no_go_deg >= self.angles[1]:
            self.angles = np.linspace(config.no_go_deg, 180.0, len(ANGLES_DEG))
        else:
            self.angles = self.angles.copy()
            self.angles[0] = config.no_go_deg

    def speed(self, wind_m_s: float, beta_rad: float) -> float:
        if not math.isfinite(wind_m_s) or not math.isfinite(beta_rad):
            raise ValueError("polar inputs must be finite")
        if wind_m_s < 0 or not 0 <= beta_rad <= math.pi:
            raise ValueError("polar requires nonnegative wind and beta in [0,pi]")
        beta = math.degrees(beta_rad)
        if wind_m_s == 0 or beta <= self.config.no_go_deg:
            return 0.0
        rows = [float(np.interp(beta, self.angles, row)) for row in SPEEDS_M_S]
        return float(np.interp(wind_m_s, WIND_SPEEDS, rows)) * self.config.polar_scale

    def best_upwind_angle(self, wind_m_s: float) -> float:
        if not math.isfinite(wind_m_s) or wind_m_s < 0:
            raise ValueError("wind speed must be finite and nonnegative")
        angles = np.linspace(math.radians(self.config.no_go_deg), math.pi / 2, 551)
        knots = [
            float(np.interp(wind_m_s, WIND_SPEEDS, column))
            for column in zip(*SPEEDS_M_S, strict=True)
        ]
        vmgs = np.interp(np.degrees(angles), self.angles, knots) * np.cos(angles)
        return float(angles[int(np.argmax(vmgs))])
