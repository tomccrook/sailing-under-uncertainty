"""Vector-state Kalman approximation using only available instruments.

State is ground-frame air velocity in m/s. Q=q*dt*I is a modelling assumption.
Delayed measurements inflate R by q*age rather than pretending to be current;
this is an approximate delay treatment, not an optimal fixed-lag smoother.
"""

import math

import numpy as np
from numpy.typing import NDArray

from sail_lab.config import BeliefConfig
from sail_lab.contracts import WindBelief, WindMeasurement


class VectorWindFilter:
    def __init__(self, config: BeliefConfig):
        self.config = config
        self.mean: NDArray[np.float64] = np.asarray(config.prior_air_m_s, dtype=np.float64)
        self.covariance: NDArray[np.float64] = np.eye(2) * config.prior_std_m_s**2
        self.elapsed = 0.0
        self.last_measurement = -math.inf

    def predict(self, elapsed_s: float) -> None:
        dt = elapsed_s - self.elapsed
        if dt < -1e-8:
            raise ValueError("filter cannot move backwards in time")
        self.covariance += np.eye(2) * self.config.process_variance_m2_s3 * max(0, dt)
        self.elapsed = elapsed_s

    def update(self, measurement: WindMeasurement) -> None:
        if measurement.available_s > self.elapsed + 1e-8:
            raise ValueError("cannot assimilate an unavailable measurement")
        if measurement.measured_s <= self.last_measurement:
            return
        z = np.asarray(measurement.air_m_s)
        speed = float(np.linalg.norm(z))
        radial = z / max(speed, 1e-8)
        if speed < 1e-8:
            radial = np.array([0.0, -1.0])
        transverse = np.array([-radial[1], radial[0]])
        noise = measurement.speed_noise_m_s**2 * np.outer(radial, radial) + (
            speed * math.radians(measurement.direction_noise_deg)
        ) ** 2 * np.outer(transverse, transverse)
        delay = max(0, self.elapsed - measurement.measured_s)
        noise += np.eye(2) * (
            self.config.instrument_variance_floor_m2_s2 + self.config.process_variance_m2_s3 * delay
        )
        gain = np.linalg.solve(self.covariance + noise, self.covariance).T
        self.mean += gain @ (z - self.mean)
        identity_minus_gain = np.eye(2) - gain
        self.covariance = (
            identity_minus_gain @ self.covariance @ identity_minus_gain.T + gain @ noise @ gain.T
        )
        self.covariance = (self.covariance + self.covariance.T) / 2
        self.last_measurement = measurement.measured_s

    def snapshot(self) -> WindBelief:
        return WindBelief(
            elapsed_s=self.elapsed,
            mean_air_m_s=(float(self.mean[0]), float(self.mean[1])),
            covariance_m2_s2=(
                (float(self.covariance[0, 0]), float(self.covariance[0, 1])),
                (float(self.covariance[1, 0]), float(self.covariance[1, 1])),
            ),
        )
