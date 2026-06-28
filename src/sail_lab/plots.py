"""Exportable scientific figures from full-resolution recorded samples."""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from sail_lab.config import ExperimentConfig
from sail_lab.contracts import ReplaySample
from sail_lab.dynamics.polar import SyntheticPolar
from sail_lab.geometry import subtract


def plot_run(config: ExperimentConfig, samples: list[ReplaySample], directory: Path) -> None:
    polar = SyntheticPolar(config.boat)
    angles = np.linspace(0, math.pi, 361)
    water_air = subtract(samples[0].diagnostic_wind_air_m_s, samples[0].diagnostic_current_m_s)
    wind_speed = math.hypot(*water_air)
    speeds = [polar.speed(wind_speed, float(angle)) for angle in angles]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    axes[0].plot(
        [row.position_xy_m[0] for row in samples], [row.position_xy_m[1] for row in samples]
    )
    target = config.course.target_xy_m
    axes[0].add_patch(Circle(target, config.course.finish_radius_m, fill=False, color="black"))
    axes[0].set(xlabel="East (m)", ylabel="North (m)", title="Recorded synthetic trajectory")
    axes[0].set_aspect("equal")
    axes[1].plot(np.degrees(angles), speeds, label="Speed through water")
    axes[1].plot(np.degrees(angles), np.asarray(speeds) * np.cos(angles), label="Upwind VMG")
    axes[1].set(
        xlabel="Water-relative wind angle (degrees)",
        ylabel="m/s",
        title="Illustrative polar, uncalibrated",
    )
    axes[1].legend()
    fig.savefig(directory / "trajectory-polar.svg")
    plt.close(fig)
