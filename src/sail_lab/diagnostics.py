"""Offline M2 diagnostics. Truth is used here for scoring, never for inference."""

import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from sail_lab.config import ExperimentConfig
from sail_lab.experiments import file_hash, source_metadata, source_snapshot, write_json
from sail_lab.forecasts import issue_forecast
from sail_lab.geometry import bearing, scale, wrap
from sail_lab.observations import ObservationSystem
from sail_lab.streams import keyed_rng, seed_hierarchy
from sail_lab.world import FrozenWorld


def world_mean_interval(values: list[float], seed: int = 18) -> dict[str, Any]:
    """World-block bootstrap, not a timestep-level confidence interval."""
    x = np.asarray(values)
    draws = keyed_rng(seed, "analysis", 0).choice(x, size=(2000, len(x)), replace=True).mean(axis=1)
    return {
        "mean": float(x.mean()),
        "world_bootstrap_95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "independent_worlds": len(x),
    }


def observation_diagnostics(output: Path, worlds_per_case: int = 8) -> dict[str, Any]:
    if worlds_per_case < 2:
        raise ValueError("diagnostics require at least two independent worlds per case")
    output.mkdir(parents=True, exist_ok=False)
    duration = 600
    station = (0.0, 500.0)
    cases = {
        "clean": {},
        "noisy": {"direction_noise_deg": 6, "speed_noise_m_s": 0.35},
        "delayed": {"direction_noise_deg": 6, "speed_noise_m_s": 0.35, "latency_s": 6},
        "dropout": {
            "direction_noise_deg": 6,
            "speed_noise_m_s": 0.35,
            "latency_s": 6,
            "dropout_probability": 0.5,
        },
        "biased": {
            "direction_noise_deg": 6,
            "speed_noise_m_s": 0.35,
            "direction_bias_deg": 12,
            "speed_bias_m_s": 0.5,
        },
    }
    rows: list[dict[str, Any]] = []
    forecast_rows: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []
    resolved: dict[str, Any] = {}
    for family in ("constant", "ou", "regime", "spatial"):
        for i in range(worlds_per_case):
            seed = 21000 + i + 1000 * ("constant", "ou", "regime", "spatial").index(family)
            base = ExperimentConfig.model_validate(
                {
                    "stage": "development",
                    "root_seed": seed,
                    "world": {"family": family},
                    "forecast": {"enabled": True},
                }
            )
            world = FrozenWorld(base.world, seed, duration)
            world.save(output / f"world-{family}-{seed}.npz")
            scenarios.append(
                {
                    "family": family,
                    "seed": seed,
                    "world_hash": world.digest,
                    "streams": seed_hierarchy(seed),
                }
            )
            for case, sensor in cases.items():
                cfg = ExperimentConfig.model_validate(
                    {
                        **base.model_dump(),
                        "sensor": {"information_set": "noisy_local_known_current", **sensor},
                    }
                )
                resolved[f"{family}/{case}"] = cfg.model_dump(mode="json")
                system = ObservationSystem(cfg, world)
                shift_time = -math.inf
                previous_direction: float | None = None
                for t in range(0, duration + 1, 2):
                    system.advance(t, station)
                    obs = system.snapshot(t, station, 0, 0, 0, 0, 0)
                    assert obs.belief is not None
                    truth = np.asarray(world.at(station, t).air_m_s)
                    mean = np.asarray(obs.belief.mean_air_m_s)
                    covariance = np.asarray(obs.belief.covariance_m2_s2)
                    error = truth - mean
                    mahalanobis = float(error @ np.linalg.solve(covariance, error))
                    source = bearing(scale(tuple(truth), -1))
                    if previous_direction is not None and abs(
                        wrap(source - previous_direction)
                    ) > math.radians(15):
                        shift_time = t
                    previous_direction = source
                    direction_error = math.degrees(wrap(bearing(scale(tuple(mean), -1)) - source))
                    raw_error = math.degrees(wrap(bearing(scale(obs.wind_air_m_s, -1)) - source))
                    rows.append(
                        {
                            "family": family,
                            "case": case,
                            "seed": seed,
                            "elapsed_s": t,
                            "direction_error_deg": direction_error,
                            "raw_direction_error_deg": raw_error,
                            "speed_error_m_s": float(np.linalg.norm(mean) - np.linalg.norm(truth)),
                            "covered_95": mahalanobis <= 5.991464547,
                            "gaussian_nll": 0.5
                            * (
                                2 * math.log(2 * math.pi)
                                + float(np.linalg.slogdet(covariance)[1])
                                + mahalanobis
                            ),
                            "measurement_valid": obs.measurement_valid,
                            "measurement_age_s": t - obs.measurement_time_s,
                            "seconds_since_shift": t - shift_time
                            if math.isfinite(shift_time)
                            else None,
                        }
                    )
            forecast_cases: dict[str, dict[str, float]] = {
                "nominal": {},
                "biased": {"direction_bias_deg": 12},
                "underdispersed": {"uncertainty_scale": 0.5},
            }
            for forecast_case, parameters in forecast_cases.items():
                fc = base.forecast.model_copy(update=parameters)
                for issue in range(0, duration - 120 + 1, 30):
                    packet = issue_forecast(fc, world, seed, issue, issue // 30)
                    for point in packet.points:
                        lead = point.valid_s - issue
                        if lead not in (0, 30, 60, 120):
                            continue
                        truth_air = world.at(station, point.valid_s).air_m_s
                        error = wrap(bearing(scale(truth_air, -1)) - point.wind_from_rad)
                        speed_error = math.hypot(*truth_air) - point.speed_m_s
                        z2 = (error / point.direction_std_rad) ** 2 + (
                            speed_error / point.speed_std_m_s
                        ) ** 2
                        forecast_rows.append(
                            {
                                "family": family,
                                "case": forecast_case,
                                "seed": seed,
                                "issued_s": issue,
                                "valid_s": point.valid_s,
                                "lead_s": lead,
                                "direction_error_deg": math.degrees(error),
                                "speed_error_m_s": speed_error,
                                "direction_covered_95": abs(error)
                                <= 1.959964 * point.direction_std_rad,
                                "joint_gaussian_nll": math.log(
                                    2 * math.pi * point.direction_std_rad * point.speed_std_m_s
                                )
                                + 0.5 * z2,
                            }
                        )
    pq.write_table(pa.Table.from_pylist(rows), output / "estimator.parquet")
    pq.write_table(pa.Table.from_pylist(forecast_rows), output / "forecast.parquet")
    estimator_report: dict[str, Any] = {}
    for family in ("constant", "ou", "regime", "spatial"):
        for case in cases:
            selected = [
                r
                for r in rows
                if r["family"] == family and r["case"] == case and r["elapsed_s"] >= 60
            ]
            per_world = []
            for seed in sorted({r["seed"] for r in selected}):
                sample = [r for r in selected if r["seed"] == seed]
                per_world.append(
                    {
                        "coverage": float(np.mean([r["covered_95"] for r in sample])),
                        "direction_rmse_deg": math.sqrt(
                            float(np.mean([r["direction_error_deg"] ** 2 for r in sample]))
                        ),
                        "raw_direction_rmse_deg": math.sqrt(
                            float(np.mean([r["raw_direction_error_deg"] ** 2 for r in sample]))
                        ),
                        "speed_rmse_m_s": math.sqrt(
                            float(np.mean([r["speed_error_m_s"] ** 2 for r in sample]))
                        ),
                        "nll": float(np.mean([r["gaussian_nll"] for r in sample])),
                    }
                )
            after_shift = [
                r
                for r in selected
                if r["seconds_since_shift"] is not None and r["seconds_since_shift"] <= 20
            ]
            estimator_report[f"{family}/{case}"] = {
                key: world_mean_interval([r[key] for r in per_world]) for key in per_world[0]
            }
            estimator_report[f"{family}/{case}"]["post_shift_first_20s_mae_deg"] = (
                float(np.mean([abs(r["direction_error_deg"]) for r in after_shift]))
                if after_shift
                else None
            )
    forecast_report: dict[str, Any] = {}
    for family in ("constant", "ou", "regime", "spatial"):
        for case in ("nominal", "biased", "underdispersed"):
            for lead in (0, 30, 60, 120):
                selected = [
                    r
                    for r in forecast_rows
                    if r["family"] == family and r["case"] == case and r["lead_s"] == lead
                ]
                coverages = [
                    float(
                        np.mean([r["direction_covered_95"] for r in selected if r["seed"] == seed])
                    )
                    for seed in sorted({r["seed"] for r in selected})
                ]
                forecast_report[f"{family}/{case}/{lead:g}s"] = world_mean_interval(coverages)
    report = {
        "stage": "development",
        "duration_s": duration,
        "burn_in_s": 60,
        "worlds_per_case": worlds_per_case,
        "estimator": estimator_report,
        "forecast_direction_coverage": forecast_report,
        "limitations": [
            "Gaussian vector filter is an approximate local random-walk model.",
            "Delayed measurements inflate R; no optimal fixed-lag smoothing.",
            "Forecast arithmetic-speed Gaussian differs from log-speed world.",
            "Post-shift errors are descriptive; uncertainty uses independent worlds.",
            "Nominal uncertainty is a model setting, not an assertion of calibration.",
        ],
    }
    write_json(output / "report.json", report)
    write_json(output / "configs.json", resolved)
    write_json(output / "scenarios.json", scenarios)
    write_json(output / "source-snapshot.json", source_snapshot())
    revision, dirty, source_hash = source_metadata()
    write_json(
        output / "manifest.json",
        {
            "stage": "development",
            "code_revision": revision,
            "dirty_tree": dirty,
            "source_snapshot_hash": source_hash,
            "platform": platform.platform(),
            "artifacts": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()},
        },
    )
    return report
