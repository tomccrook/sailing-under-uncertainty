"""Fair validation tuning and a paired, held-out development pilot (never final)."""

import importlib.metadata
import itertools
import json
import math
import platform
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, model_validator

from sail_lab.config import Contract, ExperimentConfig, PolicyConfig, Positive
from sail_lab.experiments import (
    file_hash,
    project_root,
    record_run,
    simulate,
    source_metadata,
    source_snapshot,
    write_json,
)
from sail_lab.planning import upper_cvar
from sail_lab.streams import keyed_rng, seed_hierarchy
from sail_lab.world import FrozenWorld


class StudyConfig(Contract):
    stage: Literal["development_pilot"] = "development_pilot"
    validation_worlds_per_family: int = Field(default=2, ge=2)
    pilot_worlds_per_family: int = Field(default=8, ge=2)
    families: tuple[Literal["oscillating", "ou", "regime", "spatial"], ...] = (
        "oscillating",
        "ou",
        "regime",
        "spatial",
    )
    validation_seed_base: Literal[31000] = 31000
    pilot_seed_base: Literal[41000] = 41000
    horizon_grid_s: tuple[Positive, ...] = (40.0, 60.0)
    scenario_count: Literal[8] = 8
    deterministic_sensor_model: Literal["expected", "sampled"] = "expected"
    failure_cost_s: float = Field(default=2400.0, ge=2400.0, le=2400.0)
    cvar_alpha: float = Field(default=0.95, ge=0.95, le=0.95)

    @model_validator(mode="after")
    def search_is_valid(self) -> Self:
        if not self.families or len(set(self.families)) != len(self.families):
            raise ValueError("families must be nonempty and unique")
        if not self.horizon_grid_s or any(h < 10 for h in self.horizon_grid_s):
            raise ValueError("planning horizons must span at least one policy interval")
        return self


def scenario_config(family: str, seed: int, policy: PolicyConfig) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": f"{family}-{seed}-{policy.name}",
            "stage": "development",
            "root_seed": seed,
            "world": {"family": family},
            "sensor": {
                "information_set": "noisy_local_known_current",
                "direction_noise_deg": 6,
                "speed_noise_m_s": 0.35,
                "latency_s": 4,
                "dropout_probability": 0.05,
            },
            "forecast": {"enabled": True},
            "policy": policy.model_dump(),
        }
    )


def realised_cost(status: str, elapsed_s: float, failure_cost_s: float) -> float:
    if status in (
        "implementation_error",
        "invalid_numeric_state",
        "cancelled_experiment",
        "running",
    ):
        raise ValueError(f"non-sailing outcome must be investigated: {status}")
    if status not in ("finished", "boundary_violation", "deadline_exceeded"):
        raise ValueError(f"unknown outcome: {status}")
    return elapsed_s if status == "finished" else failure_cost_s


def paired_report(rows: list[dict[str, Any]], reference: str = "B1_belief") -> dict[str, Any]:
    if not rows:
        raise ValueError("no episodes to evaluate")
    for row in rows:
        expected = realised_cost(row["status"], row["elapsed_s"], 2400.0)
        if not math.isfinite(row["cost_s"]) or abs(row["cost_s"] - expected) > 1e-8:
            raise ValueError("cost disagrees with retained episode outcome")
        if not row["latencies_s"] or any(not math.isfinite(v) or v < 0 for v in row["latencies_s"]):
            raise ValueError("missing or invalid policy latency")
    ids = sorted({r["scenario_id"] for r in rows})
    names = sorted({r["variant"] for r in rows})
    indexed = {(r["scenario_id"], r["variant"]): r for r in rows}
    if len(indexed) != len(rows) or len(rows) != len(ids) * len(names):
        raise ValueError("paired episode table has duplicate or missing outcomes")
    for scenario in ids:
        if len({indexed[(scenario, name)]["world_hash"] for name in names}) != 1:
            raise ValueError("paired policies used different worlds")
    rng = keyed_rng(8841, "analysis", 0)
    bootstrap_indices = rng.integers(len(ids), size=(4000, len(ids)))
    baseline = np.array([indexed[(s, reference)]["cost_s"] for s in ids])
    results: dict[str, Any] = {}
    for name in names:
        sample = [indexed[(s, name)] for s in ids]
        costs = np.array([r["cost_s"] for r in sample])
        differences = costs - baseline
        ci = np.quantile(differences[bootstrap_indices].mean(axis=1), [0.025, 0.975]).tolist()
        tail_draws = [upper_cvar(costs[indices], 0.95) for indices in bootstrap_indices]
        results[name] = {
            "worlds": len(ids),
            "completions": sum(r["status"] == "finished" for r in sample),
            "mean_cost_s": float(costs.mean()),
            "cvar95_s": upper_cvar(costs, 0.95),
            "tail_effective_outcomes": len(ids) * 0.05,
            "cvar95_bootstrap_95": np.quantile(tail_draws, [0.025, 0.975]).tolist(),
            "paired_mean_difference_s": float(differences.mean()),
            "paired_mean_difference_95": ci,
            "reference": reference,
            "failure_statuses": {
                status: sum(r["status"] == status for r in sample)
                for status in ("finished", "boundary_violation", "deadline_exceeded")
            },
            "decision_latency_mean_s": float(
                np.mean([v for r in sample for v in r["latencies_s"]])
            ),
            "decision_latency_p95_s": float(
                np.quantile([v for r in sample for v in r["latencies_s"]], 0.95)
            ),
            "mean_tacks": float(np.mean([r["tack_count"] for r in sample])),
            "penalty_sensitivity_mean_cost_s": {
                str(penalty): float(
                    np.mean([realised_cost(r["status"], r["elapsed_s"], penalty) for r in sample])
                )
                for penalty in (1800.0, 2400.0, 3600.0)
            },
        }
    return {
        "stage": "development_pilot",
        "reference": reference,
        "policies": results,
        "analysis_root_seed": 8841,
        "bootstrap_unit": "paired independent world; fixed policies; pooled empirical family mix",
        "training_variation": "not applicable; no learned policies",
        "limitations": [
            "Exploratory held-out development worlds, not locked final tests.",
            "Small effective tail sample; CVaR cannot support a strong tail claim.",
            "Conditional OU planning model is misspecified for regimes and spatial fields.",
            "No measured boat/weather calibration or GPU benchmark.",
        ],
    }


DEFAULT_PLAN = StudyConfig()


def run_study(
    output: Path, plan: StudyConfig = DEFAULT_PLAN, progress: Callable[[str], None] = print
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC).isoformat()
    wall_start = time.perf_counter()
    write_json(output / "plan.json", plan.model_dump(mode="json"))
    write_json(output / "source-snapshot.json", source_snapshot())
    root = project_root()
    for name in ("uv.lock", "pyproject.toml"):
        if (root / name).exists():
            (output / name).write_bytes((root / name).read_bytes())
    scenarios: dict[str, list[tuple[str, int]]] = {}
    manifest_rows = []
    for split, count, base in (
        ("validation", plan.validation_worlds_per_family, 31000),
        ("pilot", plan.pilot_worlds_per_family, 41000),
    ):
        scenarios[split] = []
        for family_index, family in enumerate(plan.families):
            for i in range(count):
                seed = base + family_index * 1000 + i
                scenarios[split].append((family, seed))
                cfg = scenario_config(family, seed, PolicyConfig())
                world = FrozenWorld(cfg.world, seed, cfg.course.deadline_s)
                directory = output / "worlds"
                directory.mkdir(exist_ok=True)
                world.save(directory / f"{family}-{seed}.npz")
                manifest_rows.append(
                    {
                        "split": split,
                        "scenario_id": f"{family}-{seed}",
                        "root_seed": seed,
                        "family": family,
                        "world_hash": world.digest,
                        "streams": seed_hierarchy(seed),
                    }
                )
    write_json(output / "scenarios.json", manifest_rows)
    tuning: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    selected: dict[str, PolicyConfig] = {}

    def trial(variant: str, policy: PolicyConfig) -> float:
        costs = []
        for family, seed in scenarios["validation"]:
            cfg = scenario_config(family, seed, policy)
            write_json(output / "active-config.json", cfg.model_dump(mode="json"))
            summary, _, decisions = simulate(cfg)
            cost = realised_cost(summary.status, summary.elapsed_s, plan.failure_cost_s)
            costs.append(cost)
            tuning.append(
                {
                    "variant": variant,
                    "scenario_id": f"{family}-{seed}",
                    "policy": policy.model_dump(mode="json"),
                    "config_hash": cfg.content_hash(),
                    "summary": summary.model_dump(mode="json"),
                    "cost_s": cost,
                }
            )
            if policy.name != "polar_heuristic":
                folder = output / "validation-decisions"
                folder.mkdir(exist_ok=True)
                write_json(
                    folder / f"{variant}-{family}-{seed}-{policy.planning_horizon_s:g}.json",
                    decisions,
                )
        write_json(output / "tuning.json", tuning)
        return float(np.mean(costs))

    status = "failed"
    try:
        for wind_input in ("raw", "belief"):
            variant = f"B1_{wind_input}"
            trials = []
            for margin, commitment, guard in itertools.product(
                (5.0, 9.0, 15.0, 25.0), (20.0, 40.0), (0.0, 30.0)
            ):
                policy = PolicyConfig(
                    wind_input=wind_input,
                    layline_margin_s=margin,
                    commitment_s=commitment,
                    boundary_guard_s=guard,
                )
                score = trial(variant, policy)
                trials.append((score, policy))
            selected[variant] = min(trials, key=lambda item: item[0])[1]
            progress(
                f"{variant}: validation mean {min(t[0] for t in trials):.2f}s; "
                f"{len(trials)} settings"
            )
        for variant, name, risk in (
            ("B2_mean", "deterministic_mpc", 0.0),
            ("B3_mean", "scenario_mpc", 0.0),
            ("B3_risk", "scenario_mpc", 0.25),
        ):
            trials = []
            for horizon in plan.horizon_grid_s:
                policy = selected["B1_belief"].model_copy(
                    update={
                        "name": name,
                        "deterministic_sensor_model": plan.deterministic_sensor_model,
                        "planning_horizon_s": horizon,
                        "scenario_count": plan.scenario_count,
                        "risk_lambda": risk,
                    }
                )
                score = trial(variant, policy)
                trials.append((score, policy))
                progress(f"{variant}: horizon {horizon:g}s validation mean {score:.2f}s")
            selected[variant] = min(trials, key=lambda item: item[0])[1]
        # Freeze before any pilot outcome is opened or simulated.
        write_json(
            output / "frozen-policies.json",
            {
                "frozen_utc": datetime.now(UTC).isoformat(),
                "stage": "development_pilot",
                "selection_objective": "validation unconditional mean finish/failure cost",
                "policies": {
                    name: policy.model_dump(mode="json") for name, policy in selected.items()
                },
            },
        )
        for index, (pilot_family, seed) in enumerate(scenarios["pilot"]):
            for variant, policy in selected.items():
                cfg = scenario_config(pilot_family, seed, policy)
                write_json(output / "active-config.json", cfg.model_dump(mode="json"))
                folder = record_run(cfg, output / "runs")
                summary = json.loads((folder / "summary.json").read_text())
                manifest = json.loads((folder / "manifest.json").read_text())
                decisions = json.loads((folder / "decisions.json").read_text())
                episodes.append(
                    {
                        "scenario_id": f"{pilot_family}-{seed}",
                        "family": pilot_family,
                        "variant": variant,
                        "world_hash": manifest["scenario_manifest_hash"],
                        **summary,
                        "cost_s": realised_cost(
                            summary["status"], summary["elapsed_s"], plan.failure_cost_s
                        ),
                        "run_directory": str(folder),
                        "latencies_s": [d["compute_s"] for d in decisions],
                    }
                )
                pq.write_table(pa.Table.from_pylist(episodes), output / "episodes.parquet")
            progress(
                f"Pilot {index + 1}/{len(scenarios['pilot'])}: "
                f"{pilot_family}-{seed}; all five outcomes retained"
            )
        report = paired_report(episodes)
        report["families"] = {
            family: paired_report([r for r in episodes if r["family"] == family])
            for family in plan.families
        }
        write_json(output / "report.json", report)
        write_examples(output, episodes)
        plot_study(output, report)
        status = "succeeded"
    except BaseException:
        status = "failed"
        (output / "error.txt").write_text(traceback.format_exc())
        write_json(output / "tuning.json", tuning)
        raise
    finally:
        revision, dirty, source_hash = source_metadata()
        write_json(
            output / "manifest.json",
            {
                "stage": "development_pilot",
                "status": status,
                "started_utc": started,
                "ended_utc": datetime.now(UTC).isoformat(),
                "code_revision": revision,
                "dirty_tree": dirty,
                "source_snapshot_hash": source_hash,
                "platform": platform.platform(),
                "python": platform.python_version(),
                "elapsed_wall_s": time.perf_counter() - wall_start,
                "dependencies": {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "pydantic", "pyarrow", "matplotlib", "typer", "pyyaml")
                },
                "artifacts": {
                    str(p.relative_to(output)): file_hash(p)
                    for p in sorted(output.rglob("*"))
                    if p.is_file() and p != output / "manifest.json"
                },
            },
        )
    return report


def write_examples(output: Path, rows: list[dict[str, Any]]) -> None:
    successes = [r for r in rows if r["status"] == "finished"]
    failures = [r for r in rows if r["status"] != "finished"]
    examples = []
    for label, selected in (
        ("success", min(successes, key=lambda r: r["cost_s"]) if successes else None),
        ("failure", failures[0] if failures else None),
    ):
        if selected is None:
            continue
        directory = Path(selected["run_directory"])
        replay = json.loads((directory / "replay.json").read_text())
        last = replay["samples"][-1]
        examples.append(
            {
                "label": label,
                "scenario_id": selected["scenario_id"],
                "variant": selected["variant"],
                "status": selected["status"],
                "elapsed_s": selected["elapsed_s"],
                "cost_s": selected["cost_s"],
                "terminal_position_m": last["position_xy_m"],
                "tack_count": selected["tack_count"],
                "terminal_measurement_age_s": last["elapsed_s"]
                - last["observation"]["measurement_time_s"],
                "replay": str(directory / "replay.json"),
                "decisions": str(directory / "decisions.json"),
            }
        )
    write_json(output / "examples.json", examples)


def plot_study(output: Path, report: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    policies = report["policies"]
    labels = list(policies)
    differences = [policies[name]["paired_mean_difference_s"] for name in labels]
    ci = np.array([policies[name]["paired_mean_difference_95"] for name in labels])
    fig, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    axis.errorbar(
        differences,
        labels,
        xerr=np.array([np.array(differences) - ci[:, 0], ci[:, 1] - np.array(differences)]),
        fmt="o",
        color="#14776e",
        capsize=4,
    )
    axis.axvline(0, color="#777", linestyle="--")
    axis.set_xlabel("Mean finish/failure cost difference vs B1 belief (s); lower is better")
    axis.set_title(
        f"Development pilot: {policies[labels[0]]['worlds']} paired worlds, bootstrap 95% intervals"
    )
    fig.savefig(output / "paired-cost.svg")
    plt.close(fig)


def revise_mean_instruments(
    parent: Path, output: Path, progress: Callable[[str], None] = print
) -> dict[str, Any]:
    """Document a B2 modelling correction without erasing the original pilot.

    The existing pilot is already open: this amendment stays exploratory.
    Hyperparameter selection still reads validation outcomes only.
    """
    parent_manifest = json.loads((parent / "manifest.json").read_text())
    if parent_manifest["stage"] != "development_pilot" or parent_manifest["status"] != "succeeded":
        raise ValueError("only a completed development pilot can be amended")
    for name in ("episodes.parquet", "frozen-policies.json", "scenarios.json"):
        if file_hash(parent / name) != parent_manifest["artifacts"][name]:
            raise ValueError(f"parent artefact hash mismatch: {name}")
    frozen = json.loads((parent / "frozen-policies.json").read_text())
    if frozen["policies"]["B2_mean"].get("deterministic_sensor_model", "sampled") == "expected":
        raise ValueError("parent already uses expected instruments; no amendment needed")
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "plan.json",
        {
            "stage": "development_pilot",
            "parent": str(parent),
            "parent_manifest_hash": file_hash(parent / "manifest.json"),
            "reason": "B2 used one noisy future instrument draw; add expected instruments.",
            "horizon_grid_s": [40, 60],
            "selection": "validation unconditional mean cost only",
            "pilot_already_open": True,
            "retain_original_variant": "B2_sensor_draw",
        },
    )
    write_json(output / "source-snapshot.json", source_snapshot())
    for name in ("uv.lock", "pyproject.toml"):
        (output / name).write_bytes((project_root() / name).read_bytes())
    frozen = json.loads((parent / "frozen-policies.json").read_text())
    base = PolicyConfig.model_validate(frozen["policies"]["B2_mean"])
    scenarios = json.loads((parent / "scenarios.json").read_text())
    trials = []
    validation_rows = []
    episodes = pq.read_table(parent / "episodes.parquet").to_pylist()
    for row in episodes:
        if row["variant"] == "B2_mean":
            row["variant"] = "B2_sensor_draw"
    status = "failed"
    started = datetime.now(UTC).isoformat()
    wall_start = time.perf_counter()
    try:
        for horizon in (40.0, 60.0):
            policy = base.model_copy(
                update={"planning_horizon_s": horizon, "deterministic_sensor_model": "expected"}
            )
            costs = []
            for scenario in (s for s in scenarios if s["split"] == "validation"):
                cfg = scenario_config(scenario["family"], scenario["root_seed"], policy)
                write_json(output / "active-config.json", cfg.model_dump(mode="json"))
                summary, _, decisions = simulate(cfg)
                cost = realised_cost(summary.status, summary.elapsed_s, 2400.0)
                costs.append(cost)
                validation_rows.append(
                    {
                        "scenario_id": scenario["scenario_id"],
                        "policy": policy.model_dump(),
                        "summary": summary.model_dump(),
                        "cost_s": cost,
                    }
                )
                folder = output / "validation-decisions"
                folder.mkdir(exist_ok=True)
                write_json(folder / f"{scenario['scenario_id']}-{horizon:g}.json", decisions)
            score = float(np.mean(costs))
            trials.append((score, policy))
            write_json(output / "tuning.json", validation_rows)
            progress(f"B2 expected instruments: horizon {horizon:g}s validation mean {score:.2f}s")
        selected = min(trials, key=lambda item: item[0])[1]
        write_json(
            output / "frozen-policies.json",
            {
                "frozen_utc": datetime.now(UTC).isoformat(),
                "policies": {"B2_mean": selected.model_dump()},
                "inherited_parent_policies": frozen["policies"],
                "original_B2_sensor_model": "sampled",
            },
        )
        for index, scenario in enumerate(s for s in scenarios if s["split"] == "pilot"):
            cfg = scenario_config(scenario["family"], scenario["root_seed"], selected)
            folder = record_run(cfg, output / "runs")
            summary = json.loads((folder / "summary.json").read_text())
            manifest = json.loads((folder / "manifest.json").read_text())
            decisions = json.loads((folder / "decisions.json").read_text())
            episodes.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "family": scenario["family"],
                    "variant": "B2_mean",
                    "world_hash": manifest["scenario_manifest_hash"],
                    **summary,
                    "cost_s": realised_cost(summary["status"], summary["elapsed_s"], 2400.0),
                    "run_directory": str(folder),
                    "latencies_s": [d["compute_s"] for d in decisions],
                }
            )
            pq.write_table(pa.Table.from_pylist(episodes), output / "episodes.parquet")
            progress(f"Corrected B2 pilot {index + 1}/32: {scenario['scenario_id']}")
        report = paired_report(episodes)
        report["families"] = {
            family: paired_report([r for r in episodes if r["family"] == family])
            for family in sorted({r["family"] for r in episodes})
        }
        report["methodology_amendment"] = json.loads((output / "plan.json").read_text())
        report["limitations"].append(
            "B2 amendment followed opening the initial pilot; development evidence only."
        )
        write_json(output / "report.json", report)
        write_examples(output, episodes)
        plot_study(output, report)
        status = "succeeded"
    except BaseException:
        (output / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        revision, dirty, source_hash = source_metadata()
        write_json(
            output / "manifest.json",
            {
                "stage": "development_pilot",
                "status": status,
                "parent_manifest_hash": file_hash(parent / "manifest.json"),
                "code_revision": revision,
                "dirty_tree": dirty,
                "source_snapshot_hash": source_hash,
                "started_utc": started,
                "ended_utc": datetime.now(UTC).isoformat(),
                "elapsed_wall_s": time.perf_counter() - wall_start,
                "dependencies": {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "pydantic", "pyarrow", "matplotlib", "typer", "pyyaml")
                },
                "platform": platform.platform(),
                "python": platform.python_version(),
                "artifacts": {
                    str(p.relative_to(output)): file_hash(p)
                    for p in sorted(output.rglob("*"))
                    if p.is_file() and p != output / "manifest.json"
                },
            },
        )
    return report
