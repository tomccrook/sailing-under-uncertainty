"""Hash-locked independent cohorts; no policy fitting or selection after access."""

import hashlib
import importlib.metadata
import json
import multiprocessing
import time
import traceback
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from sail_lab.classical_study import scenario_config
from sail_lab.config import Contract, ExperimentConfig, PolicyConfig
from sail_lab.experiments import (
    file_hash,
    project_root,
    record_run,
    source_metadata,
    source_snapshot,
    write_json,
)
from sail_lab.learning_study import ProjectedHeuristic, evaluate_episode, learning_report
from sail_lab.planning import RecedingHorizonPlanner
from sail_lab.policies import PolarHeuristic, Policy
from sail_lab.training import (
    FAMILIES,
    PPOPolicy,
    TrainingConfig,
    make_scenarios,
    train_trial,
)


class EvaluationConfig(Contract):
    version: Literal["locked-cohorts-v1"] = "locked-cohorts-v1"
    nominal_per_family: int = Field(default=256, ge=1, le=1000)
    shift_per_slice: int = Field(default=32, ge=1, le=1000)
    classical_nominal_per_family: int = Field(default=32, ge=1, le=1000)
    classical_shift_per_slice: int = Field(default=4, ge=1, le=1000)
    workers: int = Field(default=4, ge=1, le=8)
    failure_cost_s: float = Field(default=2400, ge=2400, le=2400)
    bootstrap_count: int = Field(default=2000, ge=200, le=10000)
    max_wall_s: float = Field(default=2400, gt=0)


# One interpretable change per slice, with the process-appropriate family named.
SHIFTS: tuple[tuple[str, str, dict[str, dict[str, Any]]], ...] = (
    ("persistence_slow", "ou", {"world": {"direction_kappa_s_inv": 0.005}}),
    ("gust_strong", "ou", {"world": {"log_speed_sigma_sqrt_s": 0.08}}),
    ("regime_short", "regime", {"world": {"regime_mean_duration_s": 80, "regime_shift_deg": 30}}),
    ("spatial_short", "spatial", {"world": {"spatial_length_m": 150, "spatial_amplitude_deg": 20}}),
    ("current_east", "ou", {"world": {"current_ground_m_s": (0.4, 0)}}),
    ("sensor_delay", "ou", {"sensor": {"latency_s": 16}}),
    ("sensor_dropout", "ou", {"sensor": {"dropout_probability": 0.5}}),
    ("sensor_bias", "ou", {"sensor": {"direction_bias_deg": 15, "speed_bias_m_s": 1}}),
    ("polar_slow", "ou", {"boat": {"polar_scale": 0.8}}),
    ("no_go_wide", "ou", {"boat": {"no_go_deg": 45}}),
    ("tack_slow", "ou", {"boat": {"minimum_tack_s": 20}}),
    ("recovery_slow", "ou", {"boat": {"acceleration_tau_s": 15}}),
    ("forecast_overconfident", "ou", {"forecast": {"uncertainty_scale": 0.35}}),
    (
        "course_long",
        "ou",
        {"course": {"target_xy_m": (0, 1500), "bounds_xy_m": (-900, 900, -100, 1800)}},
    ),
    (
        "perfect_instruments",
        "ou",
        {
            "sensor": {
                "information_set": "perfect_local_known_current",
                "latency_s": 0,
                "dropout_probability": 0,
                "direction_noise_deg": 0,
                "speed_noise_m_s": 0,
            }
        },
    ),
    (
        "simplified_dynamics",
        "ou",
        {"boat": {"turn_rate_deg_s": 10000, "minimum_tack_s": 0.01, "acceleration_tau_s": 0.01}},
    ),
)


def cohorts(plan: EvaluationConfig) -> list[dict[str, Any]]:
    if (
        plan.classical_nominal_per_family > plan.nominal_per_family
        or plan.classical_shift_per_slice > plan.shift_per_slice
    ):
        raise ValueError("classical cohort must be a declared subset of the full cohort")
    rows = []
    for family_index, family in enumerate(FAMILIES):
        for i in range(plan.nominal_per_family):
            config = scenario_config(family, 300000 + family_index * 10000 + i, PolicyConfig())
            config = config.model_copy(update={"stage": "locked"})
            rows.append(
                {
                    "slice": "nominal",
                    "config": config.model_dump(mode="json"),
                    "classical": i < plan.classical_nominal_per_family,
                }
            )
    for slice_index, (label, family, updates) in enumerate(SHIFTS):
        for i in range(plan.shift_per_slice):
            config = scenario_config(family, 400000 + slice_index * 10000 + i, PolicyConfig())
            data = config.model_dump()
            for section, changes in updates.items():
                data[section].update(changes)
            config = ExperimentConfig.model_validate(data)
            config = config.model_copy(update={"stage": "locked"})
            rows.append(
                {
                    "slice": label,
                    "config": config.model_dump(mode="json"),
                    "classical": i < plan.classical_shift_per_slice,
                }
            )
    return rows


def verify_manifest(directory: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "succeeded":
        raise ValueError(f"incomplete parent: {directory}")
    for name, digest in manifest["artifacts"].items():
        if file_hash(directory / name) != digest:
            raise ValueError(f"parent artifact changed: {directory / name}")
    return manifest


def dependency_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("torch", "gymnasium", "stable-baselines3", "numpy", "pyarrow", "pydantic")
    }


def numerical_source_hash() -> str:
    """Documentation may evolve after a lock; numerical code/dependencies may not."""
    digest = hashlib.sha256(json.dumps(source_snapshot(), sort_keys=True).encode())
    for name in ("uv.lock", "pyproject.toml"):
        digest.update((project_root() / name).read_bytes())
    return digest.hexdigest()


def access(output: Path, event: str, **details: Any) -> None:
    with (output / "access-log.jsonl").open("a") as handle:
        handle.write(
            json.dumps({"utc": datetime.now(UTC).isoformat(), "event": event, **details}) + "\n"
        )


def prepare_lock(
    parent: Path,
    classical: Path,
    output: Path,
    plan: EvaluationConfig,
    progress: Callable[[str], None] = print,
) -> str:
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "preparation-plan.json",
        {
            "evaluation": plan.model_dump(mode="json"),
            "additional_training_seeds": [71004, 71005],
            "selection": "M4 mean across three seeds; retain both fixed and randomised training",
            "parent": str(parent.resolve()),
            "classical_parent": str(classical.resolve()),
        },
    )
    access(output, "preparation_started")
    try:
        verify_manifest(parent)
        verify_manifest(classical)
        selection = json.loads((parent / "selection.json").read_text())
        mode = selection["selected_features"]
        trials = json.loads((parent / "trials.json").read_text())
        models = [t for t in trials if t["variant"].startswith(f"PPO_{mode}_")]
        if len(models) != 6:
            raise ValueError(
                "selected development representation must have three seeds per training type"
            )
        for randomised in (False, True):
            label = f"PPO_{mode}_{'randomised' if randomised else 'fixed'}"
            template = next(t for t in models if t["variant"].startswith(label + "/"))
            training = TrainingConfig.model_validate(
                json.loads((Path(template["path"]) / "manifest.json").read_text())["config"]
            )
            for seed in (71004, 71005):
                cfg = training.model_copy(update={"seed": seed})
                path = output / "additional-training" / f"{label}-seed-{seed}"
                progress(f"Additional independent training: {label}/seed-{seed}")
                train_trial(path, cfg, make_scenarios("train", randomised=randomised))
                models.append(
                    {
                        "variant": f"{label}/seed-{seed}",
                        "path": str(path.resolve()),
                        "model_hash": file_hash(path / "model.zip"),
                    }
                )
        baseline = json.loads((parent / "B1_belief-selected.json").read_text())
        projected = json.loads((parent / "B1_matched-selected.json").read_text())
        frozen = json.loads((classical / "frozen-policies.json").read_text())
        params = {
            "B1_belief": baseline,
            "B1_matched": projected,
            "B1_raw": frozen["inherited_parent_policies"]["B1_raw"],
            "B2_mean": frozen["policies"]["B2_mean"],
            "B3_mean": frozen["inherited_parent_policies"]["B3_mean"],
            "B3_risk": frozen["inherited_parent_policies"]["B3_risk"],
        }
        write_json(output / "scenarios.json", cohorts(plan))
        write_json(output / "source-snapshot.json", source_snapshot())
        for model in models:
            trial = Path(model["path"])
            verify_manifest(trial)
            model["manifest_hash"] = file_hash(trial / "manifest.json")
        revision, dirty, source_hash = source_metadata()
        lock = {
            "version": 1,
            "stage": "locked_learning",
            "locked_utc": datetime.now(UTC).isoformat(),
            "source_hash": source_hash,
            "numerical_source_hash": numerical_source_hash(),
            "code_revision": revision,
            "dirty_tree": dirty,
            "dependency_versions": dependency_versions(),
            "dependency_lock_hash": file_hash(project_root() / "uv.lock"),
            "source_snapshot_hash": file_hash(output / "source-snapshot.json"),
            "scenario_manifest_hash": file_hash(output / "scenarios.json"),
            "evaluation": plan.model_dump(mode="json"),
            "models": models,
            "policies": params,
            "validation_manifest_hash": file_hash(parent / "validation-manifest.json"),
            "validation_path": str((parent / "validation-manifest.json").resolve()),
            "parent_manifest_hash": file_hash(parent / "manifest.json"),
            "classical_parent_manifest_hash": file_hash(classical / "manifest.json"),
            "analysis": (
                "paired whole worlds and joint seed/world resampling; seed=8852; fractional CVaR95"
            ),
            "process_control": "local lock and access log, not a security boundary",
        }
        assert_disjoint(lock, cohorts(plan))
        write_json(output / "lock.json", lock)
        digest = file_hash(output / "lock.json")
        (output / "lock.sha256").write_text(digest + "\n")
        access(output, "policies_and_scenarios_locked", lock_hash=digest)
        return digest
    except BaseException:
        (output / "preparation-error.txt").write_text(traceback.format_exc())
        access(output, "preparation_failed")
        raise


def assert_disjoint(lock: dict[str, Any], scenarios: list[dict[str, Any]]) -> None:
    validation = json.loads(Path(lock["validation_path"]).read_text())
    excluded = {c["root_seed"] for c in validation["scenarios"]}
    for model in lock["models"]:
        training = json.loads((Path(model["path"]) / "scenario-manifest.json").read_text())
        excluded.update(c["root_seed"] for c in training["scenarios"])
    roots = [r["config"]["root_seed"] for r in scenarios]
    if len(set(roots)) != len(roots) or set(roots) & excluded:
        raise ValueError("independent world roots overlap across splits or within evaluation")


def verify_lock(output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    digest = (output / "lock.sha256").read_text().strip()
    if digest != file_hash(output / "lock.json"):
        raise ValueError("lock changed")
    lock = json.loads((output / "lock.json").read_text())
    for name, key in (
        ("scenarios.json", "scenario_manifest_hash"),
        ("source-snapshot.json", "source_snapshot_hash"),
    ):
        if file_hash(output / name) != lock[key]:
            raise ValueError(f"locked artifact changed: {name}")
    if numerical_source_hash() != lock["numerical_source_hash"]:
        raise ValueError("numeric source changed after lock; require a new registration")
    if (
        dependency_versions() != lock["dependency_versions"]
        or file_hash(project_root() / "uv.lock") != lock["dependency_lock_hash"]
    ):
        raise ValueError("dependency environment changed after lock")
    if file_hash(Path(lock["validation_path"])) != lock["validation_manifest_hash"]:
        raise ValueError("validation manifest changed")
    for model in lock["models"]:
        trial = Path(model["path"])
        if file_hash(trial / "manifest.json") != model["manifest_hash"]:
            raise ValueError("model manifest changed after lock")
        verify_manifest(trial)
        if file_hash(trial / "model.zip") != model["model_hash"]:
            raise ValueError("model changed after lock")
    scenarios = json.loads((output / "scenarios.json").read_text())
    assert_disjoint(lock, scenarios)
    return lock, scenarios


def amend_before_access(output: Path, reason: str) -> str:
    """Preserve an earlier registration; never revise after evaluation access."""
    if (output / "evaluation").exists() or any(
        json.loads(line)["event"].startswith("evaluation_")
        for line in (output / "access-log.jsonl").read_text().splitlines()
    ):
        raise ValueError("evaluation already accessed; new independent cohort required")
    previous = (output / "lock.sha256").read_text().strip()
    if previous != file_hash(output / "lock.json"):
        raise ValueError("original registration changed")
    lock = json.loads((output / "lock.json").read_text())
    if file_hash(output / "scenarios.json") != lock["scenario_manifest_hash"]:
        raise ValueError("prepared scenario manifest changed")
    for model in lock["models"]:
        trial = Path(model["path"])
        if file_hash(trial / "manifest.json") != model["manifest_hash"]:
            raise ValueError("prepared model changed")
        verify_manifest(trial)
    original = output / "original-lock.json"
    if original.exists():
        raise ValueError("only one preaccess amendment supported; use a new registration")
    original.write_bytes((output / "lock.json").read_bytes())
    (output / "original-source-snapshot.json").write_bytes(
        (output / "source-snapshot.json").read_bytes()
    )
    write_json(output / "source-snapshot.json", source_snapshot())
    revision, dirty, source_hash = source_metadata()
    lock.update(
        source_hash=source_hash,
        numerical_source_hash=numerical_source_hash(),
        source_snapshot_hash=file_hash(output / "source-snapshot.json"),
        code_revision=revision,
        dirty_tree=dirty,
        registration_amendment={
            "previous_lock_hash": previous,
            "reason": reason,
            "utc": datetime.now(UTC).isoformat(),
            "policy_scenario_model_changes": False,
        },
    )
    write_json(output / "lock.json", lock)
    digest = file_hash(output / "lock.json")
    (output / "lock.sha256").write_text(digest + "\n")
    verify_lock(output)
    access(
        output,
        "preaccess_registration_amended",
        previous_lock_hash=previous,
        lock_hash=digest,
        reason=reason,
    )
    return digest


_models: dict[str, Policy] = {}
_params: dict[str, PolicyConfig] = {}


def _initialise(lock: dict[str, Any]) -> None:
    global _models, _params
    _models = {m["variant"]: PPOPolicy(Path(m["path"])) for m in lock["models"]}
    _params = {
        name: PolicyConfig.model_validate(params) for name, params in lock["policies"].items()
    }


def _classical_agent(config: ExperimentConfig, name: str, params: PolicyConfig) -> Policy:
    if name == "B1_matched":
        return ProjectedHeuristic(config, params)
    if params.name == "polar_heuristic":
        return PolarHeuristic(config.boat, params)
    return RecedingHorizonPlanner(config.model_copy(update={"policy": params}))


def _evaluate_world(row: dict[str, Any]) -> list[dict[str, Any]]:
    config = ExperimentConfig.model_validate(row["config"])
    results = []
    for name, params in _params.items():
        if not row["classical"] and params.name != "polar_heuristic":
            continue
        policy_config = config.model_copy(update={"policy": params})
        result = evaluate_episode(policy_config, _classical_agent(config, name, params), name)
        result.update(slice=row["slice"], classical_cohort=row["classical"])
        results.append(result)
    for name, model in _models.items():
        result = evaluate_episode(config, model, name)
        result.update(slice=row["slice"], classical_cohort=row["classical"])
        results.append(result)
    return results


def run_locked(output: Path, progress: Callable[[str], None] = print) -> dict[str, Any]:
    lock, scenarios = verify_lock(output)
    result_directory = output / "evaluation"
    result_directory.mkdir(exist_ok=False)
    lock_hash = file_hash(output / "lock.json")
    access(output, "evaluation_started", lock_hash=lock_hash)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    status = "failed"
    plan = EvaluationConfig.model_validate(lock["evaluation"])
    try:
        with ProcessPoolExecutor(
            max_workers=plan.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialise,
            initargs=(lock,),
        ) as executor:
            pending = iter(scenarios)
            futures = {
                executor.submit(_evaluate_world, next(pending))
                for _ in range(min(plan.workers, len(scenarios)))
            }
            completed = 0
            try:
                while futures:
                    if time.perf_counter() - started > plan.max_wall_s:
                        raise TimeoutError("locked evaluation wall budget exceeded")
                    done, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
                    for future in done:
                        futures.remove(future)
                        rows.extend(future.result())
                        completed += 1
                        following = next(pending, None)
                        if following is not None:
                            futures.add(executor.submit(_evaluate_world, following))
                        if completed % 32 == 0 or completed == len(scenarios):
                            pq.write_table(
                                pa.Table.from_pylist(rows),
                                result_directory / "partial-episodes.parquet",
                            )
                            progress(
                                f"Locked evaluation: {completed}/{len(scenarios)} worlds; "
                                f"{len(rows)} retained episodes"
                            )
            finally:
                for future in futures:
                    future.cancel()
        # Recheck before accepting results, including model/source hashes.
        verify_lock(output)
        expected = (
            len(scenarios) * (len(lock["models"]) + 3) + sum(r["classical"] for r in scenarios) * 3
        )
        if len(rows) != expected:
            raise ValueError("incomplete locked episode table")
        pq.write_table(pa.Table.from_pylist(rows), result_directory / "episodes.parquet")
        full = [r for r in rows if r["variant"] not in ("B2_mean", "B3_mean", "B3_risk")]
        report = learning_report(full, bootstrap_count=plan.bootstrap_count)
        report.update(stage="locked_learning", lock_hash=lock_hash, slices={})
        for label in [
            "nominal",
            "shifted",
            *[s[0] for s in SHIFTS],
            *[f"nominal_{f}" for f in FAMILIES],
        ]:
            subset = (
                [r for r in full if r["slice"] != "nominal"]
                if label == "shifted"
                else [r for r in full if r["slice"] == label]
            )
            if label.startswith("nominal_"):
                subset = [r for r in full if r["slice"] == "nominal" and r["family"] == label[8:]]
            report["slices"][label] = learning_report(subset, bootstrap_count=plan.bootstrap_count)
        classical = [r for r in rows if r["classical_cohort"] and "/seed-" not in r["variant"]]
        report["classical_cohort"] = learning_report(
            classical, bootstrap_count=plan.bootstrap_count
        )
        report["classical_cohort"]["paired_vs_B2"] = learning_report(
            classical, reference="B2_mean", bootstrap_count=plan.bootstrap_count
        )
        report["classical_cohort"]["paired_vs_B3_mean"] = learning_report(
            classical, reference="B3_mean", bootstrap_count=plan.bootstrap_count
        )
        report["compute"] = {
            "wall_s": time.perf_counter() - started,
            "workers": plan.workers,
            "device": "cpu",
            "latency_condition": "measured in concurrent local world workers",
        }
        report["episodes"] = len(rows)
        report["limitations"] = [
            "Synthetic model only; no real-world sailing trial.",
            "Deliberate nominal/shift mixture has no measured weather-frequency interpretation.",
            "Five training seeds are not five times as many independent worlds.",
            "Classical and robustness cohorts have insufficient tail precision for strong claims.",
            "New process families, matched compute and LSTM are not covered by this lock.",
        ]
        write_json(result_directory / "report.json", report)
        status = "succeeded"
    except BaseException:
        (result_directory / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        if rows:
            pq.write_table(
                pa.Table.from_pylist(rows), result_directory / "partial-episodes.parquet"
            )
        write_json(
            result_directory / "manifest.json",
            {
                "version": 1,
                "status": status,
                "lock_hash": lock_hash,
                "wall_s": time.perf_counter() - started,
                "retained_episodes": len(rows),
                "artifacts": {
                    p.name: file_hash(p)
                    for p in sorted(result_directory.iterdir())
                    if p.is_file() and p.name != "manifest.json"
                },
            },
        )
        access(
            output,
            "evaluation_completed" if status == "succeeded" else "evaluation_failed",
            retained_episodes=len(rows),
        )
    return report


def export_learning(study: Path, output: Path, public: Path) -> None:
    directory = study / "evaluation" if (study / "lock.json").exists() else study
    verify_manifest(directory)
    output.mkdir(parents=True, exist_ok=False)
    report = json.loads((directory / "report.json").read_text())
    trials = (
        json.loads((study / "trials.json").read_text())
        if directory == study
        else json.loads((study / "lock.json").read_text())["models"]
    )
    curves = []
    for trial in trials:
        path = Path(trial["path"])
        verify_manifest(path)
        episodes = json.loads((path / "training-episodes.json").read_text())
        steps = TrainingConfig.model_validate(
            json.loads((path / "manifest.json").read_text())["config"]
        ).total_steps
        points = []
        for step in range(8192, steps + 1, 8192):
            recent = [r for r in episodes if r["training_step"] <= step][-100:]
            if recent:
                points.append(
                    {
                        "steps": step,
                        "mean_cost_s": float(
                            np.mean(
                                [
                                    r["elapsed_s"] if r["status"] == "finished" else 2400
                                    for r in recent
                                ]
                            )
                        ),
                    }
                )
        if points:
            curves.append({"variant": trial["variant"], "points": points})
    report["curves"] = curves
    write_json(output / "learning.json", report)
    write_json(
        output / "manifest.json",
        {
            "version": 1,
            "source_manifest_hash": file_hash(directory / "manifest.json"),
            "artifacts": {"learning.json": file_hash(output / "learning.json")},
        },
    )
    public.mkdir(parents=True, exist_ok=True)
    (public / "learning.json").write_bytes((output / "learning.json").read_bytes())


def record_worst(study: Path, output: Path, public: Path) -> list[dict[str, Any]]:
    """Diagnostic replays after evaluation, never selection inputs."""
    lock, scenarios = verify_lock(study)
    verify_manifest(study / "evaluation")
    output.mkdir(parents=True, exist_ok=False)
    rows = pq.read_table(study / "evaluation/episodes.parquet").to_pylist()
    index = {f"{r['config']['world']['family']}-{r['config']['root_seed']}": r for r in scenarios}
    models = {m["variant"]: m for m in lock["models"]}
    params = {v: PolicyConfig.model_validate(p) for v, p in lock["policies"].items()}
    examples = []
    for variant in sorted({r["variant"] for r in rows}):
        selected = max(
            (r for r in rows if r["variant"] == variant),
            key=lambda r: (r["cost_s"], r["elapsed_s"]),
        )
        config = ExperimentConfig.model_validate(index[selected["scenario_id"]]["config"])
        model_id = None
        if variant in models:
            agent: Policy = PPOPolicy(Path(models[variant]["path"]))
            model_id = models[variant]["model_hash"]
        else:
            config = config.model_copy(update={"policy": params[variant]})
            agent = _classical_agent(config, variant, params[variant])
        path = record_run(
            config, output / "runs", policy=agent, policy_label=variant, model_id=model_id
        )
        summary = json.loads((path / "summary.json").read_text())
        if (
            summary["status"] != selected["status"]
            or abs(summary["elapsed_s"] - selected["elapsed_s"]) > 1e-8
        ):
            raise ValueError("diagnostic rerun disagrees with locked episode")
        examples.append(
            {
                "variant": variant,
                "scenario_id": selected["scenario_id"],
                "status": summary["status"],
                "elapsed_s": summary["elapsed_s"],
                "replay_path": str((path / "replay.json").resolve()),
            }
        )
    write_json(output / "examples.json", examples)
    public.mkdir(parents=True, exist_ok=True)
    # Show the first fixed-model seed's worst nominal/shift result, explicitly
    # chosen after evaluation as a diagnostic, not a favourable demo episode.
    example = next(e for e in examples if "_fixed/seed-" in e["variant"])
    (public / "replay.json").write_bytes(Path(example["replay_path"]).read_bytes())
    access(study, "worst_diagnostic_replays_recorded", examples=len(examples))
    return examples
