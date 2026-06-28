"""Independent planner replication using the frozen numerical environment.

Run with python -m scripts.planner_replication; generated artifacts remain local.
"""

import argparse
import json
import math
import multiprocessing
import subprocess
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

import sail_lab.planning as planning
from sail_lab.classical_study import realised_cost, scenario_config
from sail_lab.config import Contract, ExperimentConfig, PolicyConfig
from sail_lab.contracts import Action, Observation
from sail_lab.dynamics.simulator import Simulator
from sail_lab.experiments import file_hash, project_root, write_json
from sail_lab.learning_study import evaluate_episode
from sail_lab.locked_evaluation import dependency_versions, numerical_source_hash, verify_lock
from sail_lab.planning import RecedingHorizonPlanner, upper_cvar
from sail_lab.policies import PolarHeuristic, Policy
from sail_lab.training import FAMILIES

HORIZONS = (60, 120, 240, 480)
VARIANTS = ("B1_belief", "B2_mean", "B2_budget", "B3_mean", "B3_risk")
SLICES = ("nominal", "sensor_delay", "delay_tack")
BASES = (800000, 850000, 900000)
METHOD_FILES = ("scripts/planner_replication.py",)


class ReplicationConfig(Contract):
    version: Literal["planner-replication-v1"] = "planner-replication-v1"
    nominal_per_family: int = Field(default=256, ge=1, le=256)
    stress_per_family: int = Field(default=32, ge=1, le=256)
    workers: int = Field(default=4, ge=1, le=4)
    max_wall_s: int = Field(default=7200, ge=30, le=14400)
    bootstrap_count: int = Field(default=4000, ge=200, le=10000)


def method_hashes() -> dict[str, str]:
    return {name: file_hash(project_root() / name) for name in METHOD_FILES}


def event(directory: Path, name: str, **details: Any) -> None:
    with (directory / "access-log.jsonl").open("a") as handle:
        handle.write(
            json.dumps({"utc": datetime.now(UTC).isoformat(), "event": name, **details}) + "\n"
        )


def scenarios(plan: ReplicationConfig, validation: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slice_index, label in enumerate(("validation",) if validation else SLICES):
        count = (
            2
            if validation
            else (plan.nominal_per_family if label == "nominal" else plan.stress_per_family)
        )
        for family_index, family in enumerate(FAMILIES):
            for index in range(count):
                seed = (
                    (720000 + family_index * 1000 + index)
                    if validation
                    else (BASES[slice_index] + family_index * 10000 + index)
                )
                data = scenario_config(family, seed, PolicyConfig()).model_dump()
                data["stage"] = "development" if validation else "locked"
                if label in ("sensor_delay", "delay_tack"):
                    data["sensor"]["latency_s"] = 12
                if label == "delay_tack":
                    data["boat"]["minimum_tack_s"] = 16
                config = ExperimentConfig.model_validate(data)
                rows.append({"slice": label, "config": config.model_dump(mode="json")})
    roots = [r["config"]["root_seed"] for r in rows]
    if len(set(roots)) != len(rows):
        raise ValueError("duplicate world roots")
    return rows


class MeteredPolicy:
    """Delegating step counter, scoped to one process; never reads outer diagnostic state."""

    def __init__(
        self,
        config: ExperimentConfig,
        deadline: float = math.inf,
        stopped: Callable[[], bool] = lambda: False,
    ):
        self.config, self.deadline, self.stopped = config, deadline, stopped
        self.agent: Policy = (
            PolarHeuristic(config.boat, config.policy)
            if config.policy.name == "polar_heuristic"
            else RecedingHorizonPlanner(config)
        )
        self.reset()

    def reset(self) -> None:
        self.agent.reset()
        self.steps: list[int] = []
        self.step_calls: list[int] = []
        self.ceilings: list[int] = []
        self.rollouts: list[int] = []

    def check_budget(self) -> None:
        if self.stopped() or time.monotonic() > self.deadline:
            raise TimeoutError("replication cancelled or cooperative wall budget exhausted")

    def act(self, obs: Observation) -> Action:
        self.check_budget()
        if not isinstance(self.agent, RecedingHorizonPlanner):
            self.steps.append(0)
            self.step_calls.append(0)
            self.ceilings.append(0)
            self.rollouts.append(0)
            return self.agent.act(obs)
        owner = self
        count = 0
        calls = 0
        original = planning.__dict__["Simulator"]

        class CountedSimulator(Simulator):
            depth = 0

            def step(self, dt: float | None = None) -> None:
                nonlocal count, calls
                owner.check_budget()
                calls += 1
                if self.depth == 0:
                    count += 1
                self.depth += 1
                try:
                    super().step(dt)
                finally:
                    self.depth -= 1

        try:
            planning.__dict__["Simulator"] = CountedSimulator
            action = self.agent.act(obs)
        finally:
            planning.__dict__["Simulator"] = original
        decision = self.agent.last_decision
        rollouts = len(decision["candidates"]) * decision["sample_count"]
        remaining = min(self.config.policy.planning_horizon_s, obs.remaining_s)
        ceiling = rollouts * math.ceil(remaining / self.config.clock.physics_dt_s)
        if count > ceiling:
            raise ValueError("measured step count exceeds registered work ceiling")
        self.steps.append(count)
        self.step_calls.append(calls)
        self.ceilings.append(ceiling)
        self.rollouts.append(rollouts)
        return action


_deadline = math.inf
_stop: Any = None


def initialise(deadline: float, stop: Any) -> None:
    global _deadline, _stop
    _deadline, _stop = deadline, stop


def evaluate_world(
    scenario: dict[str, Any],
    policies: dict[str, dict[str, Any]],
    output: str,
    offset: int,
) -> int:
    directory = Path(output)
    config = ExperimentConfig.model_validate(scenario["config"])
    labels = list(policies)
    labels = labels[offset % len(labels) :] + labels[: offset % len(labels)]
    completed = 0
    try:
        for label in labels:
            cfg = config.model_copy(update={"policy": PolicyConfig.model_validate(policies[label])})
            agent = MeteredPolicy(cfg, _deadline, lambda: _stop is not None and _stop.is_set())
            row = evaluate_episode(cfg, agent, label)
            row.update(
                slice=scenario["slice"],
                model_steps=agent.steps,
                model_step_calls=agent.step_calls,
                model_step_ceilings=agent.ceilings,
                model_rollouts=agent.rollouts,
            )
            target = directory / f"{config.root_seed}-{label}.json"
            with target.open("x") as handle:
                handle.write(json.dumps(row, indent=2) + "\n")
            completed += 1
    except BaseException:
        (directory / f"{config.root_seed}-error.txt").write_text(traceback.format_exc())
        raise
    return completed


def execute(
    rows: list[dict[str, Any]],
    policies: dict[str, dict[str, Any]],
    output: Path,
    workers: int,
    max_wall_s: int,
    progress: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    output.mkdir(exist_ok=False)
    context = multiprocessing.get_context("spawn")
    stop = context.Event()
    start = time.monotonic()
    deadline = start + max_wall_s
    event(output, "execution_started", worlds=len(rows), policies=list(policies))
    status = "failed"
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=initialise,
            initargs=(deadline, stop),
        ) as executor:
            pending = iter(enumerate(rows))
            futures = set()
            for _ in range(min(workers, len(rows))):
                offset, row = next(pending)
                futures.add(executor.submit(evaluate_world, row, policies, str(output), offset))
            completed = 0
            try:
                while futures:
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            "replication wall budget exhausted; no partial statistics"
                        )
                    done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                    for future in done:
                        future.result()
                        futures.remove(future)
                        completed += 1
                        following = next(pending, None)
                        if following is not None:
                            offset, row = following
                            futures.add(
                                executor.submit(evaluate_world, row, policies, str(output), offset)
                            )
                    if done and (completed % 16 == 0 or completed == len(rows)):
                        progress(f"{completed}/{len(rows)} paired worlds complete")
            finally:
                if futures:
                    stop.set()
                    for future in futures:
                        future.cancel()
        result = [json.loads(p.read_text()) for p in sorted(output.glob("*-*.json"))]
        validate_rows(result, rows, policies)
        status = "succeeded"
        return result
    except BaseException:
        stop.set()
        (output / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        retained = [json.loads(p.read_text()) for p in sorted(output.glob("*-*.json"))]
        if retained:
            pq.write_table(pa.Table.from_pylist(retained), output / "episodes.parquet")
        event(
            output,
            "execution_completed" if status == "succeeded" else "execution_failed",
            retained_episodes=len(retained),
        )
        write_json(
            output / "manifest.json",
            {
                "status": status,
                "wall_s": time.monotonic() - start,
                "retained_episodes": len(retained),
                "expected_episodes": len(rows) * len(policies),
                "artifacts": {
                    p.name: file_hash(p)
                    for p in sorted(output.iterdir())
                    if p.is_file() and p.name != "manifest.json"
                },
            },
        )


def validate_rows(
    rows: list[dict[str, Any]],
    registered: list[dict[str, Any]],
    policies: dict[str, dict[str, Any]],
) -> None:
    expected = {(r["config"]["root_seed"], p) for r in registered for p in policies}
    index = {(r["root_seed"], r["variant"]): r for r in rows}
    if set(index) != expected or len(index) != len(rows):
        raise ValueError("missing, extra or duplicate registered episodes")
    for scenario in registered:
        base = ExperimentConfig.model_validate(scenario["config"])
        paired = [index[(base.root_seed, p)] for p in policies]
        if len({r["world_hash"] for r in paired}) != 1:
            raise ValueError("unpaired worlds")
        for label in policies:
            row = index[(base.root_seed, label)]
            config = base.model_copy(
                update={"policy": PolicyConfig.model_validate(policies[label])}
            )
            if row["config_hash"] != config.content_hash() or row["slice"] != scenario["slice"]:
                raise ValueError("episode configuration differs from registration")
            if not math.isfinite(row["cost_s"]) or row["cost_s"] != realised_cost(
                row["status"], row["elapsed_s"], 2400
            ):
                raise ValueError("invalid cost or implementation fault")
            latency = row["latencies_s"]
            if not latency or any(not math.isfinite(t) or t < 0 for t in latency):
                raise ValueError("missing or invalid latency")
            if not (
                len(latency)
                == len(row["model_steps"])
                == len(row["model_step_ceilings"])
                == len(row["model_step_calls"])
                == len(row["model_rollouts"])
            ):
                raise ValueError("missing work counters")
            if any(
                not isinstance(a, int) or not 0 <= a <= b
                for a, b in zip(row["model_steps"], row["model_step_ceilings"], strict=True)
            ):
                raise ValueError("invalid model work counters")


def validate(parent: Path, output: Path, workers: int = 4) -> dict[str, Any]:
    lock, _ = verify_lock(parent)
    output.mkdir(parents=True, exist_ok=False)
    candidates = {
        f"B2_h{h}": {**lock["policies"]["B2_mean"], "planning_horizon_s": h} for h in HORIZONS
    }
    candidates.update({p: lock["policies"][p] for p in ("B1_belief", "B3_mean", "B3_risk")})
    rows = scenarios(ReplicationConfig(), validation=True)
    plan = {
        "stage": "development_validation",
        "parent": str(parent.resolve()),
        "parent_lock_sha256": file_hash(parent / "lock.json"),
        "policies": candidates,
        "scenarios": rows,
        "method_hashes": method_hashes(),
        "numerical_source_sha256": numerical_source_hash(),
        "selection": "minimum mean cost then shorter horizon",
        "workers": workers,
        "max_wall_s": 1200,
    }
    write_json(output / "plan.json", plan)
    started = time.monotonic()
    results = execute(rows, candidates, output / "episodes", workers, 1200)
    means = {
        h: float(np.mean([r["cost_s"] for r in results if r["variant"] == f"B2_h{h}"]))
        for h in HORIZONS
    }
    selected = min(HORIZONS, key=lambda h: (means[h], h))
    selected_policy = candidates[f"B2_h{selected}"]
    report = {
        "stage": "development_validation",
        "selected_horizon_s": selected,
        "selected_policy": selected_policy,
        "candidate_mean_cost_s": means,
        "worlds": len(rows),
        "episodes": len(results),
        "wall_s": time.monotonic() - started,
        "method_hashes": method_hashes(),
        "timing": {
            p: {
                "wall_s_sum": sum(r["wall_s"] for r in results if r["variant"] == p),
                "decision_latency_p95_s": float(
                    np.quantile(
                        [t for r in results if r["variant"] == p for t in r["latencies_s"]], 0.95
                    )
                ),
            }
            for p in candidates
        },
    }
    if (
        plan["method_hashes"] != method_hashes()
        or plan["numerical_source_sha256"] != numerical_source_hash()
    ):
        raise ValueError("method changed during validation")
    write_json(output / "selection.json", report)
    write_json(
        output / "manifest.json",
        {
            "status": "succeeded",
            "artifacts": {
                "plan.json": file_hash(output / "plan.json"),
                "selection.json": file_hash(output / "selection.json"),
                "episodes/manifest.json": file_hash(output / "episodes/manifest.json"),
            },
        },
    )
    return report


def prepare(parent: Path, validation: Path, output: Path, plan: ReplicationConfig) -> str:
    from sail_lab.locked_evaluation import verify_manifest

    lock, old_scenarios = verify_lock(parent)
    verify_manifest(validation)
    verify_manifest(validation / "episodes")
    selection = json.loads((validation / "selection.json").read_text())
    if selection["method_hashes"] != method_hashes():
        raise ValueError("validation used different methods")
    policies = {p: lock["policies"][p] for p in VARIANTS if p != "B2_budget"}
    policies["B2_budget"] = selection["selected_policy"]
    policies = {p: policies[p] for p in VARIANTS}
    registered = scenarios(plan)
    forbidden = {r["config"]["root_seed"] for r in old_scenarios}
    forbidden.update(r["config"]["root_seed"] for r in scenarios(plan, validation=True))
    if any(r["config"]["root_seed"] in forbidden for r in registered):
        raise ValueError("test roots overlap historical or validation worlds")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "scenarios.json", registered)
    write_json(output / "plan.json", plan.model_dump(mode="json"))
    write_json(
        output / "source-snapshot.json", {p: (project_root() / p).read_text() for p in METHOD_FILES}
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record = {
        "version": "planner-replication-lock-v1",
        "registered_utc": datetime.now(UTC).isoformat(),
        "parent_path": str(parent.resolve()),
        "parent_lock_sha256": file_hash(parent / "lock.json"),
        "validation_path": str(validation.resolve()),
        "validation_manifest_sha256": file_hash(validation / "manifest.json"),
        "method_hashes": method_hashes(),
        "numerical_source_sha256": numerical_source_hash(),
        "dependency_versions": dependency_versions(),
        "git_head": revision,
        "dirty_tree": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "policies": policies,
        "files": {
            p: file_hash(output / p)
            for p in ("scenarios.json", "plan.json", "source-snapshot.json")
        },
        "analysis_seed": 885300,
        "analysis": "stratified paired world bootstrap; fractional CVaR95; pointwise intervals",
        "primary_slice": "nominal",
        "primary_contrasts": ["B2_budget", "B2_mean", "B3_mean", "B1_belief"],
        "scope": (
            "existing-family replication and matched forward-model work ceiling; "
            "no equal runtime or new-family claim"
        ),
    }
    write_json(output / "lock.json", record)
    digest = file_hash(output / "lock.json")
    (output / "lock.sha256").write_text(digest + "\n")
    event(output, "registered_before_test_access", lock_sha256=digest)
    return digest


def verify(output: Path) -> tuple[dict[str, Any], ReplicationConfig, list[dict[str, Any]]]:
    lock = json.loads((output / "lock.json").read_text())
    if file_hash(output / "lock.json") != (output / "lock.sha256").read_text().strip():
        raise ValueError("replication lock changed")
    if (
        lock["method_hashes"] != method_hashes()
        or lock["numerical_source_sha256"] != numerical_source_hash()
        or lock["dependency_versions"] != dependency_versions()
    ):
        raise ValueError("replication numerical methods/environment changed")
    for name, digest in lock["files"].items():
        if file_hash(output / name) != digest:
            raise ValueError("registered scenarios/config changed")
    parent = Path(lock["parent_path"])
    verify_lock(parent)
    if file_hash(parent / "lock.json") != lock["parent_lock_sha256"]:
        raise ValueError("original M5 lock changed")
    if (
        file_hash(Path(lock["validation_path"]) / "manifest.json")
        != lock["validation_manifest_sha256"]
    ):
        raise ValueError("validation manifest changed")
    return (
        lock,
        ReplicationConfig.model_validate_json((output / "plan.json").read_text()),
        json.loads((output / "scenarios.json").read_text()),
    )


def bootstrap_indices(rows: list[dict[str, Any]], count: int) -> np.ndarray[Any, Any]:
    rng = np.random.default_rng(885300)
    families = sorted({r["family"] for r in rows})
    return np.concatenate(
        [
            rng.choice(
                [i for i, r in enumerate(rows) if r["family"] == f],
                size=(count, sum(r["family"] == f for r in rows)),
                replace=True,
            )
            for f in families
        ],
        axis=1,
    )


def analyse(rows: list[dict[str, Any]], count: int) -> dict[str, Any]:
    ids = sorted({r["root_seed"] for r in rows})
    index = {(r["root_seed"], r["variant"]): r for r in rows}
    if len(index) != len(rows) or set(index) != {(i, v) for i in ids for v in VARIANTS}:
        raise ValueError("analysis requires every policy on every world")
    world_rows = [index[(i, "B1_belief")] for i in ids]
    draw = bootstrap_indices(world_rows, count)
    costs = {v: np.array([index[(i, v)]["cost_s"] for i in ids]) for v in VARIANTS}
    tail_draws = {
        v: np.array([upper_cvar(c[indices]) for indices in draw]) for v, c in costs.items()
    }
    candidate = costs["B3_risk"]
    contrasts = {}
    for ref in VARIANTS[:-1]:
        difference = candidate - costs[ref]
        contrasts[ref] = {
            "mean_difference_s": float(difference.mean()),
            "mean_difference_95": np.quantile(
                difference[draw].mean(axis=1), [0.025, 0.975]
            ).tolist(),
            "cvar95_difference_s": upper_cvar(candidate) - upper_cvar(costs[ref]),
            "cvar95_difference_95": np.quantile(
                tail_draws["B3_risk"] - tail_draws[ref], [0.025, 0.975]
            ).tolist(),
            "penalty_sensitivity": {
                str(p): {
                    "mean_difference_s": float(
                        np.mean(
                            [
                                realised_cost(
                                    index[(i, "B3_risk")]["status"],
                                    index[(i, "B3_risk")]["elapsed_s"],
                                    p,
                                )
                                - realised_cost(
                                    index[(i, ref)]["status"], index[(i, ref)]["elapsed_s"], p
                                )
                                for i in ids
                            ]
                        )
                    ),
                    "cvar95_difference_s": upper_cvar(
                        [
                            realised_cost(
                                index[(i, "B3_risk")]["status"],
                                index[(i, "B3_risk")]["elapsed_s"],
                                p,
                            )
                            for i in ids
                        ]
                    )
                    - upper_cvar(
                        [
                            realised_cost(
                                index[(i, ref)]["status"], index[(i, ref)]["elapsed_s"], p
                            )
                            for i in ids
                        ]
                    ),
                }
                for p in (1800, 2400, 3600)
            },
        }
    policies = {}
    for v in VARIANTS:
        sample = [index[(i, v)] for i in ids]
        failures = sum(r["status"] != "finished" for r in sample)
        latencies = [t for r in sample for t in r["latencies_s"]]
        policies[v] = {
            "worlds": len(ids),
            "mean_cost_s": float(costs[v].mean()),
            "cvar95_s": upper_cvar(costs[v]),
            "failure_counts": {
                s: sum(r["status"] == s for r in sample)
                for s in ("finished", "deadline_exceeded", "boundary_violation")
            },
            "zero_failure_upper95": 1 - 0.05 ** (1 / len(ids)) if failures == 0 else None,
            "decision_latency_mean_s": float(np.mean(latencies)),
            "decision_latency_p95_s": float(np.quantile(latencies, 0.95)),
            "model_steps": sum(sum(r["model_steps"]) for r in sample),
            "model_step_calls_including_splits": sum(sum(r["model_step_calls"]) for r in sample),
            "allocated_model_step_ceilings": sum(sum(r["model_step_ceilings"]) for r in sample),
            "model_rollouts": sum(sum(r["model_rollouts"]) for r in sample),
            "mean_tacks": float(np.mean([r["tack_count"] for r in sample])),
            "episode_wall_s_sum": sum(r["wall_s"] for r in sample),
        }
    return {
        "worlds": len(ids),
        "tail_effective_worlds": len(ids) * 0.05,
        "bootstrap_seed": 885300,
        "bootstrap_count": count,
        "bootstrap_unit": "paired worlds stratified by wind family",
        "policies": policies,
        "B3_risk_minus": contrasts,
    }


def run(output: Path) -> dict[str, Any]:
    lock, plan, registered = verify(output)
    if (output / "evaluation").exists():
        raise ValueError("evaluation already attempted; preserve original run")
    event(output, "test_access_started", lock_sha256=file_hash(output / "lock.json"))
    try:
        rows = execute(
            registered, lock["policies"], output / "evaluation", plan.workers, plan.max_wall_s
        )
        verify(output)
        report: dict[str, Any] = {
            "stage": "independent_locked_replication",
            "lock_sha256": file_hash(output / "lock.json"),
            "episodes": len(rows),
            "primary_slice": "nominal",
            "slices": {},
            "nominal_families": {},
            "precision_target_met": plan.nominal_per_family >= 256,
            "limitations": [
                "Synthetic existing wind families only; no empirical or new-family generalisation.",
                "Equal work ceilings do not mean equal actual work, horizons or wall times.",
                "Pointwise intervals and multiple endpoints; no simultaneous significance claim.",
                "Family/stress tail samples remain small; no outcome-based sample expansion.",
            ],
        }
        for label in SLICES:
            report["slices"][label] = analyse(
                [r for r in rows if r["slice"] == label], plan.bootstrap_count
            )
        for family in FAMILIES:
            report["nominal_families"][family] = analyse(
                [r for r in rows if r["slice"] == "nominal" and r["family"] == family],
                plan.bootstrap_count,
            )
        write_json(output / "report.json", report)
        event(output, "analysis_completed", retained_episodes=len(rows))
        write_json(
            output / "manifest.json",
            {
                "status": "succeeded",
                "artifacts": {
                    p: file_hash(output / p)
                    for p in (
                        "lock.json",
                        "lock.sha256",
                        "scenarios.json",
                        "plan.json",
                        "source-snapshot.json",
                        "evaluation/manifest.json",
                        "report.json",
                        "access-log.jsonl",
                    )
                },
            },
        )
        return report
    except BaseException:
        (output / "error.txt").write_text(traceback.format_exc())
        event(output, "test_access_failed")
        write_json(
            output / "manifest.json",
            {
                "status": "failed",
                "artifacts": {
                    p.name: file_hash(p)
                    for p in output.iterdir()
                    if p.is_file() and p.name != "manifest.json"
                },
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "prepare", "run", "verify"))
    parser.add_argument("--parent", type=Path, default=Path("artifacts/studies/m5"))
    parser.add_argument(
        "--validation", type=Path, default=Path("artifacts/studies/m5-replication-validation")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/studies/m5-replication"))
    parser.add_argument("--config", type=Path, default=Path("configs/replication/default.json"))
    args = parser.parse_args()
    value: Any
    if args.command == "validate":
        value = validate(args.parent, args.validation)
    elif args.command == "prepare":
        value = prepare(
            args.parent,
            args.validation,
            args.output,
            ReplicationConfig.model_validate_json(args.config.read_text()),
        )
    elif args.command == "run":
        value = run(args.output)
    else:
        lock, plan, rows = verify(args.output)
        value = {
            "lock_sha256": file_hash(args.output / "lock.json"),
            "worlds": len(rows),
            "plan": plan.model_dump(),
            "scope": lock["scope"],
        }
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
