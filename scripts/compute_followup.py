"""Locked measured-work and new-process evaluation; preserves all older locks."""

import argparse
import json
import math
import multiprocessing
import subprocess
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from sail_lab.classical_study import realised_cost, scenario_config
from sail_lab.config import Contract, ExperimentConfig, PolicyConfig
from sail_lab.dynamics.simulator import Simulator
from sail_lab.experiments import file_hash, project_root, write_json
from sail_lab.geometry import wrap
from sail_lab.locked_evaluation import dependency_versions, numerical_source_hash, verify_manifest
from sail_lab.planning import upper_cvar
from sail_lab.training import FAMILIES
from scripts.followup_worlds import NEW_FAMILIES, NewWorldConfig, make_world
from scripts.planner_replication import MeteredPolicy, event
from scripts.planner_replication import verify as verify_parent

VARIANTS = ("B1_belief", "B2_matched", "B3_mean_matched", "B3_risk_matched", "B3_risk_original")
HORIZONS = (110, 120, 130, 140, 150)
METHOD_FILES = (
    "scripts/compute_followup.py",
    "scripts/followup_worlds.py",
    "scripts/planner_replication.py",
)
ANALYSIS_SEED = 887700


class FollowupConfig(Contract):
    version: Literal["matched-new-family-v1"] = "matched-new-family-v1"
    nominal_per_family: int = Field(default=256, ge=1, le=256)
    new_per_family: int = Field(default=512, ge=1, le=512)
    workers: int = Field(default=4, ge=1, le=4)
    max_wall_s: int = Field(default=10800, ge=30, le=14400)
    bootstrap_count: int = Field(default=4000, ge=200, le=10000)
    validation_matching_tolerance: float = Field(default=0.05, gt=0, le=0.1)
    test_matching_tolerance: float = Field(default=0.10, gt=0, le=0.15)


def hashes() -> dict[str, str]:
    return {name: file_hash(project_root() / name) for name in METHOD_FILES}


def worlds(plan: FollowupConfig, validation: bool = False) -> list[dict[str, Any]]:
    records = []
    for fi, family in enumerate(FAMILIES if validation else (*FAMILIES, *NEW_FAMILIES)):
        new = family in NEW_FAMILIES
        n = 2 if validation else (plan.new_per_family if new else plan.nominal_per_family)
        for i in range(n):
            root = (1000000 if validation else 1100000) + fi * 10000 + i
            # Proxy holds ordinary public instrument settings, not the realised process.
            cfg = scenario_config("ou" if new else family, root, PolicyConfig())
            cfg = cfg.model_copy(update={"stage": "development" if validation else "locked"})
            records.append(
                {
                    "slice": "new_family" if new else "nominal",
                    "family": family,
                    "config": cfg.model_dump(mode="json"),
                    "process": NewWorldConfig.model_validate({"family": family}).model_dump(
                        mode="json"
                    )
                    if new
                    else None,
                }
            )
    return records


def policy_set(parent: Path, horizon: int, candidates: bool = False) -> dict[str, dict[str, Any]]:
    locked, _, _ = verify_parent(parent)
    p = locked["policies"]
    result = {
        "B1_belief": p["B1_belief"],
        "B3_mean_matched": {**p["B3_mean"], "scenario_count": 2},
        "B3_risk_matched": {**p["B3_risk"], "scenario_count": 2},
        "B3_risk_original": p["B3_risk"],
    }
    for h in HORIZONS if candidates else (horizon,):
        result[f"B2_h{h}" if candidates else "B2_matched"] = {
            **p["B2_mean"],
            "planning_horizon_s": h,
        }
    return result


_deadline = math.inf
_stop: Any = None


def initialise(deadline: float, stop: Any) -> None:
    global _deadline, _stop
    _deadline, _stop = deadline, stop


def episode(scenario: dict[str, Any], policy: dict[str, Any], label: str) -> dict[str, Any]:
    cfg = ExperimentConfig.model_validate({**scenario["config"], "policy": policy})
    world = make_world(cfg, scenario["process"])
    sim = Simulator(cfg, world=world)
    agent = MeteredPolicy(cfg, _deadline, lambda: _stop is not None and _stop.is_set())
    agent.reset()
    started = time.perf_counter()
    latencies, errors, coverage = [], [], []
    switches, previous = 0, None
    while sim.state.status == "running":
        agent.check_budget()
        obs = sim.observation()
        before = time.perf_counter()
        action = agent.act(obs)
        latencies.append(time.perf_counter() - before)
        if previous is not None and abs(wrap(action.target_heading_rad - previous)) > math.radians(
            1
        ):
            switches += 1
        previous = action.target_heading_rad
        # Truth is consulted only after act, for offline diagnostics, never fed back.
        truth = world.at(obs.position_xy_m, obs.elapsed_s).air_m_s
        if obs.belief is not None:
            error = np.asarray(obs.belief.mean_air_m_s) - truth
            covariance = np.asarray(obs.belief.covariance_m2_s2)
            errors.append(float(error @ error))
            coverage.append(float(error @ np.linalg.solve(covariance, error)) <= 5.991464547)
        sim.request(action)
        end = sim.state.elapsed + cfg.clock.policy_interval_s
        while sim.state.status == "running" and sim.state.elapsed < end - 1e-8:
            agent.check_budget()
            sim.step(min(cfg.clock.physics_dt_s, end - sim.state.elapsed))
    s = sim.state
    return {
        "scenario_id": f"{scenario['family']}-{cfg.root_seed}",
        "root_seed": cfg.root_seed,
        "family": scenario["family"],
        "slice": scenario["slice"],
        "variant": label,
        "config_hash": cfg.content_hash(),
        "world_hash": world.digest,
        "status": s.status,
        "elapsed_s": s.elapsed,
        "cost_s": realised_cost(s.status, s.elapsed, cfg.policy.failure_cost_s),
        "distance_m": s.distance,
        "tack_count": s.tacks,
        "manoeuvre_s": s.manoeuvre_time,
        "action_switch_count": switches,
        "latencies_s": latencies,
        "wall_s": time.perf_counter() - started,
        "belief_vector_rmse_m_s": math.sqrt(float(np.mean(errors))) if errors else None,
        "belief_coverage95": float(np.mean(coverage)) if coverage else None,
        "model_steps": agent.steps,
        "model_step_calls": agent.step_calls,
        "model_step_ceilings": agent.ceilings,
        "model_rollouts": agent.rollouts,
    }


def worker(
    scenario: dict[str, Any], policies: dict[str, dict[str, Any]], output: str, offset: int
) -> None:
    directory = Path(output)
    root = scenario["config"]["root_seed"]
    labels = list(policies)
    labels = labels[offset % len(labels) :] + labels[: offset % len(labels)]
    try:
        for label in labels:
            row = episode(scenario, policies[label], label)
            with (directory / f"{root}-{label}.json").open("x") as handle:
                handle.write(json.dumps(row, indent=2) + "\n")
    except BaseException:
        (directory / f"{root}-error.txt").write_text(traceback.format_exc())
        raise


def validate_rows(
    rows: list[dict[str, Any]],
    registered: list[dict[str, Any]],
    policies: dict[str, dict[str, Any]],
) -> None:
    index = {(r["root_seed"], r["variant"]): r for r in rows}
    expected = {(s["config"]["root_seed"], p) for s in registered for p in policies}
    if len(index) != len(rows) or set(index) != expected:
        raise ValueError("missing, duplicate or extra registered episodes")
    for scenario in registered:
        root = scenario["config"]["root_seed"]
        expected_world = make_world(
            ExperimentConfig.model_validate(scenario["config"]), scenario["process"]
        ).digest
        for label, policy in policies.items():
            row = index[root, label]
            cfg = ExperimentConfig.model_validate({**scenario["config"], "policy": policy})
            if row["config_hash"] != cfg.content_hash() or row["world_hash"] != expected_world:
                raise ValueError("policy/config/world identity mismatch")
            if row["family"] != scenario["family"] or row["slice"] != scenario["slice"]:
                raise ValueError("process/cohort label mismatch")
            if row["status"] not in ("finished", "deadline_exceeded", "boundary_violation"):
                raise ValueError("implementation fault is not a sailing outcome")
            if not math.isfinite(row["elapsed_s"]) or not math.isclose(
                row["cost_s"],
                realised_cost(row["status"], row["elapsed_s"], cfg.policy.failure_cost_s),
                abs_tol=1e-8,
            ):
                raise ValueError("invalid failure-aware cost")
            arrays = [
                row[k]
                for k in (
                    "latencies_s",
                    "model_steps",
                    "model_step_calls",
                    "model_step_ceilings",
                    "model_rollouts",
                )
            ]
            if not arrays[0] or len({len(a) for a in arrays}) != 1:
                raise ValueError("missing or unequal decision counters")
            if any(not math.isfinite(t) or t < 0 for t in arrays[0]):
                raise ValueError("invalid decision latency")
            if any(
                not (0 <= s <= cap and calls >= s)
                for s, calls, cap in zip(*arrays[1:4], strict=True)
            ):
                raise ValueError("work counters exceed allocation")


def execute(
    registered: list[dict[str, Any]],
    policies: dict[str, dict[str, Any]],
    output: Path,
    plan: FollowupConfig,
) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=False)
    context = multiprocessing.get_context("spawn")
    stop = context.Event()
    start = time.monotonic()
    deadline = start + plan.max_wall_s
    event(output, "execution_started", worlds=len(registered), policies=list(policies))
    status = "failed"
    try:
        with ProcessPoolExecutor(
            max_workers=plan.workers,
            mp_context=context,
            initializer=initialise,
            initargs=(deadline, stop),
        ) as pool:
            pending = iter(enumerate(registered))
            futures = set()
            for _ in range(min(plan.workers, len(registered))):
                i, row = next(pending)
                futures.add(pool.submit(worker, row, policies, str(output), i))
            completed = 0
            try:
                while futures:
                    if time.monotonic() > deadline:
                        raise TimeoutError("registered cooperative wall cap exhausted")
                    done, futures = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                    for future in done:
                        future.result()
                        completed += 1
                        if completed % 32 == 0 or completed == len(registered):
                            print(
                                f"{completed}/{len(registered)} paired worlds complete", flush=True
                            )
                        item = next(pending, None)
                        if item is not None:
                            i, row = item
                            futures.add(pool.submit(worker, row, policies, str(output), i))
            except BaseException:
                stop.set()
                for future in futures:
                    future.cancel()
                raise
        rows = [json.loads(p.read_text()) for p in sorted(output.glob("[0-9]*-*.json"))]
        validate_rows(rows, registered, policies)
        status = "succeeded"
        return rows
    except BaseException:
        (output / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        rows = []
        for p in sorted(output.glob("[0-9]*-*.json")):
            try:
                rows.append(json.loads(p.read_text()))
            except (ValueError, OSError):
                pass  # Malformed raw files survive; failed manifest, never successful analysis.
        pq.write_table(pa.Table.from_pylist(rows), output / "episodes.parquet")
        event(output, "execution_completed", status=status, retained_episodes=len(rows))
        write_json(
            output / "manifest.json",
            {
                "status": status,
                "wall_s": time.monotonic() - start,
                "retained_episodes": len(rows),
                "expected_episodes": len(registered) * len(policies),
                "artifacts": {
                    p.name: file_hash(p)
                    for p in output.iterdir()
                    if p.is_file() and p.name != "manifest.json"
                },
            },
        )


def work_summary(rows: list[dict[str, Any]], label: str) -> dict[str, float]:
    sample = [r for r in rows if r["variant"] == label]
    decisions = sum(len(r["model_steps"]) for r in sample)
    return {
        "steps": float(sum(sum(r["model_steps"]) for r in sample)),
        "steps_per_decision": sum(sum(r["model_steps"]) for r in sample) / decisions,
        "latency_mean_s": float(np.mean([t for r in sample for t in r["latencies_s"]])),
        "mean_cost_s": float(np.mean([r["cost_s"] for r in sample])),
    }


def validate(parent: Path, output: Path, plan: FollowupConfig) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    registered = worlds(plan, True)
    policies = policy_set(parent, 120, True)
    write_json(output / "plan.json", plan.model_dump(mode="json"))
    write_json(output / "scenarios.json", registered)
    write_json(output / "methods.json", hashes())
    rows = execute(registered, policies, output / "episodes", plan)
    target = work_summary(rows, "B3_risk_matched")
    candidates = {}
    feasible = []
    for h in HORIZONS:
        v = work_summary(rows, f"B2_h{h}")
        ratio = target["steps_per_decision"] / v["steps_per_decision"]
        v["risk_over_b2_steps_per_decision"] = ratio
        v["within_validation_tolerance"] = abs(ratio - 1) <= plan.validation_matching_tolerance
        candidates[str(h)] = v
        if v["within_validation_tolerance"]:
            feasible.append(h)
    if not feasible:
        (output / "selection-error.txt").write_text(
            "No work-matched candidate; preserve validation and do not access final worlds.\n"
        )
        raise ValueError("no matched validation candidate")
    selected = min(feasible, key=lambda h: (candidates[str(h)]["mean_cost_s"], h))
    result = {
        "selected_horizon_s": selected,
        "matching_unit": "outer physics integration requests per tactical decision",
        "validation_worlds": len(registered),
        "target": target,
        "candidates": candidates,
        "method_hashes": hashes(),
        "plan": plan.model_dump(mode="json"),
        "reference_episode_wall_sum_s": sum(
            r["wall_s"]
            for r in rows
            if r["variant"]
            in (
                "B1_belief",
                "B3_mean_matched",
                "B3_risk_matched",
                "B3_risk_original",
                f"B2_h{selected}",
            )
        ),
    }
    write_json(output / "selection.json", result)
    write_json(
        output / "manifest.json",
        {
            "status": "succeeded",
            "artifacts": {
                p.name: file_hash(p)
                for p in output.iterdir()
                if p.is_file() and p.name != "manifest.json"
            },
        },
    )
    return result


def prepare(parent: Path, validation: Path, output: Path, plan: FollowupConfig) -> str:
    verify_parent(parent)
    verify_manifest(validation)
    verify_manifest(validation / "episodes")
    selected = json.loads((validation / "selection.json").read_text())
    if selected["method_hashes"] != hashes() or selected["plan"] != plan.model_dump(mode="json"):
        raise ValueError("changed validation methods/config")
    registered = worlds(plan)
    previous: set[int] = set()
    for path in (
        Path("artifacts/studies/m5/scenarios.json"),
        parent / "scenarios.json",
        validation / "scenarios.json",
    ):
        data = json.loads(path.read_text())
        previous.update(s.get("config", s)["root_seed"] for s in data)
    if previous & {s["config"]["root_seed"] for s in registered}:
        raise ValueError("overlapping roots")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan.model_dump(mode="json"))
    write_json(output / "scenarios.json", registered)
    write_json(
        output / "source-snapshot.json", {p: (project_root() / p).read_text() for p in METHOD_FILES}
    )
    lock = {
        "version": "matched-new-family-lock-v1",
        "parent_path": str(parent.resolve()),
        "parent_lock_sha256": file_hash(parent / "lock.json"),
        "validation_path": str(validation.resolve()),
        "validation_manifest_sha256": file_hash(validation / "manifest.json"),
        "methods": hashes(),
        "numerical_source_sha256": numerical_source_hash(),
        "dependencies": dependency_versions(),
        "policies": policy_set(parent, int(selected["selected_horizon_s"])),
        "files": {
            p: file_hash(output / p)
            for p in ("plan.json", "scenarios.json", "source-snapshot.json")
        },
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_tree": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "analysis_seed": ANALYSIS_SEED,
        "primary_contrasts": (
            "matched B3 risk minus matched B2, B1 belief, matched B3 mean; nominal "
            "and pooled new-family endpoints"
        ),
        "matching": (
            "test ratio of cohort mean actual outer integration requests per decision "
            "within 10%; wall latency and whole-episode work separately reported"
        ),
        "uncertainty_decision": (
            "retain original risk/B1 and risk/mean-B3 unresolved conclusions; no "
            "optional sample expansion or equivalence claim"
        ),
    }
    write_json(output / "lock.json", lock)
    digest = file_hash(output / "lock.json")
    (output / "lock.sha256").write_text(digest + "\n")
    event(output, "registered_before_test_access", lock_sha256=digest)
    return digest


def verify(output: Path) -> tuple[dict[str, Any], FollowupConfig, list[dict[str, Any]]]:
    lock = json.loads((output / "lock.json").read_text())
    if file_hash(output / "lock.json") != (output / "lock.sha256").read_text().strip():
        raise ValueError("lock changed")
    if (
        lock["methods"] != hashes()
        or lock["numerical_source_sha256"] != numerical_source_hash()
        or lock["dependencies"] != dependency_versions()
    ):
        raise ValueError("methods/environment changed")
    for p, digest in lock["files"].items():
        if file_hash(output / p) != digest:
            raise ValueError("registered inputs changed")
    verify_parent(Path(lock["parent_path"]))
    if (
        file_hash(Path(lock["parent_path"]) / "lock.json") != lock["parent_lock_sha256"]
        or file_hash(Path(lock["validation_path"]) / "manifest.json")
        != lock["validation_manifest_sha256"]
    ):
        raise ValueError("parent/validation changed")
    plan = FollowupConfig.model_validate_json((output / "plan.json").read_text())
    registered = json.loads((output / "scenarios.json").read_text())
    if registered != worlds(plan):
        raise ValueError("scenario generator mismatch")
    return lock, plan, registered


def analyse(rows: list[dict[str, Any]], plan: FollowupConfig) -> dict[str, Any]:
    roots = sorted({r["root_seed"] for r in rows})
    index = {(r["root_seed"], r["variant"]): r for r in rows}
    if len(index) != len(rows) or set(index) != {(r, v) for r in roots for v in VARIANTS}:
        raise ValueError("unpaired/incomplete analysis")
    family = [index[(r, VARIANTS[0])]["family"] for r in roots]
    rng = np.random.default_rng(ANALYSIS_SEED)
    draw = np.concatenate(
        [
            rng.choice(
                [i for i, f in enumerate(family) if f == name],
                size=(plan.bootstrap_count, family.count(name)),
            )
            for name in sorted(set(family))
        ],
        axis=1,
    )
    costs = {v: np.array([index[(r, v)]["cost_s"] for r in roots]) for v in VARIANTS}
    tails = {v: np.array([upper_cvar(x[d]) for d in draw]) for v, x in costs.items()}
    contrasts: dict[str, Any] = {}
    for candidate in ("B3_risk_matched", "B3_risk_original"):
        contrasts[candidate] = {}
        for ref in (v for v in VARIANTS if v != candidate):
            delta = costs[candidate] - costs[ref]
            contrasts[candidate][ref] = {
                "mean_difference_s": float(delta.mean()),
                "mean_difference_95": np.quantile(
                    delta[draw].mean(axis=1), [0.025, 0.975]
                ).tolist(),
                "cvar95_difference_s": upper_cvar(costs[candidate]) - upper_cvar(costs[ref]),
                "cvar95_difference_95": np.quantile(
                    tails[candidate] - tails[ref], [0.025, 0.975]
                ).tolist(),
                "penalty_sensitivity": {
                    str(p): {
                        "mean_difference_s": float(
                            np.mean(
                                [
                                    realised_cost(
                                        index[(r, candidate)]["status"],
                                        index[(r, candidate)]["elapsed_s"],
                                        p,
                                    )
                                    - realised_cost(
                                        index[(r, ref)]["status"], index[(r, ref)]["elapsed_s"], p
                                    )
                                    for r in roots
                                ]
                            )
                        ),
                        "cvar95_difference_s": upper_cvar(
                            [
                                realised_cost(
                                    index[(r, candidate)]["status"],
                                    index[(r, candidate)]["elapsed_s"],
                                    p,
                                )
                                for r in roots
                            ]
                        )
                        - upper_cvar(
                            [
                                realised_cost(
                                    index[(r, ref)]["status"], index[(r, ref)]["elapsed_s"], p
                                )
                                for r in roots
                            ]
                        ),
                    }
                    for p in (1800, 2400, 3600)
                },
            }
    policies = {}
    for v in VARIANTS:
        sample = [index[(r, v)] for r in roots]
        ws = work_summary(sample, v)
        failures = sum(r["status"] != "finished" for r in sample)
        coverages = [r["belief_coverage95"] for r in sample if r["belief_coverage95"] is not None]
        policies[v] = {
            **ws,
            "mean_cost_s": float(costs[v].mean()),
            "cvar95_s": upper_cvar(costs[v]),
            "failure_counts": {
                s: sum(r["status"] == s for r in sample)
                for s in ("finished", "boundary_violation", "deadline_exceeded")
            },
            "zero_failure_upper95": 1 - 0.05 ** (1 / len(roots)) if not failures else None,
            "decision_latency_p95_s": float(
                np.quantile([t for r in sample for t in r["latencies_s"]], 0.95)
            ),
            "recursive_step_calls": sum(sum(r["model_step_calls"]) for r in sample),
            "mean_tacks": float(np.mean([r["tack_count"] for r in sample])),
            "mean_distance_m": float(np.mean([r["distance_m"] for r in sample])),
            "mean_elapsed_s": float(np.mean([r["elapsed_s"] for r in sample])),
            "cost_median_s": float(np.median(costs[v])),
            "cost_q95_s": float(np.quantile(costs[v], 0.95)),
            "belief_coverage95": float(np.mean(coverages)) if coverages else None,
            "episode_wall_s_sum": sum(r["wall_s"] for r in sample),
        }
    ratios = {
        v: policies[v]["steps_per_decision"] / policies["B2_matched"]["steps_per_decision"]
        for v in ("B3_mean_matched", "B3_risk_matched")
    }
    return {
        "worlds": len(roots),
        "tail_effective_worlds": 0.05 * len(roots),
        "policies": policies,
        "contrasts": contrasts,
        "matched_steps_per_decision_ratios": ratios,
        "matching_tolerance": plan.test_matching_tolerance,
        "work_matching_passed": all(
            abs(v - 1) <= plan.test_matching_tolerance for v in ratios.values()
        ),
        "bootstrap_count": plan.bootstrap_count,
        "analysis_seed": ANALYSIS_SEED,
        "bootstrap_unit": "whole paired worlds stratified by actual process family",
        "intervals": "pointwise percentile 95%; not simultaneous or equivalence inference",
    }


def run(output: Path) -> dict[str, Any]:
    lock, plan, registered = verify(output)
    event(output, "test_access_started")
    try:
        rows = execute(registered, lock["policies"], output / "evaluation", plan)
        verify(output)
        report: dict[str, Any] = {
            "stage": "locked_measured_work_new_family",
            "lock_sha256": file_hash(output / "lock.json"),
            "episodes": len(rows),
            "cohorts": {},
            "families": {},
            "uncertainty_decision": (
                "Original B3 risk versus tuned B1 and mean B3 remains unresolved. New "
                "two-sample policies are different estimands; no pooling or optional "
                "expansion."
            ),
            "limitations": [
                "Synthetic two new families, not real sailing.",
                "Work matching is a declared tolerance on cohort mean actual integration "
                "requests per decision, not exact per-state equality, equal horizon, "
                "elapsed runtime or total episode work.",
                "Two planning samples make CVaR95 a maximum-sample proxy.",
                "New-family individual tails remain smaller than pooled tails; pointwise "
                "multiple endpoints.",
            ],
        }
        for name in ("nominal", "new_family"):
            report["cohorts"][name] = analyse([r for r in rows if r["slice"] == name], plan)
        for name in (*FAMILIES, *NEW_FAMILIES):
            report["families"][name] = analyse([r for r in rows if r["family"] == name], plan)
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
                        "plan.json",
                        "scenarios.json",
                        "source-snapshot.json",
                        "report.json",
                        "access-log.jsonl",
                        "evaluation/manifest.json",
                    )
                },
            },
        )
        return report
    except BaseException:
        (output / "error.txt").write_text(traceback.format_exc())
        event(output, "failed")
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
    parser.add_argument("--config", type=Path, default=Path("configs/followup/default.json"))
    parser.add_argument("--parent", type=Path, default=Path("artifacts/studies/m5-replication"))
    parser.add_argument(
        "--validation", type=Path, default=Path("artifacts/studies/compute-validation")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/studies/compute-followup"))
    args = parser.parse_args()
    plan = FollowupConfig.model_validate_json(args.config.read_text())
    if args.command == "validate":
        result = validate(args.parent, args.validation, plan)
    elif args.command == "prepare":
        result = {"lock_sha256": prepare(args.parent, args.validation, args.output, plan)}
    elif args.command == "run":
        result = run(args.output)
    else:
        lock, _, registered = verify(args.output)
        result = {
            "lock_sha256": file_hash(args.output / "lock.json"),
            "worlds": len(registered),
            "methods": lock["methods"],
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
