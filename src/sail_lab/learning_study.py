"""Development memory/randomisation comparisons and public-boundary evaluation."""

import math
import time
import traceback
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from sail_lab.classical_study import realised_cost
from sail_lab.config import ExperimentConfig, PolicyConfig
from sail_lab.contracts import Action, Observation
from sail_lab.dynamics.simulator import Simulator
from sail_lab.experiments import file_hash, source_metadata, source_snapshot, write_json
from sail_lab.geometry import wrap
from sail_lab.planning import upper_cvar
from sail_lab.policies import PolarHeuristic, Policy
from sail_lab.rl_env import FeatureMode, FeatureSpec, headings
from sail_lab.training import (
    PPOPolicy,
    TrainingConfig,
    make_scenarios,
    scenario_manifest,
    train_trial,
)
from sail_lab.world import FrozenWorld

SEEDS = (71001, 71002, 71003)
MODES: tuple[FeatureMode, ...] = ("present", "belief", "history")
STRONG_B1 = PolicyConfig(
    wind_input="belief", layline_margin_s=9, commitment_s=20, boundary_guard_s=30
)


class ProjectedHeuristic:
    def __init__(self, config: ExperimentConfig, policy: PolicyConfig):
        self.agent = PolarHeuristic(config.boat, policy)

    def reset(self) -> None:
        self.agent.reset()

    def act(self, obs: Observation) -> Action:
        desired = self.agent.act(obs).target_heading_rad
        return Action(target_heading_rad=min(headings(obs), key=lambda h: abs(wrap(h - desired))))


def evaluate_episode(
    config: ExperimentConfig, agent: Policy, variant: str, trace: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    world = FrozenWorld(config.world, config.root_seed, config.course.deadline_s)
    sim = Simulator(config, world=world)
    agent.reset()
    latencies, errors, coverages = [], [], []
    switches, previous = 0, None
    started = time.perf_counter()
    while sim.state.status == "running":
        obs = sim.observation()
        before = time.perf_counter()
        action = agent.act(obs)
        latencies.append(time.perf_counter() - before)
        if previous is not None and abs(wrap(action.target_heading_rad - previous)) > math.radians(
            1
        ):
            switches += 1
        previous = action.target_heading_rad
        # Offline diagnostics evaluated after the policy returns; never Gym info.
        truth = world.at(obs.position_xy_m, obs.elapsed_s).air_m_s
        if obs.belief is not None:
            error = np.asarray(obs.belief.mean_air_m_s) - truth
            covariance = np.asarray(obs.belief.covariance_m2_s2)
            errors.append(float(error @ error))
            coverages.append(float(error @ np.linalg.solve(covariance, error)) <= 5.991464547)
        if trace is not None:
            trace.append(
                {
                    "elapsed_s": obs.elapsed_s,
                    "observation": obs.model_dump(mode="json"),
                    "action": action.model_dump(mode="json"),
                    "diagnostic_air_m_s": truth,
                }
            )
        sim.request(action)
        end = sim.state.elapsed + config.clock.policy_interval_s
        while sim.state.status == "running" and sim.state.elapsed < end - 1e-8:
            sim.step(min(config.clock.physics_dt_s, end - sim.state.elapsed))
    s = sim.state
    return {
        "scenario_id": f"{config.world.family}-{config.root_seed}",
        "family": config.world.family,
        "root_seed": config.root_seed,
        "config_hash": config.content_hash(),
        "world_hash": world.digest,
        "variant": variant,
        "status": s.status,
        "elapsed_s": s.elapsed,
        "cost_s": realised_cost(s.status, s.elapsed, config.policy.failure_cost_s),
        "tack_count": s.tacks,
        "distance_m": s.distance,
        "manoeuvre_s": s.manoeuvre_time,
        "action_switch_count": switches,
        "latencies_s": latencies,
        "wall_s": time.perf_counter() - started,
        "belief_vector_rmse_m_s": math.sqrt(float(np.mean(errors))) if errors else None,
        "belief_coverage95": float(np.mean(coverages)) if coverages else None,
    }


def learning_report(
    rows: list[dict[str, Any]], reference: str = "B1_belief", bootstrap_count: int = 2000
) -> dict[str, Any]:
    if not rows:
        raise ValueError("no evaluation episodes")
    ids = sorted({r["scenario_id"] for r in rows})
    variants = sorted({r["variant"] for r in rows})
    index = {(r["variant"], r["scenario_id"]): r for r in rows}
    if (
        len(index) != len(rows)
        or len(index) != len(ids) * len(variants)
        or reference not in variants
    ):
        raise ValueError("incomplete/duplicate paired table or missing reference")
    for scenario in ids:
        if len({index[(v, scenario)]["world_hash"] for v in variants}) != 1:
            raise ValueError("paired worlds differ")
    for row in rows:
        if (
            not math.isfinite(row["cost_s"])
            or not math.isfinite(row["elapsed_s"])
            or abs(row["cost_s"] - realised_cost(row["status"], row["elapsed_s"], 2400)) > 1e-8
        ):
            raise ValueError("inconsistent sailing cost")
        if not row["latencies_s"] or any(not math.isfinite(t) or t < 0 for t in row["latencies_s"]):
            raise ValueError("missing or invalid measured latency")
    rng = np.random.default_rng(8852)
    world_indices = rng.integers(len(ids), size=(bootstrap_count, len(ids)))
    baseline = np.array([index[(reference, i)]["cost_s"] for i in ids])
    policies = {}
    baseline_tail_bootstrap = np.asarray([upper_cvar(s) for s in baseline[world_indices]])
    for variant in variants:
        records = [index[(variant, i)] for i in ids]
        costs = np.array([r["cost_s"] for r in records])
        differences = costs - baseline
        bootstrap = costs[world_indices]
        tail_bootstrap = np.asarray([upper_cvar(s) for s in bootstrap])
        successes = np.array([r["status"] == "finished" for r in records])
        policies[variant] = {
            "worlds": len(ids),
            "completions": sum(r["status"] == "finished" for r in records),
            "completion_probability_95": np.quantile(
                successes[world_indices].mean(axis=1), [0.025, 0.975]
            ).tolist(),
            "failure_categories": dict(
                Counter(r["status"] for r in records if r["status"] != "finished")
            ),
            "mean_cost_s": float(np.mean(costs)),
            "mean_cost_95": np.quantile(bootstrap.mean(axis=1), [0.025, 0.975]).tolist(),
            "median_cost_s": float(np.median(costs)),
            "median_cost_95": np.quantile(np.median(bootstrap, axis=1), [0.025, 0.975]).tolist(),
            "p95_cost_s": float(np.quantile(costs, 0.95)),
            "cvar95_s": upper_cvar(costs),
            "cvar95_95": np.quantile(tail_bootstrap, [0.025, 0.975]).tolist(),
            "paired_cvar95_difference_s": upper_cvar(costs) - upper_cvar(baseline),
            "paired_cvar95_difference_95": np.quantile(
                tail_bootstrap - baseline_tail_bootstrap, [0.025, 0.975]
            ).tolist(),
            "tail_effective_outcomes": len(ids) * 0.05,
            "paired_mean_difference_s": float(np.mean(differences)),
            "paired_mean_difference_95": np.quantile(
                differences[world_indices].mean(axis=1), [0.025, 0.975]
            ).tolist(),
            "decision_latency_p95_s": float(
                np.quantile([t for r in records for t in r["latencies_s"]], 0.95)
            ),
            "mean_tacks": float(np.mean([r["tack_count"] for r in records])),
            "mean_distance_m": float(np.mean([r["distance_m"] for r in records])),
            "mean_manoeuvre_s": float(np.mean([r["manoeuvre_s"] for r in records])),
            "mean_action_switches": float(np.mean([r["action_switch_count"] for r in records])),
            "belief_vector_rmse_m_s": float(
                np.mean(
                    [
                        r["belief_vector_rmse_m_s"]
                        for r in records
                        if r.get("belief_vector_rmse_m_s") is not None
                    ]
                )
            )
            if any(r.get("belief_vector_rmse_m_s") is not None for r in records)
            else None,
            "belief_coverage95": float(
                np.mean(
                    [
                        r["belief_coverage95"]
                        for r in records
                        if r.get("belief_coverage95") is not None
                    ]
                )
            )
            if any(r.get("belief_coverage95") is not None for r in records)
            else None,
            "deadline_miss_probability": float(
                np.mean([r["status"] == "deadline_exceeded" for r in records])
            ),
            "penalty_sensitivity": {
                str(p): float(
                    np.mean([r["elapsed_s"] if r["status"] == "finished" else p for r in records])
                )
                for p in (1800, 2400, 3600)
            },
        }
    groups: dict[str, Any] = {}
    for group in sorted({v.rsplit("/seed-", 1)[0] for v in variants if "/seed-" in v}):
        models = sorted(v for v in variants if v.startswith(group + "/seed-"))
        matrix = np.array([[index[(model, i)]["cost_s"] for i in ids] for model in models])
        seed_indices = rng.integers(len(models), size=(bootstrap_count, len(models)))
        samples = matrix[seed_indices[:, :, None], world_indices[:, None, :]]
        paired = samples.mean(axis=(1, 2)) - baseline[world_indices].mean(axis=1)
        groups[group] = {
            "training_seeds": len(models),
            "independent_worlds": len(ids),
            "mean_cost_s": float(matrix.mean()),
            "paired_mean_difference_s": float((matrix - baseline).mean()),
            "hierarchical_paired_mean_95": np.quantile(paired, [0.025, 0.975]).tolist(),
            "between_training_sd_mean_cost_s": float(matrix.mean(axis=1).std(ddof=1))
            if len(models) > 1
            else None,
            "model_mean_costs_s": matrix.mean(axis=1).tolist(),
            "cvar95_mixture_s": upper_cvar(matrix.flatten()),
            "hierarchical_cvar95_95": np.quantile(
                [upper_cvar(s.flatten()) for s in samples], [0.025, 0.975]
            ).tolist(),
            "tail_independent_worlds": len(ids) * 0.05,
            "conditioning": (
                "joint resampling of training seeds and paired whole worlds; "
                "fixed finite family mixture"
            ),
        }
    return {
        "stage": "development_learning",
        "reference": reference,
        "policies": policies,
        "learned_groups": groups,
        "independent_worlds": len(ids),
        "episodes": len(rows),
        "bootstrap_count": bootstrap_count,
        "limitations": [
            "synthetic, not a real sailing trial",
            "seed-model episodes are not independent worlds",
            "no final-test selection",
        ],
    }


def run_learning_study(
    output: Path, steps: int = 65536, threads: int = 1, progress: Callable[[str], None] = print
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    validation = make_scenarios("validation", 8)
    write_json(output / "validation-manifest.json", scenario_manifest(validation, "validation"))
    write_json(
        output / "plan.json",
        {
            "version": 1,
            "stage": "development_learning",
            "seeds": SEEDS,
            "steps_per_trial": steps,
            "fixed_features": ["present", "belief", "history"],
            "randomised": "validation-selected representation",
            "selection": "lowest mean unconditional validation cost, averaged over all three seeds",
            "learning_rate": 0.0003,
            "search": "one registered hyperparameter setting; no best-seed selection",
            "training_worlds_per_family": 128,
            "validation_worlds_per_family": 8,
            "torch_threads": threads,
            "reference": STRONG_B1.model_dump(mode="json"),
            "action_matching": "B1 nearest public command alongside unrestricted B1",
            "baseline_grid": {
                "margin_s": [5, 9, 15],
                "commitment_s": [20, 40],
                "guard_s": [30, 45],
            },
        },
    )
    write_json(output / "source-snapshot.json", source_snapshot())
    rows: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    selection: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        for projected in (False, True):
            label = "B1_matched" if projected else "B1_belief"
            searches: list[dict[str, Any]] = []
            candidate_rows: list[list[dict[str, Any]]] = []
            for margin in (5, 9, 15):
                for commitment in (20, 40):
                    for guard in (30, 45):
                        params = STRONG_B1.model_copy(
                            update={
                                "layline_margin_s": margin,
                                "commitment_s": commitment,
                                "boundary_guard_s": guard,
                            }
                        )
                        outcomes = []
                        for config in validation:
                            policy = (
                                ProjectedHeuristic(config, params)
                                if projected
                                else PolarHeuristic(config.boat, params)
                            )
                            outcomes.append(evaluate_episode(config, policy, label))
                        candidate_rows.append(outcomes)
                        searches.append(
                            {
                                "policy": params.model_dump(mode="json"),
                                "mean_cost_s": float(np.mean([r["cost_s"] for r in outcomes])),
                                "episodes": outcomes,
                            }
                        )
            selected_baseline = min(range(len(searches)), key=lambda i: searches[i]["mean_cost_s"])
            write_json(output / f"{label}-tuning.json", searches)
            write_json(output / f"{label}-selected.json", searches[selected_baseline]["policy"])
            rows.extend(candidate_rows[selected_baseline])
            progress(
                f"Tuned {label}: {searches[selected_baseline]['mean_cost_s']:.1f} s on validation"
            )
        for mode in MODES:
            for seed in SEEDS:
                label = f"PPO_{mode}_fixed/seed-{seed}"
                progress(f"Training {label}: {steps} decision steps")
                cfg = TrainingConfig(
                    seed=seed,
                    total_steps=steps,
                    torch_threads=threads,
                    features=FeatureSpec(mode=mode),
                )
                path = output / "trials" / label.replace("/", "-")
                result = train_trial(path, cfg, make_scenarios("train"))
                trials.append(
                    {
                        "variant": label,
                        "path": str(path.resolve()),
                        "wall_s": result["wall_s"],
                        "model_hash": file_hash(path / "model.zip"),
                    }
                )
                agent = PPOPolicy(path)
                for config in validation:
                    rows.append(evaluate_episode(config, agent, label))
                write_json(output / "partial-episodes.json", rows)
                mean_cost = np.mean([r["cost_s"] for r in rows if r["variant"] == label])
                progress(f"Validation {label}: mean {mean_cost:.1f} s")
        group_means = {
            mode: float(
                np.mean(
                    [r["cost_s"] for r in rows if r["variant"].startswith(f"PPO_{mode}_fixed/")]
                )
            )
            for mode in ("present", "belief", "history")
        }
        selected = min(MODES, key=lambda m: group_means[m])
        selection = {
            "selected_features": selected,
            "validation_mean_costs_s": group_means,
            "individual_seed_selection": False,
        }
        write_json(output / "selection.json", selection)
        for seed in SEEDS:
            label = f"PPO_{selected}_randomised/seed-{seed}"
            progress(f"Training {label}: {steps} decision steps")
            cfg = TrainingConfig(
                seed=seed,
                total_steps=steps,
                torch_threads=threads,
                features=FeatureSpec(mode=selected),
            )
            path = output / "trials" / label.replace("/", "-")
            result = train_trial(path, cfg, make_scenarios("train", randomised=True))
            trials.append(
                {
                    "variant": label,
                    "path": str(path.resolve()),
                    "wall_s": result["wall_s"],
                    "model_hash": file_hash(path / "model.zip"),
                }
            )
            agent = PPOPolicy(path)
            for config in validation:
                rows.append(evaluate_episode(config, agent, label))
            write_json(output / "partial-episodes.json", rows)
        write_json(output / "episodes.json", rows)
        pq.write_table(pa.Table.from_pylist(rows), output / "episodes.parquet")
        write_json(output / "trials.json", trials)
        report = learning_report(rows)
        report["selection"] = selection
        report["wall_s"] = time.perf_counter() - started
        write_json(output / "report.json", report)
        manifest_status = "succeeded"
    except BaseException:
        manifest_status = "failed"
        (output / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        write_json(output / "partial-episodes.json", rows)
        write_json(output / "trials.json", trials)
        revision, dirty, source_hash = source_metadata()
        write_json(
            output / "manifest.json",
            {
                "version": 1,
                "status": manifest_status,
                "code_revision": revision,
                "dirty_tree": dirty,
                "source_hash": source_hash,
                "wall_s": time.perf_counter() - started,
                "artifacts": {
                    str(p.relative_to(output)): file_hash(p)
                    for p in sorted(output.rglob("*"))
                    if p.is_file() and p != output / "manifest.json"
                },
            },
        )
    return report
