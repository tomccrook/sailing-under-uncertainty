"""Replication counters, split guards and complete-denominator statistics."""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import sail_lab.planning as planning
from sail_lab.classical_study import scenario_config
from sail_lab.config import ExperimentConfig, PolicyConfig
from sail_lab.dynamics.simulator import Simulator
from sail_lab.learning_study import evaluate_episode
from sail_lab.planning import RecedingHorizonPlanner
from scripts.planner_replication import (
    VARIANTS,
    MeteredPolicy,
    ReplicationConfig,
    analyse,
    bootstrap_indices,
    execute,
    method_hashes,
    scenarios,
    validate_rows,
)


def fixture(policy: str = "scenario_mpc") -> ExperimentConfig:
    data = scenario_config(
        "ou",
        790000,
        PolicyConfig(
            name=policy,
            wind_input="belief",
            planning_horizon_s=20,
            scenario_count=2,
        ),
    ).model_dump()
    data["course"]["deadline_s"] = 25
    return ExperimentConfig.model_validate(data)


@pytest.mark.parametrize("policy", ["scenario_mpc", "deterministic_mpc"])
def test_instrumentation_preserves_episode_outcome_and_restores_constructor(policy: str) -> None:
    cfg = fixture(policy)
    original = evaluate_episode(cfg, RecedingHorizonPlanner(cfg), "original")
    metered = MeteredPolicy(cfg)
    measured = evaluate_episode(cfg, metered, "metered")
    for key in (
        "elapsed_s",
        "status",
        "tack_count",
        "action_switch_count",
        "distance_m",
        "world_hash",
    ):
        assert original[key] == measured[key]
    assert planning.Simulator is Simulator
    assert 0 < sum(metered.steps) <= sum(metered.ceilings)
    assert len(metered.steps) == len(measured["latencies_s"])
    measured.update(
        slice="fixture",
        model_steps=metered.steps,
        model_step_calls=metered.step_calls,
        model_step_ceilings=metered.ceilings,
        model_rollouts=metered.rollouts,
    )
    validate_rows(
        [measured],
        [{"slice": "fixture", "config": cfg.model_dump(mode="json")}],
        {"metered": cfg.policy.model_dump(mode="json")},
    )
    broken = dict(measured, model_steps=[10**9] * len(metered.steps))
    with pytest.raises(ValueError, match="work counters"):
        validate_rows(
            [broken],
            [{"slice": "fixture", "config": cfg.model_dump(mode="json")}],
            {"metered": cfg.policy.model_dump(mode="json")},
        )


def test_failure_inside_rollout_restores_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = fixture()
    agent = MeteredPolicy(cfg)
    calls = 0

    def fault() -> bool:
        nonlocal calls
        calls += 1
        return calls > 2

    agent.stopped = fault
    with pytest.raises(TimeoutError):
        agent.act(Simulator(cfg).observation())
    assert planning.Simulator is Simulator
    assert agent.steps == []
    expired = MeteredPolicy(cfg, deadline=time.monotonic() - 1)
    with pytest.raises(TimeoutError):
        expired.act(Simulator(cfg).observation())


def test_registered_worlds_are_disjoint_and_stress_changes_are_shared() -> None:
    plan = ReplicationConfig()
    worlds = scenarios(plan)
    validation = scenarios(plan, validation=True)
    assert len(worlds) == 1280 and len(validation) == 8
    assert len({w["config"]["root_seed"] for w in worlds}) == 1280
    assert not {w["config"]["root_seed"] for w in worlds} & {
        w["config"]["root_seed"] for w in validation
    }
    assert sum(w["slice"] == "nominal" for w in worlds) == 1024
    for row in worlds:
        cfg = ExperimentConfig.model_validate(row["config"])
        assert cfg.stage == "locked" and cfg.root_seed >= 800000
        if row["slice"] == "delay_tack":
            assert cfg.sensor.latency_s == 12 and cfg.boat.minimum_tack_s == 16
    with pytest.raises(ValueError):
        ReplicationConfig(nominal_per_family=257)


def test_stratified_resampling_retains_family_sizes_and_pairs() -> None:
    rows = [{"family": "a"}, {"family": "a"}, {"family": "b"}]
    indices = bootstrap_indices(rows, 50)
    assert indices.shape == (50, 3)
    assert np.all(indices[:, :2] < 2) and np.all(indices[:, 2] == 2)
    assert np.array_equal(indices, bootstrap_indices(rows, 50))


def fabricated_rows() -> list[dict[str, Any]]:
    rows = []
    for seed in range(20):
        for label in VARIANTS:
            fail = seed == 19
            elapsed = 100 + seed + (3 if label == "B3_risk" else 0)
            rows.append(
                {
                    "root_seed": seed,
                    "family": "ou",
                    "variant": label,
                    "status": "boundary_violation" if fail else "finished",
                    "elapsed_s": float(elapsed),
                    "cost_s": 2400.0 if fail else float(elapsed),
                    "latencies_s": [0.01],
                    "model_steps": [10],
                    "model_step_calls": [12],
                    "model_step_ceilings": [20],
                    "model_rollouts": [1],
                    "tack_count": 2,
                    "wall_s": 0.2,
                    "world_hash": str(seed),
                }
            )
    return rows


def test_paired_cost_and_tail_statistics_keep_failures_and_exact_shift() -> None:
    rows = fabricated_rows()
    report = analyse(rows, 200)
    assert report["worlds"] == 20 and report["tail_effective_worlds"] == 1
    result = report["B3_risk_minus"]["B2_mean"]
    assert result["mean_difference_s"] == pytest.approx(19 * 3 / 20)
    assert result["cvar95_difference_s"] == pytest.approx(0)
    assert report["policies"]["B3_risk"]["failure_counts"]["boundary_violation"] == 1
    assert report["policies"]["B3_risk"]["zero_failure_upper95"] is None
    with pytest.raises(ValueError, match="every policy"):
        analyse(rows[:-1], 200)
    with pytest.raises(ValueError, match="every policy"):
        analyse(rows + [rows[0]], 200)


def test_spawned_worker_preserves_sailing_failures_and_complete_manifests(tmp_path: Path) -> None:
    cfg = fixture("polar_heuristic")
    registered = [{"slice": "fixture", "config": cfg.model_dump(mode="json")}]
    policies = {"B1_belief": cfg.policy.model_dump(mode="json")}
    results = execute(registered, policies, tmp_path / "run", workers=1, max_wall_s=30)
    assert results[0]["status"] == "deadline_exceeded" and results[0]["cost_s"] == 2400
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["status"] == "succeeded" and manifest["retained_episodes"] == 1
    assert "episodes.parquet" in manifest["artifacts"]
    with pytest.raises(FileExistsError):
        execute(registered, policies, tmp_path / "run", workers=1, max_wall_s=30)


def test_worker_fault_preserves_error_without_sailing_result(tmp_path: Path) -> None:
    cfg = fixture()
    registered = [{"slice": "fixture", "config": cfg.model_dump(mode="json")}]
    with pytest.raises(ValueError):
        execute(
            registered,
            {"broken": {"name": "unknown"}},
            tmp_path / "fault",
            workers=1,
            max_wall_s=30,
        )
    manifest = json.loads((tmp_path / "fault/manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["retained_episodes"] == 0
    assert "790000-error.txt" in manifest["artifacts"]
    assert len(method_hashes()) == 1


def test_lock_rejects_test_manifest_or_method_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sail_lab.experiments import file_hash, write_json
    from scripts import planner_replication as module

    parent = tmp_path / "parent"
    parent.mkdir()
    write_json(parent / "lock.json", {"fixture": True})
    frozen = {
        label: PolicyConfig(
            name="polar_heuristic"
            if label == "B1_belief"
            else ("deterministic_mpc" if label == "B2_mean" else "scenario_mpc"),
            wind_input="belief",
        ).model_dump(mode="json")
        for label in VARIANTS
        if label != "B2_budget"
    }
    monkeypatch.setattr(module, "verify_lock", lambda path: ({"policies": frozen}, []))
    validation = tmp_path / "validation"
    validation.mkdir()
    write_json(
        validation / "selection.json",
        {"method_hashes": module.method_hashes(), "selected_policy": frozen["B2_mean"]},
    )
    (validation / "episodes").mkdir()
    write_json(validation / "episodes/manifest.json", {"status": "succeeded", "artifacts": {}})
    write_json(
        validation / "manifest.json",
        {
            "status": "succeeded",
            "artifacts": {
                "selection.json": file_hash(validation / "selection.json"),
                "episodes/manifest.json": file_hash(validation / "episodes/manifest.json"),
            },
        },
    )
    output = tmp_path / "registered"
    plan = ReplicationConfig(nominal_per_family=1, stress_per_family=1)
    module.prepare(parent, validation, output, plan)
    _, _, worlds = module.verify(output)
    assert len(worlds) == 12
    (output / "scenarios.json").write_text("[]")
    with pytest.raises(ValueError, match="scenarios/config changed"):
        module.verify(output)
    (output / "lock.json").write_text("{}")
    with pytest.raises(ValueError, match="lock changed"):
        module.verify(output)
