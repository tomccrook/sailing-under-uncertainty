"""Static replication contracts reject attractive but incomplete comparisons."""

from copy import deepcopy
from typing import Any

import pytest

from scripts.planner_replication import VARIANTS
from scripts.replication_publication import ReplicationPublication


def publication_fixture() -> dict[str, Any]:
    def group(name: str, n: int) -> dict[str, Any]:
        return {
            "name": name,
            "worlds": n,
            "tail_effective_worlds": 0.05 * n,
            "policies": [
                {
                    "name": v,
                    "worlds": n,
                    "finished": n - 1,
                    "deadline_exceeded": 0,
                    "boundary_violation": 1,
                    "mean_cost_s": 700,
                    "cvar95_s": 2400,
                    "decision_latency_mean_s": 0.01,
                    "decision_latency_p95_s": 0.02,
                    "model_steps": 40,
                    "model_step_calls_including_splits": 42,
                    "allocated_model_step_ceilings": 80,
                    "model_rollouts": 8,
                    "zero_failure_upper95": None,
                }
                for v in VARIANTS
            ],
            "contrasts": [
                {
                    "reference": v,
                    "mean_difference_s": -2,
                    "mean_difference_95": [-4, 1],
                    "cvar95_difference_s": 0,
                    "cvar95_difference_95": [-5, 5],
                }
                for v in VARIANTS[:-1]
            ],
        }

    return {
        "lock_sha256": "0" * 64,
        "original_lock_sha256": "1" * 64,
        "numerical_source_sha256": "2" * 64,
        "method_sha256": "3" * 64,
        "report_sha256": "4" * 64,
        "episodes_sha256": "5" * 64,
        "total_worlds": 16,
        "retained_episodes": 80,
        "selected_deterministic_horizon_s": 120,
        "evaluation_wall_s": 60,
        "workers": 4,
        "bootstrap_count": 200,
        "precision_target_met": False,
        "cohorts": [group("nominal", 8), group("sensor_delay", 4), group("delay_tack", 4)],
        "nominal_families": [group(f, 2) for f in ("oscillating", "ou", "regime", "spatial")],
        "limitations": ["Synthetic contract fixture; these numbers are not measured results."],
    }


def test_complete_fixture_and_failure_denominators() -> None:
    data = publication_fixture()
    value = ReplicationPublication.model_validate(data)
    assert value.retained_episodes == 80
    data["cohorts"][0]["policies"][0]["boundary_violation"] = 0
    with pytest.raises(ValueError, match="excludes outcomes"):
        ReplicationPublication.model_validate(data)


@pytest.mark.parametrize(
    "change",
    [
        "remove_controller",
        "remove_stress",
        "missing_episodes",
        "false_precision",
        "bad_tail",
        "bad_work",
        "reversed_interval",
        "duplicate_family",
    ],
)
def test_rejects_incomplete_or_unearned_claims(change: str) -> None:
    data = deepcopy(publication_fixture())
    nominal = data["cohorts"][0]
    if change == "remove_controller":
        nominal["policies"].pop()
    elif change == "remove_stress":
        data["cohorts"].pop()
    elif change == "missing_episodes":
        data["retained_episodes"] = 79
    elif change == "false_precision":
        data["precision_target_met"] = True
    elif change == "bad_tail":
        nominal["tail_effective_worlds"] = 51.2
    elif change == "bad_work":
        nominal["policies"][0]["model_steps"] = 90
    elif change == "reversed_interval":
        nominal["contrasts"][0]["mean_difference_95"] = [5, -5]
    else:
        data["nominal_families"][1] = deepcopy(data["nominal_families"][0])
    with pytest.raises(ValueError):
        ReplicationPublication.model_validate(data)


def test_publication_verifies_real_episode_bytes_and_retains_failures(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq

    from sail_lab.config import ExperimentConfig, PolicyConfig
    from sail_lab.experiments import file_hash, write_json
    from sail_lab.learning_study import evaluate_episode
    from scripts import replication_publication as module
    from scripts.planner_replication import MeteredPolicy, ReplicationConfig, analyse, scenarios

    root: Path = tmp_path
    plan = ReplicationConfig(nominal_per_family=1, stress_per_family=1, bootstrap_count=200)
    registered = scenarios(plan)
    policies = {v: PolicyConfig(wind_input="belief").model_dump(mode="json") for v in VARIANTS}
    rows = []
    for world in registered:
        world["config"]["course"]["deadline_s"] = 25
        for label in VARIANTS:
            cfg = ExperimentConfig.model_validate(world["config"]).model_copy(
                update={"policy": PolicyConfig.model_validate(policies[label])}
            )
            meter = MeteredPolicy(cfg)
            row = evaluate_episode(cfg, meter, label)
            row.update(
                slice=world["slice"],
                model_steps=meter.steps,
                model_step_calls=meter.step_calls,
                model_step_ceilings=meter.ceilings,
                model_rollouts=meter.rollouts,
            )
            rows.append(row)
    run = root / "run"
    (run / "evaluation").mkdir(parents=True)
    write_json(run / "lock.json", {"fixture": "explicitly short-deadline integration fixture"})
    for name, data in [
        ("scenarios.json", registered),
        ("plan.json", plan.model_dump()),
        ("source-snapshot.json", {}),
    ]:
        write_json(run / name, data)
    report = {
        "stage": "independent_locked_replication",
        "lock_sha256": file_hash(run / "lock.json"),
        "episodes": len(rows),
        "precision_target_met": False,
        "limitations": ["Contract fixture; no policy effectiveness claim."],
        "slices": {
            s: analyse([r for r in rows if r["slice"] == s], 200)
            for s in ("nominal", "sensor_delay", "delay_tack")
        },
        "nominal_families": {
            f: analyse([r for r in rows if r["slice"] == "nominal" and r["family"] == f], 200)
            for f in ("oscillating", "ou", "regime", "spatial")
        },
    }
    write_json(run / "report.json", report)
    pq.write_table(pa.Table.from_pylist(rows), run / "evaluation/episodes.parquet")
    write_json(
        run / "evaluation/manifest.json",
        {
            "status": "succeeded",
            "wall_s": 1,
            "artifacts": {"episodes.parquet": file_hash(run / "evaluation/episodes.parquet")},
        },
    )
    write_json(
        run / "manifest.json",
        {"status": "succeeded", "artifacts": {"report.json": file_hash(run / "report.json")}},
    )
    lock = {
        "policies": policies,
        "parent_lock_sha256": "1" * 64,
        "numerical_source_sha256": "2" * 64,
        "method_hashes": {"scripts/planner_replication.py": "3" * 64},
        "parent_path": "private-path",
        "validation_path": "private-path",
    }
    monkeypatch.setattr(module, "verify", lambda p: (lock, plan, registered))
    result = module.publish(run, root / "export", root / "public")
    assert result.retained_episodes == 60
    assert result.cohorts[0].policies[0].deadline_exceeded == 4
    assert file_hash(root / "public/replication/episodes.parquet") == file_hash(
        run / "evaluation/episodes.parquet"
    )
    assert (root / "export/nominal.svg").is_file() and (root / "export/nominal.pdf").is_file()
    assert "private-path" not in (root / "export/registration.json").read_text()
    report["episodes"] = 59
    write_json(run / "report.json", report)
    write_json(
        run / "manifest.json",
        {"status": "succeeded", "artifacts": {"report.json": file_hash(run / "report.json")}},
    )
    with pytest.raises(ValueError, match="identity/count"):
        module.publish(run, root / "bad-export", root / "bad-public")
    assert not (root / "bad-public").exists()
