import json
import os
import time
from pathlib import Path

import pytest

from sail_lab.config import ExperimentConfig
from sail_lab.policies import PolarHeuristic
from scripts.bounded_jobs import (
    BudgetedPolicy,
    Cancelled,
    Episode,
    JobSpec,
    LocalJobs,
    WallLimit,
    evaluate_tasks,
    merge_shards,
    prepare_demo,
)


def small_spec(tmp_path: Path, *, deadline: float = 1800) -> JobSpec:
    prepared = prepare_demo(tmp_path / "spec.json")
    config = prepared.episodes[0].config
    values = config.model_dump()
    values["course"]["deadline_s"] = deadline
    return prepared.model_copy(
        update={
            "episodes": (
                Episode(label="B1_belief", config=ExperimentConfig.model_validate(values)),
            )
        }
    )


@pytest.mark.parametrize("change", ["locked", "seed", "planning", "forecast", "path", "duplicate"])
def test_jobs_reject_final_data_unbounded_work_and_unsafe_paths(
    tmp_path: Path, change: str
) -> None:
    data = small_spec(tmp_path).model_dump()
    episode = data["episodes"][0]["config"]
    if change == "locked":
        episode["stage"] = "locked"
    elif change == "seed":
        episode["root_seed"] = 310000
    elif change == "planning":
        episode["policy"]["planning_horizon_s"] = 1000
    elif change == "forecast":
        episode["forecast"]["horizon_s"] = 1_000_000
    elif change == "path":
        episode["name"] = "../../private"
    else:
        data["episodes"] = data["episodes"] * 2
    with pytest.raises(ValueError):
        JobSpec.model_validate(data)


def test_partitioned_worker_matches_local_record_and_keeps_deadline_failure(tmp_path: Path) -> None:
    spec = small_spec(tmp_path, deadline=20)
    evaluate_tasks(spec, (0,), tmp_path / "whole", tmp_path / "no-cancel")
    evaluate_tasks(spec, (0,), tmp_path / "shard", tmp_path / "no-cancel")
    whole = merge_shards(spec, (tmp_path / "whole",))[0]
    shard = merge_shards(spec, (tmp_path / "shard",))[0]
    for key in (
        "label",
        "root_seed",
        "config_sha256",
        "world_sha256",
        "status",
        "elapsed_s",
        "distance_m",
        "cost_s",
        "tacks",
    ):
        assert whole[key] == shard[key]
    assert whole["status"] == "deadline_exceeded" and whole["cost_s"] == 2400
    with pytest.raises(ValueError, match="duplicate"):
        merge_shards(spec, (tmp_path / "whole", tmp_path / "shard"))
    with pytest.raises(ValueError, match="Missing"):
        merge_shards(spec, ())
    path = tmp_path / "shard/episodes.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="table changed"):
        merge_shards(spec, (tmp_path / "shard",))


def test_cancelled_episode_is_a_worker_fault_not_a_sailing_failure(tmp_path: Path) -> None:
    spec = small_spec(tmp_path)
    (tmp_path / "cancel").touch()
    with pytest.raises(RuntimeError) as caught:
        evaluate_tasks(spec, (0,), tmp_path / "attempt", tmp_path / "cancel")
    assert isinstance(caught.value.__cause__, Cancelled)
    manifest = json.loads((tmp_path / "attempt/manifest.json").read_text())
    assert manifest["status"] == "cancelled" and manifest["completed_episodes"] == 0
    run = next((tmp_path / "attempt/runs").glob("*/manifest.json"))
    assert json.loads(run.read_text())["status"] == "failed"
    assert run.with_name("error.txt").exists()
    with pytest.raises(ValueError, match="Incomplete"):
        merge_shards(spec, (tmp_path / "attempt",))


def test_sqlite_job_identity_cancellation_retry_and_stale_worker(tmp_path: Path) -> None:
    spec = small_spec(tmp_path, deadline=20)
    jobs = LocalJobs(tmp_path / "jobs")
    identifier = jobs.submit(spec)
    assert jobs.submit(spec) == identifier
    jobs.cancel(identifier)
    assert jobs.status(identifier)["status"] == "cancelled"
    jobs.retry(identifier)
    assert jobs.status(identifier)["attempt"] == 2
    jobs.worker(identifier)
    assert jobs.status(identifier)["status"] == "succeeded"
    with pytest.raises(ValueError, match="inspected"):
        jobs.retry(identifier)
    with jobs.connect() as connection:
        connection.execute(
            "UPDATE jobs SET status='running', pid=? WHERE id=?", (99_999_999, identifier)
        )
    assert jobs.status(identifier)["status"] == "failed"
    assert "partial" in jobs.status(identifier)["error"]
    with pytest.raises(ValueError, match="exhausted"):
        jobs.retry(identifier)


def test_active_registry_prevents_nested_workers_and_cancel_is_explicit(tmp_path: Path) -> None:
    spec = small_spec(tmp_path, deadline=20)
    jobs = LocalJobs(tmp_path / "jobs")
    identifier = jobs.submit(spec)
    with jobs.connect() as connection:
        connection.execute(
            "UPDATE jobs SET status='running', pid=? WHERE id=?", (os.getpid(), identifier)
        )
    with pytest.raises(ValueError, match="one active"):
        jobs.worker(identifier)
    jobs.cancel(identifier)
    assert jobs.status(identifier)["status"] == "cancel_requested"
    assert (jobs.directory(identifier) / "cancel").exists()


def test_wall_budget_and_changed_source_fail_before_policy_actions(tmp_path: Path) -> None:
    spec = small_spec(tmp_path)
    config = spec.episodes[0].config
    guarded = BudgetedPolicy(
        PolarHeuristic(config.boat, config.policy), time.monotonic() - 1, tmp_path / "cancel"
    )
    with pytest.raises(WallLimit):
        guarded.reset()
    spec = spec.model_copy(update={"worker_source_sha256": "0" * 64})
    with pytest.raises(ValueError, match="changed"):
        evaluate_tasks(spec, (0,), tmp_path / "attempt", tmp_path / "cancel")
    assert json.loads((tmp_path / "attempt/manifest.json").read_text())["status"] == "failed"
