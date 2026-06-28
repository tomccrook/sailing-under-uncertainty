from pathlib import Path

import pytest
from pydantic import ValidationError

from sail_lab.config import ExperimentConfig
from sail_lab.locked_evaluation import numerical_source_hash
from scripts.bounded_jobs import Episode, JobSpec, LocalJobs
from scripts.research_bridge import Request, collect, prepare


def test_allowlisted_treatments_keep_tuned_baselines_and_shared_information() -> None:
    normal = prepare(Request(variation="baseline", roots=(624000,)))
    changed = prepare(Request(variation="delay_tack", roots=(624000,)))
    a, b = changed.episodes
    assert a.config.sensor == b.config.sensor
    assert a.config.boat == b.config.boat
    assert a.config.sensor.latency_s == 12
    assert a.config.boat.minimum_tack_s == 16
    assert a.config.policy == normal.episodes[0].config.policy
    assert b.config.policy == normal.episodes[1].config.policy
    assert a.config.policy.layline_margin_s == 5 and a.config.policy.boundary_guard_s == 45
    assert b.config.policy.risk_lambda == 0.25 and b.config.policy.scenario_count == 8
    assert changed.numerical_source_sha256 == numerical_source_hash()
    with pytest.raises(ValidationError):
        Request(variation="baseline", roots=(310000,))


def test_bridge_retains_actual_deadline_failures_and_verified_provenance(tmp_path: Path) -> None:
    normal = prepare(Request(variation="baseline", roots=(624001,), max_wall_s=30))
    tasks = []
    for task in normal.episodes:
        fields = task.config.model_dump(mode="json")
        fields["course"]["deadline_s"] = 20
        tasks.append(Episode(label=task.label, config=ExperimentConfig.model_validate(fields)))
    spec = JobSpec.model_validate({**normal.model_dump(mode="json"), "episodes": tasks})
    jobs = LocalJobs(tmp_path / "jobs")
    job = jobs.submit(spec)
    jobs.worker(job)
    collected = collect(jobs, job)
    assert len(collected["outcomes"]) == 2
    assert all(
        row["status"] == "deadline_exceeded" and row["cost_s"] == 2400
        for row in collected["outcomes"]
    )
    assert collected["stage"] == "development"
    assert len(collected["provenance"]["bridge_source_sha256"]) == 64
