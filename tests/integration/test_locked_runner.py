import json
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest

from sail_lab.classical_study import scenario_config
from sail_lab.config import PolicyConfig
from sail_lab.experiments import record_run, simulate, write_json
from sail_lab.learning_study import evaluate_episode
from sail_lab.locked_evaluation import EvaluationConfig, run_locked
from sail_lab.policies import PolarHeuristic


def test_compact_evaluation_equals_full_recorded_simulator_and_labels(tmp_path: Path) -> None:
    config = scenario_config(
        "regime", 300000, PolicyConfig(wind_input="belief", boundary_guard_s=30)
    )
    agent = PolarHeuristic(config.boat, config.policy)
    row = evaluate_episode(config, agent, "contract-fixture")
    summary, _, _ = simulate(config, agent)
    assert row["elapsed_s"] == summary.elapsed_s
    assert row["status"] == summary.status
    assert row["tack_count"] == summary.tack_count
    assert row["belief_coverage95"] is not None
    path = record_run(
        config, tmp_path, policy=agent, policy_label="contract-fixture", model_id="fixture-model-id"
    )
    replay = json.loads((path / "replay.json").read_text())
    assert replay["summary"]["elapsed_s"] == row["elapsed_s"]
    assert replay["summary"]["policy_id"] == "contract-fixture"
    assert replay["policy_parameters"] is None
    assert json.loads((path / "manifest.json").read_text())["training_model_ids"] == [
        "fixture-model-id"
    ]


def test_locked_worker_fault_preserves_failed_run_and_access_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FaultPool:
        def __init__(self, **kwargs: Any):
            pass

        def __enter__(self) -> "FaultPool":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def submit(self, *args: Any) -> Future[object]:
            future: Future[object] = Future()
            future.set_exception(RuntimeError("deliberate worker fault"))
            return future

    write_json(tmp_path / "lock.json", {})
    lock = {"evaluation": EvaluationConfig(workers=1).model_dump(), "models": []}
    monkeypatch.setattr("sail_lab.locked_evaluation.verify_lock", lambda output: (lock, [{}]))
    monkeypatch.setattr("sail_lab.locked_evaluation.ProcessPoolExecutor", FaultPool)
    with pytest.raises(RuntimeError, match="deliberate worker fault"):
        run_locked(tmp_path)
    manifest = json.loads((tmp_path / "evaluation/manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["retained_episodes"] == 0
    assert "error.txt" in manifest["artifacts"]
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "access-log.jsonl").read_text().splitlines()
    ]
    assert events == ["evaluation_started", "evaluation_failed"]
