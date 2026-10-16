import json
import math
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from sail_lab.cli import app
from sail_lab.config import ExperimentConfig
from sail_lab.contracts import Observation, ReplayBundle
from sail_lab.experiments import RunManifest, file_hash, record_run, seed_hierarchy, simulate
from sail_lab.validation import validate_simulator
from sail_lab.world import FrozenWorld


def test_timestep_convergence_and_direct_finish(tmp_path: Path) -> None:
    report = validate_simulator(tmp_path / "m1.json")
    assert report["passed"]
    summary, rows, decisions = simulate(ExperimentConfig())
    assert summary.status == "finished"
    assert summary.tack_count == 1
    assert summary.manoeuvre_s == pytest.approx(12.5)
    assert not rows[-1].manoeuvre_active
    assert len(decisions) == summary.decisions
    assert math.dist(rows[-1].position_xy_m, (0, 1000)) == pytest.approx(10)


def test_config_run_parquet_replay_and_hashes_agree(tmp_path: Path) -> None:
    config = ExperimentConfig()
    directory = record_run(config, tmp_path)
    manifest = RunManifest.model_validate_json((directory / "manifest.json").read_text())
    replay = ReplayBundle.model_validate_json((directory / "replay.json").read_text())
    trajectory = pq.read_table(directory / "trajectory.parquet").to_pylist()
    episodes = pq.read_table(directory / "episodes.parquet").to_pylist()
    assert manifest.status == "succeeded"
    assert replay.config_hash == config.content_hash() == manifest.config_hash
    assert replay.samples[-1].elapsed_s == replay.summary.elapsed_s
    assert trajectory[-1]["elapsed_s"] == episodes[0]["elapsed_s"] == replay.summary.elapsed_s
    assert trajectory[-1]["tack_count"] == replay.summary.tack_count
    assert len(trajectory) == len(replay.samples)
    for name, digest in manifest.artifacts.items():
        assert file_hash(directory / name) == digest
    assert (directory / "source-snapshot.json").exists()


def test_repeated_physics_and_policy_runs_reproduce() -> None:
    a, rows_a, decisions_a = simulate(ExperimentConfig())
    b, rows_b, decisions_b = simulate(ExperimentConfig())
    assert rows_a == rows_b
    assert a.elapsed_s == b.elapsed_s
    assert [d["action"] for d in decisions_a] == [d["action"] for d in decisions_b]
    seeds = seed_hierarchy(20261003)
    assert seeds == seed_hierarchy(20261003)
    assert len({tuple(value) for value in seeds.values()}) == 7


def test_sailing_failures_are_recorded(tmp_path: Path) -> None:
    config = ExperimentConfig.model_validate(
        {"world": {"wind_speed_m_s": 0}, "course": {"deadline_s": 5}}
    )
    directory = record_run(config, tmp_path)
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["status"] == "deadline_exceeded"
    assert json.loads((directory / "manifest.json").read_text())["status"] == "succeeded"


def test_implementation_fault_preserves_failed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fault(config: ExperimentConfig, world: FrozenWorld | None = None) -> None:
        raise ArithmeticError("injected numeric fault")

    monkeypatch.setattr("sail_lab.experiments.simulate", fault)
    with pytest.raises(RuntimeError, match="Experiment failed"):
        record_run(ExperimentConfig(), tmp_path)
    directory = next(path for path in tmp_path.iterdir() if path.is_dir())
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "injected numeric fault" in (directory / "error.txt").read_text()
    assert "episodes.parquet" not in manifest["artifacts"]


def test_policy_observation_has_no_diagnostics_or_future() -> None:
    config = ExperimentConfig()
    _, samples, _ = simulate(config)
    observation = samples[0].observation
    assert isinstance(observation, Observation)
    assert not any("diagnostic" in key or "future" in key for key in observation.model_dump())
    assert observation.information_set == "perfect_local_known_current"


def test_documented_cli_group() -> None:
    result = CliRunner().invoke(app, ["validate-config", "configs/experiments/smoke.yaml"])
    assert result.exit_code == 0
    assert "Valid schema" in result.output


def test_corrupt_replay_and_future_timestamp_rejected() -> None:
    config = ExperimentConfig.model_validate(
        {"course": {"deadline_s": 3}, "world": {"wind_speed_m_s": 0}}
    )
    summary, samples, _ = simulate(config)
    bundle = ReplayBundle(
        run_id="unit",
        config_hash=config.content_hash(),
        course=config.course,
        summary=summary,
        samples=samples,
    )
    payload = bundle.model_dump(mode="json")
    payload["summary"]["elapsed_s"] += 1
    with pytest.raises(ValidationError, match="terminal time"):
        ReplayBundle.model_validate(payload)
    payload = bundle.model_dump(mode="json")
    payload["samples"][1]["observation"]["measurement_time_s"] = 100
    with pytest.raises(ValidationError, match="future"):
        ReplayBundle.model_validate(payload)
