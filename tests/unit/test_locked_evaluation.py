import json
from pathlib import Path

import pytest

from sail_lab.config import ExperimentConfig
from sail_lab.experiments import file_hash, project_root, source_metadata, write_json
from sail_lab.locked_evaluation import (
    EvaluationConfig,
    amend_before_access,
    assert_disjoint,
    cohorts,
    dependency_versions,
    numerical_source_hash,
    verify_lock,
)


def test_cohorts_are_independent_named_and_predeclared_subsets() -> None:
    rows = cohorts(EvaluationConfig())
    assert len(rows) == 1536
    assert len({r["config"]["root_seed"] for r in rows}) == 1536
    assert sum(r["slice"] == "nominal" for r in rows) == 1024
    assert sum(r["classical"] for r in rows) == 192
    assert {r["slice"] for r in rows} >= {
        "sensor_bias",
        "perfect_instruments",
        "simplified_dynamics",
    }
    for row in rows:
        ExperimentConfig.model_validate(row["config"])
    with pytest.raises(ValueError, match="subset"):
        cohorts(EvaluationConfig(nominal_per_family=1))


def test_disjoint_roots_include_actual_training_manifest(tmp_path: Path) -> None:
    validation = tmp_path / "validation.json"
    write_json(validation, {"scenarios": [{"root_seed": 100}]})
    trial = tmp_path / "trial"
    trial.mkdir()
    write_json(trial / "scenario-manifest.json", {"scenarios": [{"root_seed": 200}]})
    lock = {"validation_path": str(validation), "models": [{"path": str(trial)}]}
    for roots in ([100], [200], [300, 300]):
        with pytest.raises(ValueError, match="overlap"):
            assert_disjoint(lock, [{"config": {"root_seed": r}} for r in roots])
    assert_disjoint(lock, [{"config": {"root_seed": 300}}])


def test_hash_lock_rejects_manifest_and_source_changes(tmp_path: Path) -> None:
    write_json(tmp_path / "scenarios.json", [{"config": {"root_seed": 300}}])
    write_json(tmp_path / "source-snapshot.json", {})
    validation = tmp_path / "validation.json"
    write_json(validation, {"scenarios": [{"root_seed": 100}]})
    lock = {
        "source_hash": source_metadata()[2],
        "scenario_manifest_hash": file_hash(tmp_path / "scenarios.json"),
        "source_snapshot_hash": file_hash(tmp_path / "source-snapshot.json"),
        "validation_path": str(validation),
        "validation_manifest_hash": file_hash(validation),
        "models": [],
        "dependency_versions": dependency_versions(),
        "dependency_lock_hash": file_hash(project_root() / "uv.lock"),
        "numerical_source_hash": numerical_source_hash(),
    }
    write_json(tmp_path / "lock.json", lock)
    (tmp_path / "lock.sha256").write_text(file_hash(tmp_path / "lock.json"))
    verify_lock(tmp_path)
    (tmp_path / "scenarios.json").write_text("[]")
    with pytest.raises(ValueError, match="locked artifact"):
        verify_lock(tmp_path)
    (tmp_path / "lock.json").write_text(json.dumps({**lock, "source_hash": "changed"}))
    with pytest.raises(ValueError, match="lock changed"):
        verify_lock(tmp_path)


def test_documentation_can_change_but_dependencies_cannot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("README.md", "uv.lock", "pyproject.toml"):
        (tmp_path / name).write_text("original")
    monkeypatch.setattr("sail_lab.locked_evaluation.project_root", lambda: tmp_path)
    initial = numerical_source_hash()
    (tmp_path / "README.md").write_text("updated research report")
    assert numerical_source_hash() == initial
    (tmp_path / "uv.lock").write_text("new dependencies")
    assert numerical_source_hash() != initial


def test_preaccess_amendment_is_forbidden_after_evaluation_starts(tmp_path: Path) -> None:
    (tmp_path / "evaluation").mkdir()
    with pytest.raises(ValueError, match="already accessed"):
        amend_before_access(tmp_path, "test fixture")
