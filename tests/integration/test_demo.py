import json

import pytest

from sail_lab.config import ExperimentConfig
from sail_lab.experiments import file_hash, record_run
from scripts.demo_contracts import Details, Probability
from scripts.export_demo import pairing_hash, publish_recording


@pytest.fixture
def recorded(tmp_path):
    config = ExperimentConfig.model_validate(
        {
            "stage": "locked",
            "course": {"deadline_s": 3},
            "world": {"wind_speed_m_s": 0},
            "policy": {"name": "scenario_mpc", "scenario_count": 2, "planning_horizon_s": 10},
        }
    )
    directory = record_run(config, tmp_path / "runs", policy_label="B3_risk")
    manifest = json.loads((directory / "manifest.json").read_text())
    replay = json.loads((directory / "replay.json").read_text())
    entry = {
        "run_directory": str(directory),
        "manifest_hash": file_hash(directory / "manifest.json"),
        "probabilities": [],
        "cost_s": 2400,
    }
    row = {
        **replay["summary"],
        "variant": "B3_risk",
        "config_hash": replay["config_hash"],
        "world_hash": manifest["scenario_manifest_hash"],
        "cost_s": 2400,
    }
    return directory, entry, row, config


def test_publication_preserves_actual_candidates_and_failure(recorded, tmp_path):
    directory, entry, row, config = recorded
    recording, pair = publish_recording(entry, row, tmp_path / "export")
    assert recording.status == "deadline_exceeded" and recording.cost_s == 2400
    assert pair == pairing_hash(config)
    details = Details.model_validate_json(
        (tmp_path / "export" / recording.run_id / "details.json").read_text()
    )
    assert details.decisions[0].planning is not None
    assert (
        details.decisions[0].model_dump(mode="json")
        == json.loads((directory / "decisions.json").read_text())[0]
    )


@pytest.mark.parametrize(
    "field,value", [("world_hash", "0" * 64), ("elapsed_s", 2), ("tack_count", 9)]
)
def test_publisher_rejects_wrong_evaluation_outcome(recorded, tmp_path, field, value):
    _, entry, row, _ = recorded
    with pytest.raises(ValueError, match="locked outcome"):
        publish_recording(entry, {**row, field: value}, tmp_path / "export")


def test_publisher_rejects_changed_recording(recorded, tmp_path):
    directory, entry, row, _ = recorded
    (directory / "decisions.json").write_text("[]")
    with pytest.raises(ValueError, match="artifact changed"):
        publish_recording(entry, row, tmp_path / "export")


@pytest.mark.parametrize("fault", ["score", "choice", "time"])
def test_decision_contract_rejects_invented_scores_or_future_observation(recorded, fault):
    directory, _, row, config = recorded
    decisions = json.loads((directory / "decisions.json").read_text())
    decision = decisions[0]
    if fault == "score":
        decision["planning"]["candidates"][0]["mean_s"] += 10
    elif fault == "choice":
        decision["action"]["target_heading_rad"] += 1
    else:
        decision["observation"]["elapsed_s"] = 1
    with pytest.raises(ValueError):
        Details(
            run_id="fixture",
            world_hash=row["world_hash"],
            variant="B3_risk",
            config=config,
            decisions=decisions,
        )


def test_pairing_hash_includes_exogenous_sensors_and_dynamics():
    config = ExperimentConfig()
    different_policy = config.model_copy(
        update={"policy": config.policy.model_copy(update={"risk_lambda": 0.9})}
    )
    assert pairing_hash(config) == pairing_hash(different_policy)
    slower_boat = config.model_copy(
        update={"boat": config.boat.model_copy(update={"minimum_tack_s": 20})}
    )
    assert pairing_hash(config) != pairing_hash(slower_boat)
    assert pairing_hash(config) != pairing_hash(config.model_copy(update={"root_seed": 42}))


def test_probability_is_a_distribution_and_matches_deterministic_choice():
    valid = dict(elapsed_s=0, headings_rad=[0] * 8, probabilities=[0.125] * 8, selected_index=0)
    Probability.model_validate(valid)
    with pytest.raises(ValueError, match="sum to one"):
        Probability.model_validate({**valid, "probabilities": [0.1] * 8})
    with pytest.raises(ValueError, match="maximum probability"):
        Probability.model_validate({**valid, "probabilities": [0.1] * 7 + [0.3]})
