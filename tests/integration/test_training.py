import json
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from stable_baselines3 import PPO

from sail_lab.config import ExperimentConfig
from sail_lab.dynamics.simulator import Simulator
from sail_lab.rl_env import SailingEnv
from sail_lab.training import PPOPolicy, TrainingConfig, make_scenarios, train_trial


def test_training_configs_and_world_splits() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(gamma=0.99)
    with pytest.raises(ValidationError):
        TrainingConfig(total_steps=1234)
    train = make_scenarios("train", 2)
    validation = make_scenarios("validation", 2)
    assert not {c.root_seed for c in train} & {c.root_seed for c in validation}
    randomised = make_scenarios("train", 2, randomised=True)
    assert [c.root_seed for c in train] == [c.root_seed for c in randomised]
    assert randomised == make_scenarios("train", 2, randomised=True)
    assert randomised[0].boat != train[0].boat


def test_cpu_training_replication_optimizer_and_artifact_verification(tmp_path: Path) -> None:
    config = TrainingConfig(total_steps=128, n_envs=1, rollout_steps=128, batch_size=64, epochs=1)
    for name in ("first", "replica"):
        result = train_trial(tmp_path / name, config, make_scenarios("train", 1))
        assert result["status"] == "succeeded" and result["actual_steps"] == 128
    first = PPO.load(tmp_path / "first/model.zip", device="cpu")
    replica = PPO.load(tmp_path / "replica/model.zip", device="cpu")
    assert first.policy.optimizer.state_dict()["state"]
    for a, b in zip(first.policy.parameters(), replica.policy.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    agent = PPOPolicy(tmp_path / "first")
    obs = Simulator(ExperimentConfig()).observation()
    action = agent.act(obs)
    agent.reset()
    assert action == agent.act(obs)
    assert np.isfinite(action.target_heading_rad)
    (tmp_path / "first/features.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        PPOPolicy(tmp_path / "first")


def test_training_fault_is_preserved_not_a_sailing_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object) -> None:
        raise RuntimeError("deliberate physics fault")

    monkeypatch.setattr(SailingEnv, "step", fail)
    with pytest.raises(RuntimeError, match="deliberate physics fault"):
        train_trial(
            tmp_path / "failed", TrainingConfig(total_steps=1024), make_scenarios("train", 1)
        )
    manifest = json.loads((tmp_path / "failed/manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "error.txt" in manifest["artifacts"]
    assert "scenario-manifest.json" in manifest["artifacts"]
