"""Scientific integrity checks for actual-work matching and unseen processes."""

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from sail_lab.config import ExperimentConfig, PolicyConfig
from sail_lab.dynamics.simulator import Simulator
from sail_lab.learning_study import evaluate_episode
from scripts import compute_followup as study
from scripts.followup_worlds import NEW_FAMILIES, NewWorld, NewWorldConfig
from scripts.planner_replication import MeteredPolicy


def short_scenario(family: str = "ou") -> dict:
    plan = study.FollowupConfig(nominal_per_family=1, new_per_family=1)
    row = next(r for r in study.worlds(plan) if r["family"] == family)
    row["config"]["course"]["deadline_s"] = 20
    return row


@pytest.mark.parametrize("family", NEW_FAMILIES)
def test_new_world_frozen_query_independent_and_seeded(family: str) -> None:
    cfg = ExperimentConfig(root_seed=334)
    process = NewWorldConfig.model_validate({"family": family})
    a, b = NewWorld(cfg, process), NewWorld(cfg, process)
    assert a.digest == b.digest
    assert not a.direction.flags.writeable
    points = [0.0, 10.5, 1777.0, 400.0]
    forward = [a.at((12, 34), t) for t in points]
    reverse = {t: b.at((-99, 50), t) for t in reversed(points)}
    assert all(x == reverse[t] for x, t in zip(forward, points, strict=True))
    assert max(abs(a.direction)) <= process.direction_limit_deg
    assert min(a.speed) >= 0
    assert NewWorld(cfg.model_copy(update={"root_seed": 335}), process).digest != a.digest
    with pytest.raises(ValueError):
        a.at((0, 0), math.inf)


def test_shot_noise_arrivals_have_declared_rate() -> None:
    counts = [
        len(NewWorld(ExperimentConfig(root_seed=i), NewWorldConfig(family="gust_front")).events)
        for i in range(200)
    ]
    # Poisson event counts over 1800s, independent worlds; six-standard-error tolerance.
    expected = 1800 / 140
    assert abs(np.mean(counts) - expected) < 6 * math.sqrt(expected / 200)


def test_new_future_truth_cannot_change_initial_policy_action() -> None:
    row = short_scenario("gust_front")
    cfg = ExperimentConfig.model_validate(row["config"])
    process = NewWorldConfig.model_validate(row["process"])
    a, b = NewWorld(cfg, process), NewWorld(cfg, process)
    b.direction = b.direction.copy()
    b.direction[1:] = 45  # Same available past, different hidden future.
    obs_a, obs_b = Simulator(cfg, world=a).observation(), Simulator(cfg, world=b).observation()
    assert obs_a == obs_b
    planner = cfg.model_copy(
        update={"policy": PolicyConfig(name="scenario_mpc", wind_input="belief", scenario_count=2)}
    )
    assert MeteredPolicy(planner).act(obs_a) == MeteredPolicy(planner).act(obs_b)


def test_existing_episode_matches_frozen_implementation() -> None:
    row = short_scenario()
    policy = PolicyConfig(name="scenario_mpc", scenario_count=2, wind_input="belief").model_dump(
        mode="json"
    )
    cfg = ExperimentConfig.model_validate({**row["config"], "policy": policy})
    old = evaluate_episode(cfg, MeteredPolicy(cfg), "test")
    new = study.episode(row, policy, "test")
    for key in (
        "world_hash",
        "config_hash",
        "status",
        "elapsed_s",
        "cost_s",
        "tack_count",
        "distance_m",
        "belief_coverage95",
    ):
        assert new[key] == old[key]


def test_manifests_are_disjoint_balanced_and_new_family_absent_from_validation() -> None:
    plan = study.FollowupConfig()
    validation, final = study.worlds(plan, True), study.worlds(plan)
    assert len(validation) == 8 and len(final) == 2048
    val_roots = {r["config"]["root_seed"] for r in validation}
    assert not val_roots & {r["config"]["root_seed"] for r in final}
    assert not set(NEW_FAMILIES) & {r["family"] for r in validation}
    assert sum(r["slice"] == "new_family" for r in final) == 1024
    assert all(r["process"] is not None for r in final if r["family"] in NEW_FAMILIES)


def test_complete_rows_reject_omitted_failure_and_world_mismatch() -> None:
    row = short_scenario()
    policies = {"B1_belief": PolicyConfig(wind_input="belief").model_dump(mode="json")}
    actual = study.episode(row, policies["B1_belief"], "B1_belief")
    assert actual["status"] == "deadline_exceeded"
    study.validate_rows([actual], [row], policies)
    with pytest.raises(ValueError, match="missing"):
        study.validate_rows([], [row], policies)
    changed = copy.deepcopy(actual)
    changed["world_hash"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        study.validate_rows([changed], [row], policies)


def test_worker_fault_keeps_failed_manifest_and_raw_error(tmp_path: Path) -> None:
    row = short_scenario()
    row["config"]["clock"]["physics_dt_s"] = 3
    policies = {"B1_belief": PolicyConfig().model_dump(mode="json")}
    with pytest.raises(ValueError):
        study.execute([row], policies, tmp_path / "run", study.FollowupConfig(workers=1))
    import json

    record = json.loads((tmp_path / "run/manifest.json").read_text())
    assert record["status"] == "failed"
    assert record["expected_episodes"] == 1
    assert (tmp_path / f"run/{row['config']['root_seed']}-error.txt").exists()


def test_analysis_retains_failure_penalty_and_exposes_nonmatching_work() -> None:
    rows = []
    for root in range(4):
        for label in study.VARIANTS:
            steps = 10 if label == "B2_matched" else (20 if "matched" in label else 0)
            rows.append(
                {
                    "root_seed": root,
                    "family": "test",
                    "variant": label,
                    "cost_s": 2400.0,
                    "elapsed_s": 20.0,
                    "status": "deadline_exceeded",
                    "model_steps": [steps],
                    "model_step_calls": [steps],
                    "latencies_s": [0.1],
                    "tack_count": 0,
                    "distance_m": 0.0,
                    "belief_coverage95": None,
                    "wall_s": 0.1,
                }
            )
    result = study.analyse(rows, study.FollowupConfig(bootstrap_count=200))
    assert not result["work_matching_passed"]
    assert result["policies"]["B1_belief"]["failure_counts"]["deadline_exceeded"] == 4
    assert result["contrasts"]["B3_risk_matched"]["B2_matched"]["cvar95_difference_95"] == [
        0.0,
        0.0,
    ]
