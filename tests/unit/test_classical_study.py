import pytest

from sail_lab.classical_study import paired_report, realised_cost
from sail_lab.config import ExperimentConfig, PolicyConfig
from sail_lab.dynamics.simulator import Simulator
from sail_lab.planning import candidate_headings
from sail_lab.policies import PolarHeuristic


def test_failed_finish_cost_and_paired_missing_duplicate_or_unpaired_world_rejected():
    assert realised_cost("boundary_violation", 200, 2400) == 2400
    assert realised_cost("finished", 500, 2400) == 500
    with pytest.raises(ValueError, match="investigated"):
        realised_cost("implementation_error", 500, 2400)
    rows = []
    for world in range(4):
        for variant in ("B1_belief", "B3_risk"):
            failed = world == 0
            rows.append(
                {
                    "scenario_id": f"w{world}",
                    "variant": variant,
                    "world_hash": str(world),
                    "status": "boundary_violation" if failed else "finished",
                    "elapsed_s": 200
                    if failed
                    else 500 + world - (10 if variant == "B3_risk" else 0),
                    "cost_s": 2400 if failed else 500 + world - (10 if variant == "B3_risk" else 0),
                    "latencies_s": [0.1, 0.2],
                    "tack_count": 1,
                }
            )
    report = paired_report(rows)
    risk = report["policies"]["B3_risk"]
    assert risk["worlds"] == 4 and risk["completions"] == 3
    assert risk["cvar95_s"] == pytest.approx(2400)
    assert risk["paired_mean_difference_s"] == pytest.approx(-7.5)
    with pytest.raises(ValueError, match="missing"):
        paired_report(rows[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        paired_report([*rows, rows[0]])
    mismatched = [*rows[:-1], {**rows[-1], "world_hash": "other"}]
    with pytest.raises(ValueError, match="different worlds"):
        paired_report(mismatched)


def test_boundary_guard_uses_public_bounds_and_is_a_common_feasible_action():
    cfg = ExperimentConfig(policy=PolicyConfig(wind_input="belief", boundary_guard_s=30))
    obs = Simulator(cfg).observation().model_copy(update={"position_xy_m": (580.0, 100.0)})
    guarded = PolarHeuristic(cfg.boat, cfg.policy).act(obs)
    assert guarded.target_heading_rad < 0
    assert guarded.target_heading_rad in candidate_headings(obs, cfg)


def test_study_fault_preserves_error_config_and_partial_evidence(tmp_path, monkeypatch):
    import json

    from sail_lab.classical_study import StudyConfig, run_study

    def fault(*args, **kwargs):
        raise ArithmeticError("injected study failure")

    monkeypatch.setattr("sail_lab.classical_study.simulate", fault)
    output = tmp_path / "fault"
    with pytest.raises(ArithmeticError, match="injected study failure"):
        run_study(
            output,
            StudyConfig(families=("ou",), pilot_worlds_per_family=2),
            progress=lambda _: None,
        )
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
    assert (output / "active-config.json").exists() and (output / "scenarios.json").exists()
    assert "injected study failure" in (output / "error.txt").read_text()
