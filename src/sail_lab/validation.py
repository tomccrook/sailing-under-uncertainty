"""M1 convergence gate with fixed decision and instrument clocks."""

import math
from pathlib import Path

from sail_lab.config import Contract, ExperimentConfig
from sail_lab.contracts import EpisodeSummary
from sail_lab.experiments import simulate, source_metadata, write_json
from sail_lab.policies import ideal_reference, ideal_trajectory, two_leg_reference


class FixtureTrial(Contract):
    margin_s: float
    cost_s: float
    summary: EpisodeSummary
    run_path: str


def validate_simulator(output: Path) -> dict[str, object]:
    runs = []
    for dt in (1.0, 0.5, 0.25):
        config = ExperimentConfig.model_validate({"clock": {"physics_dt_s": dt}})
        summary, _, _ = simulate(config)
        runs.append({"physics_dt_s": dt, **summary.model_dump(mode="json")})
    times = [float(row["elapsed_s"]) for row in runs]
    relative_difference = max(abs(t - times[-1]) for t in times) / times[-1]
    passed = (
        relative_difference < 0.01
        and all(row["status"] == "finished" for row in runs)
        and len({row["tack_count"] for row in runs}) == 1
        and all(row["tack_count"] == 1 for row in runs)
    )
    mirrored = ExperimentConfig.model_validate(
        {"policy": {"initial_side": -1}, "course": {"initial_heading_deg": -50}}
    )
    summary, _, _ = simulate(mirrored)
    symmetry_error = abs(summary.elapsed_s - times[0])
    passed = passed and symmetry_error < 1e-6
    ideal = ideal_trajectory(7.0)[-1]
    passed = passed and math.isclose(ideal[0], ideal_reference()["arrival_s"], abs_tol=1e-8)
    report: dict[str, object] = {
        "gate": "M1 numerical",
        "passed": passed,
        "evidence_level": "synthetic",
        "ideal_reference": ideal_reference(),
        "lossless_corresponding_polar_reference": two_leg_reference(
            1000, 3.2, math.radians(50), 10
        ),
        "convergence": runs,
        "relative_finish_time_difference": relative_difference,
        "tolerance": {"relative_finish_time": 0.01, "tack_count_change": 0},
        "reflection_time_error_s": symmetry_error,
        "code_revision": source_metadata()[0],
        "source_snapshot_hash": source_metadata()[2],
        "config_hashes": [
            ExperimentConfig.model_validate({"clock": {"physics_dt_s": dt}}).content_hash()
            for dt in (1.0, 0.5, 0.25)
        ],
        "limitations": ["Fixture checks only; no calibrated boat or stochastic-policy conclusion."],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    if not passed:
        raise RuntimeError(f"M1 numerical acceptance failed; inspect {output}")
    return report


def tune_fixture(output: Path) -> dict[str, object]:
    """Small explicitly fixture-only B1 search. Retains all trials, including failures."""
    from sail_lab.experiments import record_run

    trials: list[FixtureTrial] = []
    for margin in (0.0, 5.0, 9.0, 12.0, 20.0):
        config = ExperimentConfig.model_validate(
            {"name": "b1-fixture-tuning", "policy": {"layline_margin_s": margin}}
        )
        directory = record_run(config, output / "runs")
        summary = EpisodeSummary.model_validate_json((directory / "summary.json").read_text())
        cost = summary.elapsed_s if summary.status == "finished" else config.course.deadline_s + 600
        trials.append(
            FixtureTrial(margin_s=margin, cost_s=cost, summary=summary, run_path=str(directory))
        )
    selected = min(trials, key=lambda trial: trial.cost_s)
    report = {
        "stage": "fixture",
        "not_final_test": True,
        "parameter": "layline_margin_s",
        "objective": "finish time; failures score deadline + 600 s",
        "trials": [trial.model_dump(mode="json") for trial in trials],
        "selected_margin_s": selected.margin_s,
        "warning": "Fixture-only engineering selection; M3 requires broad validation tuning.",
    }
    write_json(output / "tuning.json", report)
    return report
