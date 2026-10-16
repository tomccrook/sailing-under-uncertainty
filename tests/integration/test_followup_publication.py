"""Publication must reject lost failures and unsupported scientific claims."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sail_lab.locked_evaluation import verify_manifest
from scripts import followup_publication as publication
from scripts.compute_followup import VARIANTS


def cohort(name: str = "nominal", worlds: int = 4) -> dict:
    return {
        "name": name,
        "worlds": worlds,
        "tail_effective_worlds": 0.05 * worlds,
        "matching_tolerance": 0.1,
        "work_matching_passed": True,
        "policies": [
            {
                "name": label,
                "finished": worlds - 1,
                "boundary_violation": 1,
                "deadline_exceeded": 0,
                "mean_cost_s": 1000,
                "cvar95_s": 2400,
                "steps": 100,
                "steps_per_decision": 100,
                "decision_latency_p95_s": 0.02,
                "latency_mean_s": 0.01,
                "recursive_step_calls": 102,
                "zero_failure_upper95": None,
            }
            for label in VARIANTS
        ],
        "contrasts": [
            {
                "reference": label,
                "mean_difference_s": -2,
                "mean_difference_95": [-4, 1],
                "cvar95_difference_s": 0,
                "cvar95_difference_95": [-1, 1],
            }
            for label in VARIANTS
            if label != "B3_risk_matched"
        ],
    }


def empirical() -> dict:
    # Illustrative contract fixture; it is not an empirical performance result.
    return {
        "registration_sha256": "0" * 64,
        "report_sha256": "1" * 64,
        "raw_gps_rows": 100,
        "excluded_gps_rows": 4,
        "accepted_gps_rows": 90,
        "rejected_gps_rows": 6,
        "registered_train_identifiers": 3,
        "registered_heldout_identifiers": 2,
        "train_voyages": 2,
        "heldout_voyages": 2,
        "projection_heldout_pairs": 20,
        "projection_heldout_p99_relative": 0.0001,
        "projection_gate_passed": True,
        "wind_train_days": 60,
        "wind_heldout_days": 30,
        "wind_heldout_pairs": 100,
        "predictors": [
            {
                "name": name,
                "heldout_pairs": 100,
                "vector_rmse_m_s": score,
                "speed_rmse_m_s": 0.6,
                "direction_rmse_deg_at_speed_ge_0_5": 10,
                "coverage95": 0.94,
                "gaussian_nll_mean": 2,
            }
            for name, score in (("AR1", 0.8), ("persistence", 0.81), ("climatology", 7))
        ],
        "vector_rmse_difference_m_s": -0.01,
        "vector_rmse_difference95": [-0.02, 0.01],
        "attribution": {"sailing": "Illustrative fixture", "wind": "Illustrative fixture"},
        "limitations": ["Contract fixture only"],
        "boat_calibrated": False,
    }


def test_work_publication_cannot_hide_failures_or_claim_unmatched_work() -> None:
    value = cohort()
    publication.WorkCohort.model_validate(value)
    lost = copy.deepcopy(value)
    lost["policies"][0]["boundary_violation"] = 0
    with pytest.raises(ValidationError, match="excluded outcomes"):
        publication.WorkCohort.model_validate(lost)
    unmatched = copy.deepcopy(value)
    unmatched["policies"][2]["steps_per_decision"] = 120
    with pytest.raises(ValidationError, match="unsupported matching"):
        publication.WorkCohort.model_validate(unmatched)
    unmatched["work_matching_passed"] = False
    assert not publication.WorkCohort.model_validate(unmatched).work_matching_passed


def test_zero_failure_uncertainty_and_fractional_tail_are_required() -> None:
    value = cohort()
    p = value["policies"][0]
    p.update(finished=4, boundary_violation=0)
    with pytest.raises(ValidationError, match="zero-failure bound"):
        publication.WorkCohort.model_validate(value)
    p["zero_failure_upper95"] = 1 - 0.05 ** (1 / 4)
    publication.WorkCohort.model_validate(value)
    value["tail_effective_worlds"] = 1
    with pytest.raises(ValidationError, match="tail count"):
        publication.WorkCohort.model_validate(value)


def test_complete_cohort_and_family_denominators_are_required() -> None:
    value = {
        "lock_sha256": "0" * 64,
        "report_sha256": "1" * 64,
        "episodes_sha256": "2" * 64,
        "worlds": 8,
        "episodes": 40,
        "horizon_s": 130,
        "wall_s": 5,
        "cohorts": [cohort(), cohort("new_family")],
        "families": [cohort(f, 1) for f in ("oscillating", "ou", "regime", "spatial")]
        + [cohort(f, 2) for f in ("gust_front", "persistent_trend")],
        "limitations": ["Illustrative publication fixture"],
    }
    publication.ComputePublication.model_validate(value)
    value["episodes"] = 39
    with pytest.raises(ValidationError, match="incomplete cohort"):
        publication.ComputePublication.model_validate(value)
    value["episodes"] = 40
    value["families"][0] = cohort("oscillating", 2)
    with pytest.raises(ValidationError, match="family denominator"):
        publication.ComputePublication.model_validate(value)


@pytest.mark.parametrize("field", ["raw_gps_rows", "rejected_gps_rows", "excluded_gps_rows"])
def test_recorded_publication_accounts_for_every_source_row(field: str) -> None:
    value = empirical()
    publication.EmpiricalPublication.model_validate(value)
    value[field] -= 1
    with pytest.raises(ValidationError, match="missing GPS rows"):
        publication.EmpiricalPublication.model_validate(value)


def test_recorded_targets_and_scope_are_enforced() -> None:
    value = empirical()
    value["predictors"][0]["heldout_pairs"] = 99
    with pytest.raises(ValidationError, match="unequal wind targets"):
        publication.EmpiricalPublication.model_validate(value)
    value = empirical()
    value["boat_calibrated"] = True
    with pytest.raises(ValidationError):
        publication.EmpiricalPublication.model_validate(value)
    value = empirical()
    value["projection_gate_passed"] = False
    with pytest.raises(ValidationError, match="invalid projection gate"):
        publication.EmpiricalPublication.model_validate(value)
    value = empirical()
    value["vector_rmse_difference_m_s"] = -0.1
    with pytest.raises(ValidationError, match="incorrect observed comparison"):
        publication.EmpiricalPublication.model_validate(value)


def test_static_export_identity_detects_artifact_tampering(tmp_path: Path) -> None:
    output = tmp_path / "export"
    output.mkdir()
    (output / "report.json").write_text('{"fixture":true}\n')
    publication.finish(
        output, tmp_path / "public", "empirical", publication.EmpiricalPublication(**empirical())
    )
    verify_manifest(output)
    assert json.loads((tmp_path / "public/empirical.json").read_text())["boat_calibrated"] is False
    (output / "report.json").write_text('{"fixture":false}\n')
    with pytest.raises(ValueError, match="artifact changed"):
        verify_manifest(output)
