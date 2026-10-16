import pytest
from pydantic import ValidationError

from sail_lab.config import ExperimentConfig


def test_unknown_truth_and_cloud_fields_rejected() -> None:
    for payload in (
        {"hidden_wind": True},
        {"cloud_submission_enabled": True},
        {"retain_failed_episodes": False},
        {"sensor": {"allow_diagnostic_truth": True}},
    ):
        with pytest.raises(ValidationError):
            ExperimentConfig.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"world": {"wind_speed_m_s": -1}},
        {"course": {"deadline_s": float("nan")}},
        {"course": {"bounds_xy_m": [600, -600, -100, 1200]}},
        {"course": {"target_xy_m": [0, 0]}},
        {"clock": {"physics_dt_s": 0.3}},
        {"boat": {"acceleration_tau_s": 0}},
    ],
)
def test_invalid_config(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(payload)


def test_hash_covers_nested_parameters() -> None:
    a = ExperimentConfig()
    b = ExperimentConfig.model_validate({"boat": {"minimum_tack_s": 12}})
    assert a.content_hash() == ExperimentConfig().content_hash()
    assert a.content_hash() != b.content_hash()
