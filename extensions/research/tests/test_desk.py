import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pytest
from pydantic import ValidationError

from sail_research.contracts import Budget, ExperimentRequest, ReportDraft
from sail_research.desk import ResearchDesk
from sail_research.evidence import compare, cvar
from sail_research.fixtures import create_suite
from sail_research.models import OllamaAdapter
from sail_research.storage import DeskStore
from sail_research.workflow import fixed_draft, parse_action

PROJECT = Path(__file__).resolve().parents[3]


@pytest.fixture
def suite(tmp_path: Path) -> ResearchDesk:
    create_suite(tmp_path / "suite")
    return ResearchDesk(
        tmp_path / "suite/desk",
        PROJECT,
        Budget(max_tool_calls=1000, max_jobs=0, max_reserved_wall_s=0, max_episode_tasks=0),
    )


def test_fractional_tail_and_failure_aware_statistics(suite: ResearchDesk) -> None:
    assert cvar(np.arange(1, 26, dtype=float)) == pytest.approx((25 + 0.25 * 24) / 1.25)
    for path in (suite.store.root / "experiments").glob("*.json"):
        experiment, _ = suite.store.experiment(path.stem)
        result = compare(experiment, "reference", "candidate")
        if result.comparable:
            rows = experiment.outcomes
            expected = np.mean([r.cost_s for r in rows if r.label == "candidate"]) - np.mean(
                [r.cost_s for r in rows if r.label == "reference"]
            )
            assert result.mean_difference_s == pytest.approx(expected)
            assert result.failures["candidate"] == 1
        else:
            assert result.mean_difference_s is None


def test_registered_information_faults_and_omitted_failures(suite: ResearchDesk) -> None:
    found = set()
    for path in (suite.store.root / "experiments").glob("*.json"):
        experiment, _ = suite.store.experiment(path.stem)
        found.update(compare(experiment, "reference", "candidate").issues)
    assert found == {
        "sensor_mismatch",
        "polar_mismatch",
        "unequal_compute",
        "hidden_state_leakage",
        "omitted_failures",
        "unsupported_claim",
        "unpaired_worlds",
        "invalid_cost",
    }
    for path in (suite.store.root / "experiments").glob("*.json"):
        experiment, _ = suite.store.experiment(path.stem)
        if "omitted_failures" in compare(experiment, "reference", "candidate").issues:
            assert len(experiment.outcomes) == 31
            assert not compare(experiment, "reference", "candidate").comparable


def test_corruption_path_and_unrelated_evidence_cannot_export(suite: ResearchDesk) -> None:
    pair = {"experiment_id": "development-00", "reference": "reference", "candidate": "candidate"}
    reply = suite.execute("compare_runs", pair)
    assert reply.ok
    draft = fixed_draft(pair, reply)
    assert suite.execute("export_evidence_report", draft.model_dump(mode="json")).ok
    bad = {**draft.model_dump(mode="json"), "evidence_uris": ["evidence://" + "0" * 64]}
    assert not suite.execute("export_evidence_report", bad).ok
    other = suite.execute("compare_runs", {**pair, "experiment_id": "development-01"})
    assert other.evidence is not None
    assert not suite.execute(
        "export_evidence_report", {**bad, "evidence_uris": [other.evidence.uri]}
    ).ok
    assert not suite.execute("get_run_manifest", {"experiment_id": "../../labels"}).ok
    assert reply.evidence is not None
    (suite.store.root / "evidence" / f"{reply.evidence.sha256}.json").write_text("{}")
    assert not suite.execute("export_evidence_report", draft.model_dump(mode="json")).ok


def test_unsupported_accusations_and_metrics_rejected(suite: ResearchDesk) -> None:
    for path in (suite.store.root / "experiments").glob("*.json"):
        experiment, _ = suite.store.experiment(path.stem)
        result = compare(experiment, "reference", "candidate")
        if not result.issues:
            task = {
                "experiment_id": experiment.experiment_id,
                "reference": "reference",
                "candidate": "candidate",
            }
            reply = suite.execute("compare_runs", task)
            draft = fixed_draft(task, reply)
            assert not suite.execute(
                "export_evidence_report",
                {**draft.model_dump(mode="json"), "issues": ["hidden_state_leakage"]},
            ).ok
            with pytest.raises(ValidationError):
                ReportDraft.model_validate({**draft.model_dump(), "mean_difference_s": -999})
            break


def test_atomic_budget_persists_and_ambiguous_submission_not_retried(tmp_path: Path) -> None:
    budget = Budget(max_tool_calls=2, max_jobs=1, max_reserved_wall_s=150, max_episode_tasks=8)
    store = DeskStore(tmp_path / "desk", budget)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: store.reserve("same", 150, 8), range(4)))
    assert sum(created for created, _ in results) == 1
    with pytest.raises(ValueError, match="exhausted"):
        store.reserve("another", 150, 8)
    store.begin_call("bad", {})
    store.begin_call("good", {})
    restarted = DeskStore(tmp_path / "desk", budget)
    with pytest.raises(ValueError, match="exhausted"):
        restarted.begin_call("retry", {})
    with pytest.raises(ValueError, match="cannot change"):
        DeskStore(tmp_path / "desk", Budget(max_tool_calls=3))


@pytest.mark.parametrize(
    "changes",
    [
        {"stage": "locked_test"},
        {"roots": [310000]},
        {"variation": "arbitrary_code"},
        {"roots": [624000] * 2},
        {"max_wall_s": 999},
        {"shell": "rm"},
    ],
)
def test_final_tuning_arbitrary_config_and_excess_budget_rejected(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ExperimentRequest.model_validate({"variation": "baseline", **changes})


def test_local_api_contract_and_remote_inference_disabled() -> None:
    captured = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "local-model",
                "message": {"content": '{"kind":"tool","tool":"compare_runs","arguments":{}}'},
                "prompt_eval_count": 5,
                "eval_count": 8,
            },
        )

    adapter = OllamaAdapter(
        "local-model", client=httpx.Client(transport=httpx.MockTransport(respond))
    )
    generated = adapter.generate([{"role": "user", "content": "audit"}])
    assert parse_action(generated["raw"]).tool == "compare_runs"
    assert captured[0]["stream"] is False
    assert captured[0]["options"]["num_predict"] == 700
    assert captured[0]["format"]["additionalProperties"] is False
    with pytest.raises(ValueError, match="local Ollama"):
        OllamaAdapter("model", "https://paid.example/api/chat")


def test_malformed_model_output_not_salvaged() -> None:
    for text in (
        "```json\n{}\n```",
        'Here: {"kind":"tool","tool":"compare_runs"}',
        '{"kind":"tool","tool_name":"compare_runs"}',
    ):
        with pytest.raises(ValidationError):
            parse_action(text)
