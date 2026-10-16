import asyncio
from pathlib import Path
from typing import Any

from sail_research.client import connect
from sail_research.contracts import Budget
from sail_research.desk import ResearchDesk
from sail_research.evaluation import score_record, summarize, wilson
from sail_research.fixtures import create_suite
from sail_research.workflow import fixed_draft, model_workflow

PROJECT = Path(__file__).resolve().parents[3]


def test_scorer_retains_failed_denominators_false_claims_and_wrong_numbers(tmp_path: Path) -> None:
    manifest = create_suite(tmp_path / "suite")
    desk = ResearchDesk(
        tmp_path / "suite/desk",
        PROJECT,
        Budget(max_tool_calls=1000, max_jobs=0, max_reserved_wall_s=0, max_episode_tasks=0),
    )
    experiment = next(
        desk.store.experiment(t["experiment_id"])[0]
        for t in manifest["tasks"]
        if t["split"] == "development"
        and len(desk.store.experiment(t["experiment_id"])[0].outcomes) == 31
    )
    task = {
        "experiment_id": experiment.experiment_id,
        "reference": "reference",
        "candidate": "candidate",
    }
    reply = desk.execute("compare_runs", task)
    draft = fixed_draft(task, reply)
    report = desk.execute("export_evidence_report", draft.model_dump(mode="json"))
    record = {
        "task": task,
        "strategy": "local_assistant",
        "wall_s": 2,
        "trace": [],
        "draft": draft.model_dump(mode="json"),
        "report": report.model_dump(mode="json"),
    }
    expected = {"issues": ["omitted_failures"], "comparable": False}
    good = score_record(record, expected, experiment, desk.store)
    assert good["exact_issue_set"] and good["numerical_correct"] and good["all_links_correct"]
    wrong = {**draft.model_dump(mode="json"), "issues": ["polar_mismatch"], "verdict": "qualified"}
    bad = score_record(
        {**record, "draft": wrong, "report": {"ok": False}}, expected, experiment, desk.store
    )
    assert bad["false_positives"] == 1 and bad["false_negatives"] == 1
    assert bad["unsupported_structured_claims"] == 2
    missing = score_record(
        {**record, "draft": {}, "report": {"ok": False}}, expected, experiment, desk.store
    )
    results = summarize(
        [{"strategy": "local_assistant", "score": score} for score in (good, bad, missing)]
    )
    assert results["local_assistant"]["tasks"] == 3
    assert results["local_assistant"]["accepted_reports"] == 1
    assert results["local_assistant"]["exact_issue_sets"] == 1
    assert results["local_assistant"]["missed_issues"] == 2
    assert wilson(0, 12)[1] > 0  # type: ignore[index]
    assert wilson(12, 12)[0] < 1  # type: ignore[index]


def test_model_cannot_submit_jobs_or_escape_task_and_partial_output_survives(
    tmp_path: Path,
) -> None:
    create_suite(tmp_path / "suite")
    budget = Budget(max_tool_calls=1000, max_jobs=0, max_reserved_wall_s=0, max_episode_tasks=0)
    (tmp_path / "budget.json").write_text(budget.model_dump_json())

    class Fake:
        def __init__(self, raw: str, reason: str = "stop"):
            self.raw, self.reason = raw, reason

        def generate(
            self, messages: list[dict[str, str]], max_wall_s: float = 90
        ) -> dict[str, Any]:
            return {"raw": self.raw, "finish_reason": self.reason, "generation_tokens": 5}

    async def check() -> None:
        async with connect(tmp_path / "suite/desk", PROJECT, tmp_path / "budget.json") as client:
            task = {
                "experiment_id": "development-00",
                "reference": "reference",
                "candidate": "candidate",
            }
            submit = await model_workflow(
                client,
                task,
                Fake(
                    '{"kind":"tool","tool":"submit_bounded_experiment","arguments":{"variation":"baseline"}}'
                ),
            )
            assert submit["status"] == "failed" and "outside" in submit["error"]
            assert not any("tool" in row for row in submit["trace"])
            other = await model_workflow(
                client,
                task,
                Fake(
                    '{"kind":"tool","tool":"get_run_manifest","arguments":{"experiment_id":"evaluation-00"}}'
                ),
            )
            assert "outside" in other["error"]
            partial = await model_workflow(client, task, Fake('{"kind":', "length"))
            assert partial["trace"][0]["generation"]["raw"] == '{"kind":'
            assert "budget exhausted" in partial["error"]

    asyncio.run(check())
