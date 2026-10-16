import json
from pathlib import Path

import pytest

from scripts.export_investigation import publish


def test_static_audit_publication_retains_failures_and_refuses_favourable_subset(
    tmp_path: Path,
) -> None:
    names = ("fixed_script", "local_assistant", "researcher_critic")
    summaries = {}
    records = []
    for name in names:
        good = name == "fixed_script"
        summaries[name] = {
            "tasks": 1,
            "accepted_reports": int(good),
            "exact_issue_sets": int(good),
            "exact_rate_wilson95": [0, 1],
            "false_accusations": int(not good),
            "missed_issues": 0,
            "unsupported_structured_claims": int(not good),
            "numerical_correct_reports": int(good),
            "budget_compliant_tasks": 1,
            "wall_s": 0.1,
            "model_calls": 0 if good else 2,
            "generated_tokens": 0 if good else 100,
            "prompt_tokens": 0 if good else 500,
        }
        records.append(
            {
                "task": {"split": "evaluation", "task_id": "evaluation-00"},
                "strategy": name,
                "status": "succeeded" if good else "failed",
                "error": None if good else "False claim rejected",
                "score": {
                    "expected_issues": [],
                    "predicted_issues": [] if good else ["omitted_failures"],
                    "exact_issue_set": good,
                    "accepted_report": good,
                    "all_links_correct": good,
                    "numerical_correct": good,
                },
            }
        )
    data = {
        "status": "succeeded",
        "registration": {
            "split": "evaluation",
            "model_repository": "local/test",
            "model_revision": "a" * 40,
            "source_sha256": "b" * 64,
            "negotiated_protocol": "2026-07-28",
            "limits": "Synthetic fixture",
        },
        "strategies": summaries,
        "records": records,
    }
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.json").write_text(json.dumps(data))
    public = tmp_path / "public"
    catalogue = publish(run, public)
    assert len(catalogue.tasks) == 3
    assert sum(r.status == "failed" for r in catalogue.tasks) == 2
    assert (public / "investigation-evidence.json").read_bytes() == (
        run / "results.json"
    ).read_bytes()
    data["records"] = records[:1]
    (run / "results.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="missing|denominator"):
        publish(run, tmp_path / "invalid-public")
    assert not (tmp_path / "invalid-public").exists()
