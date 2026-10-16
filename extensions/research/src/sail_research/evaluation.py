"""Frozen labelled investigations, retained failures and independent scoring."""

import importlib.metadata
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from sail_research.client import connect
from sail_research.contracts import Comparison, Experiment, digest
from sail_research.models import MLXAdapter
from sail_research.storage import DeskStore, read_json
from sail_research.workflow import SYSTEM, fixed_workflow, model_workflow

STRATEGIES = ("fixed_script", "local_assistant", "researcher_critic")


def source_files(project: Path) -> dict[str, str]:
    import hashlib

    paths = list((project / "extensions/research/src").rglob("*.py")) + [
        project / "extensions/research/pyproject.toml",
        project / "extensions/research/uv.lock",
        project / "scripts/research_bridge.py",
    ]
    return {
        str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }


def wilson(successes: int, total: int) -> tuple[float, float] | None:
    if not total:
        return None
    z = 1.959963984540054
    center = (successes / total + z * z / (2 * total)) / (1 + z * z / total)
    half = (
        z
        * math.sqrt(
            (successes / total) * (1 - successes / total) / total + z * z / (4 * total * total)
        )
        / (1 + z * z / total)
    )
    return max(0, center - half), min(1, center + half)


def independent_mean(experiment: Experiment, reference: str, candidate: str) -> float:
    by_key = {(r.root_seed, r.label): r for r in experiment.outcomes}
    return sum(
        by_key[(seed, candidate)].cost_s - by_key[(seed, reference)].cost_s
        for seed in experiment.expected_roots
    ) / len(experiment.expected_roots)


def score_record(
    record: dict[str, Any], expected: dict[str, Any], experiment: Experiment, store: DeskStore
) -> dict[str, Any]:
    draft = record.get("draft", {})
    predicted = set(draft.get("issues", ()))
    truth = set(expected["issues"])
    task = record["task"]
    links = draft.get("evidence_uris", [])
    valid_links = 0
    source = store.experiment(experiment.experiment_id)[1]
    for uri in links:
        try:
            cited = store.get(uri)
            linked = cited.get("payload", {}).get("source", {})
            if uri == source.uri or linked.get("uri") == source.uri:
                valid_links += 1
        except (ValueError, OSError):
            pass
    unsupported = int(
        bool(draft)
        and draft.get("verdict") == "qualified"
        and (not expected["comparable"] or "unsupported_claim" in truth)
    ) + int(bool(draft) and draft.get("verdict") == "invalid_comparison" and expected["comparable"])
    accepted = record.get("report", {}).get("ok", False)
    numerical = False
    if accepted:
        analysis = Comparison.model_validate(record["report"]["payload"]["server_verified_metrics"])
        numerical = (
            analysis.mean_difference_s is None
            if not expected["comparable"]
            else math.isclose(
                analysis.mean_difference_s if analysis.mean_difference_s is not None else math.inf,
                independent_mean(experiment, task["reference"], task["candidate"]),
                abs_tol=1e-9,
            )
        )
    generations = [row["generation"] for row in record["trace"] if "generation" in row]
    return {
        "true_positives": len(predicted & truth),
        "false_positives": len(predicted - truth),
        "false_negatives": len(truth - predicted),
        "expected_issues": sorted(truth),
        "predicted_issues": sorted(predicted),
        "exact_issue_set": bool(draft) and predicted == truth,
        "unsupported_structured_claims": unsupported + len(predicted - truth),
        "valid_evidence_links": valid_links,
        "proposed_evidence_links": len(links),
        "all_links_correct": bool(links) and valid_links == len(links),
        "numerical_correct": numerical,
        "accepted_report": accepted,
        "task_binding_correct": bool(draft)
        and all(draft.get(k) == task[k] for k in ("experiment_id", "reference", "candidate")),
        "budget_compliant": len(generations)
        <= (5 if record["strategy"] == "researcher_critic" else 4),
        "model_calls": len(generations),
        "generated_tokens": sum(g.get("generation_tokens", 0) or 0 for g in generations),
        "prompt_tokens": sum(g.get("prompt_tokens", 0) or 0 for g in generations),
        "wall_s": record["wall_s"],
        "paid_service_cost_usd": 0,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    results = {}
    for strategy in STRATEGIES:
        scored = [r["score"] for r in records if r["strategy"] == strategy]
        n = len(scored)
        tp = sum(s["true_positives"] for s in scored)
        fp = sum(s["false_positives"] for s in scored)
        fn = sum(s["false_negatives"] for s in scored)
        exact = sum(s["exact_issue_set"] for s in scored)
        results[strategy] = {
            "tasks": n,
            "accepted_reports": sum(s["accepted_report"] for s in scored),
            "exact_issue_sets": exact,
            "exact_rate": exact / n if n else None,
            "exact_rate_wilson95": wilson(exact, n),
            "issue_precision": tp / (tp + fp) if tp + fp else None,
            "issue_recall": tp / (tp + fn) if tp + fn else None,
            "true_positives": tp,
            "false_accusations": fp,
            "missed_issues": fn,
            "unsupported_structured_claims": sum(
                s["unsupported_structured_claims"] for s in scored
            ),
            "valid_evidence_links": sum(s["valid_evidence_links"] for s in scored),
            "proposed_evidence_links": sum(s["proposed_evidence_links"] for s in scored),
            "numerical_correct_reports": sum(s["numerical_correct"] for s in scored),
            "budget_compliant_tasks": sum(s["budget_compliant"] for s in scored),
            "wall_s": sum(s["wall_s"] for s in scored),
            "generated_tokens": sum(s["generated_tokens"] for s in scored),
            "prompt_tokens": sum(s["prompt_tokens"] for s in scored),
            "model_calls": sum(s["model_calls"] for s in scored),
            "paid_service_cost_usd": 0,
        }
    return results


async def evaluate(
    suite: Path, output: Path, project: Path, model_root: Path, split: str = "evaluation"
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    tasks = read_json(suite / "tasks.json")
    chosen = [t for t in tasks["tasks"] if t["split"] == split]
    if not chosen or split not in {"development", "evaluation"}:
        raise ValueError("Choose a registered task split")
    model = MLXAdapter(model_root)
    sources = source_files(project)
    registration = {
        "schema_version": "investigation-evaluation-v1",
        "status": "registered_before_model_task_access",
        "task_manifest_sha256": digest(tasks),
        "tasks_sha256": tasks["tasks_sha256"],
        "sealed_labels_sha256": tasks["labels_sha256"],
        "split": split,
        "task_ids": [t["task_id"] for t in chosen],
        "model_repository": model.manifest["repository"],
        "model_revision": model.manifest["revision"],
        "prompt_sha256": digest(SYSTEM),
        "source_files": sources,
        "source_sha256": digest(sources),
        "inference": {
            "temperature": 0,
            "thinking": False,
            "max_tokens_per_call": 700,
            "max_context_tokens": 8192,
            "max_model_turns": 4,
            "max_critic_turns": 1,
            "max_generation_wall_s": 90,
            "task_wall_s": 180,
        },
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("mcp", "numpy", "pydantic", "mlx", "mlx-lm", "transformers")
        },
        "hardware": {"platform": platform.platform(), "machine": platform.machine()},
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project, text=True
        ).strip(),
        "model_load_wall_s": model.load_wall_s,
        "limits": (
            "Known author-designed fixtures, one model and deterministic decoding; intervals "
            "describe this task suite only."
        ),
    }
    (output / "registration.json").write_text(json.dumps(registration, indent=2) + "\n")
    records = []
    started = time.monotonic()
    try:
        async with connect(
            suite / "desk", project, project / "configs/research/audit.json"
        ) as client:
            registration["negotiated_protocol"] = client.protocol_version
            (output / "registration.json").write_text(json.dumps(registration, indent=2) + "\n")
            for index, task in enumerate(chosen):
                # Rotate ordering to avoid systematically favouring one model condition's warmth.
                order = STRATEGIES[index % 3 :] + STRATEGIES[: index % 3]
                for strategy in order:
                    if strategy == "fixed_script":
                        result = await fixed_workflow(client, task)
                    else:
                        result = await model_workflow(
                            client, task, model, critic=strategy == "researcher_critic"
                        )
                    record = {"task": task, "strategy": strategy, **result}
                    records.append(record)
                    (output / f"{task['task_id']}-{strategy}.json").write_text(
                        json.dumps(record, indent=2) + "\n"
                    )
                    print(
                        json.dumps(
                            {
                                "task_id": task["task_id"],
                                "strategy": strategy,
                                "status": result["status"],
                                "wall_s": result["wall_s"],
                                "error": result.get("error"),
                            }
                        ),
                        flush=True,
                    )
        if source_files(project) != sources:
            raise ValueError(
                "Investigation source changed during evaluation; preserve incomplete run"
            )
        # Label contents are opened only after all three strategies finish every task.
        labels = read_json(suite / "labels.json")
        if digest(labels) != tasks["labels_sha256"]:
            raise ValueError("Sealed task labels changed")
        from sail_research.contracts import Budget

        store = DeskStore(
            suite / "desk",
            Budget(max_tool_calls=1000, max_jobs=0, max_reserved_wall_s=0, max_episode_tasks=0),
        )
        for record in records:
            experiment, _ = store.experiment(record["task"]["experiment_id"])
            record["score"] = score_record(
                record, labels[record["task"]["task_id"]], experiment, store
            )
        summary = {
            "status": "succeeded",
            "wall_s": time.monotonic() - started,
            "records": records,
            "strategies": summarize(records),
            "registration": registration,
        }
        (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    except Exception as error:
        (output / "error.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(error),
                    "completed_task_strategies": len(records),
                    "all_records_retained": True,
                },
                indent=2,
            )
            + "\n"
        )
        raise
