"""Static synthetic investigation evidence; never publish raw tracks, weights or job APIs."""

import hashlib
import json
from pathlib import Path
from typing import Any


def publish_evaluation(run: Path, output: Path) -> dict[str, Any]:
    data = json.loads((run / "results.json").read_text())
    if data["status"] != "succeeded" or data["registration"]["split"] != "evaluation":
        raise ValueError("Only complete registered evaluation may be published")
    rows = []
    for record in data["records"]:
        if record["task"]["split"] != "evaluation":
            raise ValueError("Mixed investigation splits")
        rows.append(
            {
                "task_id": record["task"]["task_id"],
                "strategy": record["strategy"],
                "status": record["status"],
                "error": record.get("error"),
                "score": record["score"],
                "draft": record.get("draft"),
                "trace": record["trace"],
            }
        )
    result = {
        "schema_version": 1,
        "evidence_level": "synthetic_audit_fixture",
        "model_repository": data["registration"]["model_repository"],
        "model_revision": data["registration"]["model_revision"],
        "source_sha256": data["registration"]["source_sha256"],
        "protocol_version": data["registration"]["negotiated_protocol"],
        "strategies": data["strategies"],
        "tasks": rows,
        "limits": data["registration"]["limits"],
        "raw_results_sha256": hashlib.sha256((run / "results.json").read_bytes()).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as file:
        file.write(json.dumps(result, indent=2) + "\n")
    return result
