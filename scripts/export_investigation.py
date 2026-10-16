"""Publish only complete synthetic M8 audit evidence and generated UI contracts."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.investigation_contracts import InvestigationCatalogue


def export_schemas(project: Path) -> None:
    schema: dict[str, Any] = InvestigationCatalogue.model_json_schema()
    schema["$schema"] = "http://json-schema.org/draft-07/schema#"

    def lower(value: Any) -> None:
        if isinstance(value, dict):
            if "prefixItems" in value:
                value["items"] = value.pop("prefixItems")
            for child in value.values():
                lower(child)
        elif isinstance(value, list):
            for child in value:
                lower(child)

    lower(schema)
    for path in (
        project / "schemas/investigation.schema.json",
        project / "web/investigation.schema.json",
    ):
        path.write_text(json.dumps(schema, indent=2) + "\n")


def publish(run: Path, public: Path) -> InvestigationCatalogue:
    raw = (run / "results.json").read_bytes()
    data = json.loads(raw)
    if data["status"] != "succeeded" or data["registration"]["split"] != "evaluation":
        raise ValueError("Only complete registered held-out investigations may be published")
    records = data["records"]
    if any(row["task"]["split"] != "evaluation" for row in records):
        raise ValueError("Mixed split publication")
    catalogue = InvestigationCatalogue.model_validate(
        {
            "model_repository": data["registration"]["model_repository"],
            "model_revision": data["registration"]["model_revision"],
            "source_sha256": data["registration"]["source_sha256"],
            "protocol_version": data["registration"]["negotiated_protocol"],
            "strategies": [
                {
                    "name": name,
                    **{
                        field: value[field]
                        for field in (
                            "tasks",
                            "accepted_reports",
                            "exact_issue_sets",
                            "exact_rate_wilson95",
                            "false_accusations",
                            "missed_issues",
                            "unsupported_structured_claims",
                            "numerical_correct_reports",
                            "budget_compliant_tasks",
                            "wall_s",
                            "model_calls",
                            "generated_tokens",
                            "prompt_tokens",
                        )
                    },
                }
                for name, value in data["strategies"].items()
            ],
            "tasks": [
                {
                    "task_id": r["task"]["task_id"],
                    "strategy": r["strategy"],
                    "status": r["status"],
                    "expected_issues": r["score"]["expected_issues"],
                    "predicted_issues": r["score"]["predicted_issues"],
                    "exact_issue_set": r["score"]["exact_issue_set"],
                    "accepted_report": r["score"]["accepted_report"],
                    "evidence_links_correct": r["score"]["all_links_correct"],
                    "numerical_correct": r["score"]["numerical_correct"],
                    "error": r.get("error"),
                    "explanation": r.get("draft", {}).get("explanation"),
                }
                for r in records
            ],
            "result_sha256": hashlib.sha256(raw).hexdigest(),
            "limits": data["registration"]["limits"],
        }
    )
    public.mkdir(parents=True, exist_ok=True)
    (public / "investigation-evidence.json").write_bytes(raw)
    (public / "investigations.json").write_text(catalogue.model_dump_json(indent=2) + "\n")
    return catalogue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--public", type=Path, default=Path("web/public"))
    args = parser.parse_args()
    export_schemas(Path.cwd())
    if args.run:
        publish(args.run, args.public)


if __name__ == "__main__":
    main()
