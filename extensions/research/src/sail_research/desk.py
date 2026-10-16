"""Private research capability boundary shared by stdio MCP and fixed workflows."""

import json
import subprocess
from pathlib import Path
from typing import Any

from pydantic import Field

from sail_research.contracts import (
    Budget,
    Contract,
    EvidenceRef,
    Experiment,
    ExperimentRequest,
    Identifier,
    ReportDraft,
    digest,
)
from sail_research.evidence import compare, information_issues
from sail_research.storage import DeskStore


class ToolReply(Contract):
    ok: bool
    payload: dict[str, Any] = Field(default_factory=dict)
    evidence: EvidenceRef | None = None
    error: str | None = None


class ExperimentId(Contract):
    experiment_id: Identifier


class Pair(ExperimentId):
    reference: Identifier
    candidate: Identifier


class Worst(Pair):
    limit: int = Field(default=5, ge=1, le=20)


class JobId(Contract):
    job_id: str = Field(pattern=r"^[a-f0-9]{64}$")


class Empty(Contract):
    pass


INPUTS: dict[str, type[Contract]] = {
    "list_experiments": Empty,
    "get_run_manifest": ExperimentId,
    "compare_runs": Pair,
    "inspect_worst_episodes": Worst,
    "check_information_set": Pair,
    "estimate_job_cost": ExperimentRequest,
    "submit_bounded_experiment": ExperimentRequest,
    "get_job_status": JobId,
    "export_evidence_report": ReportDraft,
}


class ResearchDesk:
    def __init__(self, root: Path, project: Path, budget: Budget):
        self.store = DeskStore(root, budget)
        self.project = project.resolve()
        self.python = self.project / ".venv/bin/python"

    def bridge(
        self, operation: str, request: ExperimentRequest | None = None, job_id: str | None = None
    ) -> dict[str, Any]:
        command = [
            str(self.python),
            "-m",
            "scripts.research_bridge",
            operation,
            "--root",
            str(self.store.root / "jobs"),
        ]
        if job_id:
            command.extend(["--job", job_id])
        response = subprocess.run(
            command,
            input=request.model_dump_json() if request else None,
            text=True,
            capture_output=True,
            cwd=self.project,
            timeout=20,
            check=False,
        )
        if response.returncode:
            raise ValueError(f"Numerical bridge failed; fault retained: {response.stderr[-1500:]}")
        value: dict[str, Any] = json.loads(response.stdout)
        return value

    def execute(self, tool: str, arguments: dict[str, Any]) -> ToolReply:
        call_id = self.store.begin_call(tool, arguments)
        try:
            if tool not in INPUTS:
                raise ValueError("Unknown research tool")
            validated = INPUTS[tool].model_validate(arguments)
            data = self._execute(tool, validated.model_dump(mode="json"))
            ref = self.store.put(
                {"tool": tool, "arguments": arguments, "payload": data}, f"{tool} verified response"
            )
            result = ToolReply(ok=True, payload=data, evidence=ref)
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            result = ToolReply(ok=False, error=f"{type(error).__name__}: {error}")
        self.store.end_call(call_id, result.model_dump(mode="json"))
        return result

    def _execute(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool == "list_experiments":
            entries = []
            for path in sorted((self.store.root / "experiments").glob("*.json")):
                experiment, ref = self.store.experiment(path.stem)
                entries.append(
                    {
                        "experiment_id": experiment.experiment_id,
                        "stage": experiment.stage,
                        "evidence_level": experiment.evidence_level,
                        "controllers": [c.label for c in experiment.controllers],
                        "source": ref.model_dump(),
                    }
                )
            return {"experiments": entries}
        if tool in {
            "get_run_manifest",
            "compare_runs",
            "inspect_worst_episodes",
            "check_information_set",
        }:
            experiment, source = self.store.experiment(arguments["experiment_id"])
            if tool == "get_run_manifest":
                data = experiment.model_dump(mode="json", exclude={"outcomes"})
                data["retained_episode_count"] = len(experiment.outcomes)
            elif tool == "compare_runs":
                data = compare(
                    experiment, arguments["reference"], arguments["candidate"]
                ).model_dump(mode="json")
            elif tool == "check_information_set":
                data = {
                    "issues": information_issues(
                        experiment, arguments["reference"], arguments["candidate"]
                    ),
                    "controllers": [c.model_dump(mode="json") for c in experiment.controllers],
                    "verification_scope": (
                        "Registered observation-field/config declarations; not a proof of "
                        "arbitrary policy code purity"
                    ),
                }
            else:
                compare(experiment, arguments["reference"], arguments["candidate"])
                data = {
                    "episodes": [
                        r.model_dump(mode="json")
                        for r in sorted(experiment.outcomes, key=lambda r: r.cost_s, reverse=True)
                        if r.label in (arguments["reference"], arguments["candidate"])
                    ][: arguments["limit"]],
                    "failure_cost_s": experiment.failure_cost_s,
                }
            return {**data, "source": source.model_dump()}
        if tool in {"estimate_job_cost", "submit_bounded_experiment"}:
            request = ExperimentRequest.model_validate(arguments)
            if tool == "estimate_job_cost":
                return {
                    "local_cpu_only": True,
                    "reserved_worker_wall_s": request.max_wall_s,
                    "episode_tasks": 2 * len(request.roots),
                    "model_or_cloud_cost_usd": 0,
                    "limits": (
                        "Cooperative worker wall limit; process startup/export not a measured "
                        "upper bound"
                    ),
                    "request_sha256": digest(arguments),
                }
            key = digest(arguments)
            created, old = self.store.reserve(key, request.max_wall_s, 2 * len(request.roots))
            if not created:
                if old is None:
                    raise ValueError(
                        "Prior launch ambiguous/failed; reservation retained, inspect owner "
                        "logs before retry"
                    )
                return {"job_id": old, "idempotent": True}
            result = self.bridge("submit", request=request)
            self.store.submitted(key, result["id"])
            return {"job_id": result["id"], "status": result["status"], "idempotent": False}
        if tool == "get_job_status":
            with self.store.connect() as db:
                owned = db.execute(
                    "SELECT 1 FROM submissions WHERE job_id=?", (arguments["job_id"],)
                ).fetchone()
            if owned is None:
                raise ValueError("Job is not registered to this desk authority")
            status = self.bridge("status", job_id=arguments["job_id"])
            if status["status"] == "succeeded":
                experiment = Experiment.model_validate(
                    self.bridge("collect", job_id=arguments["job_id"])
                )
                ref = self.store.register(experiment)
                return {
                    "job": status,
                    "experiment_id": experiment.experiment_id,
                    "source": ref.model_dump(),
                }
            return {"job": status, "faults_are_not_sailing_outcomes": True}
        if tool == "export_evidence_report":
            draft = ReportDraft.model_validate(arguments)
            experiment, source = self.store.experiment(draft.experiment_id)
            analysis = compare(experiment, draft.reference, draft.candidate)
            if not set(draft.issues).issubset(analysis.issues):
                raise ValueError("Unsupported issue accusation in report")
            if draft.verdict == "qualified" and (
                not analysis.comparable or "unsupported_claim" in analysis.issues
            ):
                raise ValueError("Unsupported conclusion: invalid comparison or claim scope")
            if draft.verdict == "invalid_comparison" and analysis.comparable:
                raise ValueError(
                    "Unsupported invalid-comparison accusation; point estimates are valid"
                )
            for uri in draft.evidence_uris:
                cited = self.store.get(uri)
                linked = cited.get("payload", {}).get("source", {})
                if uri != source.uri and linked.get("uri") != source.uri:
                    raise ValueError("Evidence link does not establish this experiment")
            data = {
                "draft": draft.model_dump(mode="json"),
                "server_verified_metrics": analysis.model_dump(mode="json"),
                "source": source.model_dump(),
                "evidence_level": experiment.evidence_level,
                "notice": (
                    "Model explanation is an unverified proposal; numerical fields are computed "
                    "by tools. No confirmatory improvement follows."
                ),
            }
            report_sha = digest(data)
            path = self.store.root / "reports" / f"{report_sha}.json"
            if not path.exists():
                with path.open("x") as file:
                    file.write(json.dumps(data, indent=2) + "\n")
            return {"report_sha256": report_sha, **data}
        raise ValueError("Unknown tool")
