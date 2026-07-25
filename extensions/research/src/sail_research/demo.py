"""A real local model selects a bounded development follow-up through MCP."""

import asyncio
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from sail_research.client import call, connect
from sail_research.contracts import Contract
from sail_research.models import MLXAdapter
from sail_research.workflow import fixed_workflow, model_workflow


class Choice(Contract):
    variation: Literal["delay", "tack", "delay_tack"]
    reason: str = Field(max_length=1000)
    strongest_limitation: str = Field(max_length=1000)


async def demonstrate(root: Path, project: Path, model_root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    model = MLXAdapter(model_root)
    trace: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "status": "running",
        "evidence_level": "development_simulation",
        "trace": trace,
        "model_revision": model.manifest["revision"],
    }

    def save() -> None:
        (root / "demo.json").write_text(json.dumps(result, indent=2) + "\n")

    async with connect(root / "desk", project, project / "configs/research/local.json") as client:
        result["protocol_version"] = client.protocol_version

        async def record(tool: str, args: dict[str, Any]) -> dict[str, Any]:
            reply = await call(client, tool, args)
            trace.append({"tool": tool, "arguments": args, "result": reply.model_dump(mode="json")})
            save()
            if not reply.ok:
                raise ValueError(reply.error)
            return reply.payload

        async def experiment(variation: str) -> str:
            request = {
                "variation": variation,
                "roots": [624000, 624001, 624002, 624003],
                "max_wall_s": 150,
            }
            await record("estimate_job_cost", request)
            submitted = await record("submit_bounded_experiment", request)
            job_id = submitted["job_id"]
            for _ in range(35):
                status = await record("get_job_status", {"job_id": job_id})
                if status["job"]["status"] == "succeeded":
                    return str(status["experiment_id"])
                if status["job"]["status"] in {"failed", "cancelled"}:
                    raise ValueError(
                        "Numerical job fault; no sailing efficacy aggregate fabricated"
                    )
                await asyncio.sleep(5)
            raise TimeoutError("Bounded demonstration status-poll budget exhausted")

        try:
            base = await experiment("baseline")
            pair = {"experiment_id": base, "reference": "B1_belief", "candidate": "B3_risk"}
            comparison = await record("compare_runs", pair)
            await record("check_information_set", pair)
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "Choose one registered follow-up. Return only JSON with variation, "
                        "reason and strongest_limitation. variation is delay, tack or "
                        "delay_tack. delay means 12 s sensor latency; tack means 16 s minimum "
                        "tack duration; delay_tack combines both. No final scenarios, training, "
                        "numerical fields or spending. Four-world tails cannot establish "
                        "efficacy."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Question: Does the uncertainty-aware tactic survive increased sensor "
                        "delay and tack cost? Choose the most informative allowed test and "
                        "identify the strongest reason an improvement claim could fail. "
                        "Baseline evidence:"
                    )
                    + json.dumps(comparison),
                },
            ]
            generation = model.generate(prompt)
            trace.append({"role": "followup_selection", "generation": generation})
            save()
            choice = Choice.model_validate_json(generation["raw"])
            result["model_choice"] = choice.model_dump()
            changed = await experiment(choice.variation)
            task = {
                "task_id": "local-followup",
                "experiment_id": changed,
                "reference": "B1_belief",
                "candidate": "B3_risk",
                "question": (
                    "Inspect this paired development comparison. Discuss tail precision, "
                    "unequal compute and why no efficacy claim is established."
                ),
            }
            result["model_report"] = await model_workflow(client, task, model)
            # Comprehensive model-independent control remains available even if the model fails.
            result["script_report"] = await fixed_workflow(client, task)
            result["status"] = "succeeded"
            result["limitations"] = [
                "Two local jobs / 16 episode tasks / 300 reserved worker wall seconds",
                "Exploratory four-world samples; no final-test access, parameter tuning "
                "or confirmatory result",
                "Model report faults retained separately from successful numerical jobs",
            ]
            save()
            return result
        except Exception as error:
            result.update({"status": "failed", "error": str(error), "all_faults_retained": True})
            save()
            raise
