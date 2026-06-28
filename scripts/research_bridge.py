"""Allowlisted MCP bridge executed only by the frozen numerical Python environment."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from sail_lab.classical_study import scenario_config
from sail_lab.config import Contract, ExperimentConfig, PolicyConfig
from sail_lab.contracts import Observation
from sail_lab.experiments import file_hash
from sail_lab.locked_evaluation import dependency_versions, numerical_source_hash
from scripts.bounded_jobs import Episode, JobSpec, LocalJobs, merge_shards


class Request(Contract):
    variation: Literal["baseline", "delay", "tack", "delay_tack"]
    roots: tuple[int, ...] = (624000, 624001, 624002, 624003)
    max_wall_s: int = Field(default=150, ge=10, le=150)
    stage: Literal["development"] = "development"

    @model_validator(mode="after")
    def allowed(self) -> "Request":
        if not 1 <= len(self.roots) <= 4 or len(set(self.roots)) != len(self.roots):
            raise ValueError("One to four unique roots required")
        if any(not 624000 <= seed < 625000 for seed in self.roots):
            raise ValueError("Only registered M8 development roots permitted")
        return self


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def prepare(request: Request) -> JobSpec:
    policies = {
        "B1_belief": PolicyConfig(wind_input="belief", layline_margin_s=5, boundary_guard_s=45),
        "B3_risk": PolicyConfig(
            name="scenario_mpc", wind_input="belief", risk_lambda=0.25, boundary_guard_s=30
        ),
    }
    episodes = []
    for seed in request.roots:
        for label, policy in policies.items():
            data = scenario_config("ou", seed, policy).model_dump(mode="json")
            if request.variation in {"delay", "delay_tack"}:
                data["sensor"]["latency_s"] = 12
            if request.variation in {"tack", "delay_tack"}:
                data["boat"]["minimum_tack_s"] = 16
            data["name"] = f"m8-{request.variation}-{seed}-{label}"
            episodes.append(Episode(label=label, config=ExperimentConfig.model_validate(data)))
    return JobSpec(
        numerical_source_sha256=numerical_source_hash(),
        worker_source_sha256=file_hash(Path(__file__).with_name("bounded_jobs.py")),
        dependency_versions=dependency_versions(),
        episodes=tuple(episodes),
        max_wall_s=request.max_wall_s,
        max_attempts=1,
    )


def start_if_available(jobs: LocalJobs, job_id: str) -> dict[str, Any]:
    status = jobs.status(job_id)
    if status["status"] == "queued":
        with jobs.connect() as db:
            active = db.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('running','cancel_requested')"
            ).fetchone()[0]
        if not active:
            jobs.start(job_id)
    return jobs.status(job_id)


def collect(jobs: LocalJobs, job_id: str) -> dict[str, Any]:
    status = jobs.status(job_id)
    if status["status"] != "succeeded":
        raise ValueError(
            "Only complete successful jobs may become comparable evidence; faults retained"
        )
    directory = jobs.directory(job_id)
    spec = JobSpec.model_validate_json((directory / "job.json").read_text())
    spec.verify_runtime()
    attempt = directory / f"attempt-{status['attempt']}"
    rows = merge_shards(spec, (attempt,))
    controllers: dict[str, Any] = {}
    for task in spec.episodes:
        config = task.config
        info = {
            name: config.model_dump(mode="json")[name]
            for name in ("sensor", "forecast", "belief", "clock")
        }
        controllers[task.label] = {
            "label": task.label,
            "information_sha256": canonical_hash(info),
            "polar_sha256": canonical_hash(config.boat.model_dump(mode="json")),
            "compute_units": config.policy.planning_horizon_s * config.policy.scenario_count
            if config.policy.name != "polar_heuristic"
            else 0,
            "observation_fields": tuple(Observation.model_fields),
        }
    return {
        "experiment_id": f"job-{job_id}",
        "stage": "development",
        "evidence_level": "recorded_simulation",
        "controllers": list(controllers.values()),
        "expected_roots": sorted({r["root_seed"] for r in rows}),
        "outcomes": [
            {
                name: row[name]
                for name in ("root_seed", "label", "world_sha256", "status", "elapsed_s", "cost_s")
            }
            for row in rows
        ],
        "failure_cost_s": 2400,
        "proposed_claim": "exploratory",
        "notes": (
            "Registered operational follow-up; B1 and B3 have unequal compute; four worlds "
            "cannot resolve tail efficacy."
        ),
        "provenance": {
            "job_sha256": job_id,
            "numerical_source_sha256": spec.numerical_source_sha256,
            "worker_source_sha256": spec.worker_source_sha256,
            "bridge_source_sha256": file_hash(Path(__file__)),
            "attempt_manifest_sha256": file_hash(attempt / "manifest.json"),
            "config_sha256s": json.dumps([e.config.content_hash() for e in spec.episodes]),
            "run_manifest_sha256s": json.dumps([r["manifest_sha256"] for r in rows]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "submit", "status", "collect"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job")
    args = parser.parse_args()
    if args.operation in {"prepare", "submit"}:
        spec = prepare(Request.model_validate_json(sys.stdin.read(4096)))
        spec.verify_runtime()
        if args.operation == "prepare":
            result = {
                "job_sha256": spec.digest(),
                "episode_tasks": len(spec.episodes),
                "max_wall_s": spec.max_wall_s,
                "spec": spec.model_dump(mode="json"),
            }
        else:
            jobs = LocalJobs(args.root)
            job_id = jobs.submit(spec)
            result = start_if_available(jobs, job_id)
    else:
        if not args.job:
            parser.error("Job identifier required")
        jobs = LocalJobs(args.root)
        result = (
            start_if_available(jobs, args.job)
            if args.operation == "status"
            else collect(jobs, args.job)
        )
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
