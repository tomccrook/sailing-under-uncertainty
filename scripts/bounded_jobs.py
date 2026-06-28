"""Internal bounded CPU workers. SQLite locally; the same shard contract in Batch."""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from sail_lab.classical_study import realised_cost, scenario_config
from sail_lab.config import Contract, ExperimentConfig, PolicyConfig
from sail_lab.contracts import Action, Observation
from sail_lab.experiments import file_hash, project_root, record_run, write_json
from sail_lab.locked_evaluation import dependency_versions, numerical_source_hash, verify_manifest
from sail_lab.planning import RecedingHorizonPlanner
from sail_lab.policies import PolarHeuristic, Policy


class Episode(Contract):
    label: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,60}$")
    config: ExperimentConfig


class JobSpec(Contract):
    schema_version: Literal["cpu-job-v1"] = "cpu-job-v1"
    purpose: Literal["development_compatibility"] = "development_compatibility"
    numerical_source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    worker_source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    dependency_versions: dict[str, str]
    episodes: tuple[Episode, ...] = Field(min_length=1, max_length=8)
    max_wall_s: float = Field(default=600, ge=1, le=600)
    max_attempts: int = Field(default=2, ge=1, le=2)
    workers: Literal[1] = 1

    @model_validator(mode="after")
    def bounded_development_only(self) -> "JobSpec":
        keys = []
        worlds: dict[int, str] = {}
        for episode in self.episodes:
            config = episode.config
            if config.stage != "development" or not 600000 <= config.root_seed < 700000:
                raise ValueError("Internal jobs require fresh development roots 600000–699999")
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", config.name):
                raise ValueError("Experiment name must be a safe local identifier")
            if (
                config.course.deadline_s > 1800
                or config.clock.physics_dt_s < 1
                or config.clock.policy_interval_s < 10
                or config.clock.sensor_interval_s < 2
                or config.clock.forecast_interval_s < 30
                or config.world.grid_dt_s < 2
                or config.forecast.horizon_s > 600
                or config.forecast.point_interval_s < 10
                or config.policy.planning_horizon_s > 120
                or config.policy.scenario_count > 16
            ):
                raise ValueError("Job exceeds the CPU step/planning budget")
            key = (config.root_seed, episode.label)
            keys.append(key)
            fields = config.model_dump(mode="json", exclude={"name", "policy"})
            world = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
            if config.root_seed in worlds and worlds[config.root_seed] != world:
                raise ValueError(
                    "Paired policies require identical non-policy scenario configuration"
                )
            worlds[config.root_seed] = world
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate policy/world task")
        return self

    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    def verify_runtime(self) -> None:
        if (
            numerical_source_hash() != self.numerical_source_sha256
            or file_hash(Path(__file__)) != self.worker_source_sha256
            or dependency_versions() != self.dependency_versions
        ):
            raise ValueError("Worker/source/dependencies changed; prepare a new job specification")


class Cancelled(RuntimeError):
    pass


class WallLimit(RuntimeError):
    pass


class BudgetedPolicy:
    def __init__(self, policy: Policy, deadline: float, cancel: Path):
        self.policy, self.deadline, self.cancel = policy, deadline, cancel

    @property
    def last_decision(self) -> Any:
        return getattr(self.policy, "last_decision", None)

    def check(self) -> None:
        if self.cancel.exists():
            raise Cancelled("Cancellation requested; partial/fault records retained")
        if time.monotonic() >= self.deadline:
            raise WallLimit("CPU worker wall-clock budget exceeded")

    def reset(self) -> None:
        self.check()
        self.policy.reset()

    def act(self, observation: Observation) -> Action:
        self.check()
        action = self.policy.act(observation)
        self.check()
        return action


def _cause(error: BaseException, target: type[BaseException]) -> bool:
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, target):
            return True
        current = current.__cause__
    return False


def evaluate_tasks(spec: JobSpec, indices: tuple[int, ...], output: Path, cancel: Path) -> None:
    """All-or-incomplete job, retaining each genuine sailing failure and every worker fault."""
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "job.json", spec.model_dump(mode="json"))
    (output / "worker-source.py").write_bytes(Path(__file__).read_bytes())
    started = datetime.now(UTC).isoformat()
    start = time.monotonic()
    rows: list[dict[str, Any]] = []
    status = "failed"
    failure: str | None = None
    error: Exception | None = None
    try:
        spec.verify_runtime()
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("Task indices must be nonempty and unique")
        for index in indices:
            if not 0 <= index < len(spec.episodes):
                raise ValueError("Shard index out of range")
            episode = spec.episodes[index]
            config = episode.config
            policy: Policy = (
                PolarHeuristic(config.boat, config.policy)
                if config.policy.name == "polar_heuristic"
                else RecedingHorizonPlanner(config)
            )
            guarded = BudgetedPolicy(policy, start + spec.max_wall_s, cancel)
            run = record_run(config, output / "runs", policy=guarded, policy_label=episode.label)
            manifest = verify_manifest(run)
            summary = json.loads((run / "summary.json").read_text())
            rows.append(
                {
                    "index": index,
                    "label": episode.label,
                    "root_seed": config.root_seed,
                    "config_sha256": config.content_hash(),
                    "world_sha256": manifest["scenario_manifest_hash"],
                    "status": summary["status"],
                    "elapsed_s": summary["elapsed_s"],
                    "cost_s": realised_cost(
                        summary["status"], summary["elapsed_s"], config.policy.failure_cost_s
                    ),
                    "distance_m": summary["distance_m"],
                    "tacks": summary["tack_count"],
                    "run_path": str(run.relative_to(output)),
                    "manifest_sha256": file_hash(run / "manifest.json"),
                }
            )
            guarded.check()
        status = "succeeded"
    except Exception as caught:
        error = caught
        status = "cancelled" if _cause(caught, Cancelled) else "failed"
        failure = (
            "wall_limit"
            if _cause(caught, WallLimit)
            else ("cancelled" if status == "cancelled" else "worker_fault")
        )
        (output / "error.txt").write_text(traceback.format_exc())
    finally:
        write_json(output / "episodes.json", rows)
        write_json(
            output / "manifest.json",
            {
                "schema_version": "cpu-shard-v1",
                "job_sha256": spec.digest(),
                "indices": indices,
                "status": status,
                "failure_kind": failure,
                "started_utc": started,
                "ended_utc": datetime.now(UTC).isoformat(),
                "wall_s": time.monotonic() - start,
                "completed_episodes": len(rows),
                "expected_episodes": len(indices),
                "numerical_source_sha256": spec.numerical_source_sha256,
                "worker_source_sha256": spec.worker_source_sha256,
                "dependency_versions": spec.dependency_versions,
                "files": {
                    name: file_hash(output / name)
                    for name in ("job.json", "episodes.json", "worker-source.py")
                },
            },
        )
    if error is not None:
        raise error


def merge_shards(spec: JobSpec, outputs: tuple[Path, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for output in outputs:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["job_sha256"] != spec.digest() or manifest["status"] != "succeeded":
            raise ValueError(
                "Incomplete or incompatible shard; worker faults cannot become episode scores"
            )
        if (
            manifest["numerical_source_sha256"] != spec.numerical_source_sha256
            or manifest["worker_source_sha256"] != spec.worker_source_sha256
            or manifest["dependency_versions"] != spec.dependency_versions
            or file_hash(output / "worker-source.py") != spec.worker_source_sha256
            or JobSpec.model_validate_json((output / "job.json").read_text()).digest()
            != spec.digest()
        ):
            raise ValueError("Shard provenance differs from the declared job")
        if manifest["files"]["job.json"] != file_hash(output / "job.json"):
            raise ValueError("Job specification changed")
        if manifest["files"]["episodes.json"] != file_hash(output / "episodes.json"):
            raise ValueError("Episode table changed")
        part: list[dict[str, Any]] = json.loads((output / "episodes.json").read_text())
        if len(part) != manifest["expected_episodes"] or sorted(r["index"] for r in part) != sorted(
            manifest["indices"]
        ):
            raise ValueError("Shard episode count/index mismatch")
        for row in part:
            index = row["index"]
            if not isinstance(index, int) or not 0 <= index < len(spec.episodes):
                raise ValueError("Invalid episode index")
            task = spec.episodes[index]
            if (
                row["config_sha256"] != task.config.content_hash()
                or row["label"] != task.label
                or row["root_seed"] != task.config.root_seed
            ):
                raise ValueError("Episode config/label mismatch")
            run = (output / row["run_path"]).resolve()
            if not run.is_relative_to(output.resolve()):
                raise ValueError("Run path escapes shard")
            parent = verify_manifest(run)
            summary = json.loads((run / "summary.json").read_text())
            expected = realised_cost(
                summary["status"], summary["elapsed_s"], task.config.policy.failure_cost_s
            )
            if (
                row["manifest_sha256"] != file_hash(run / "manifest.json")
                or row["world_sha256"] != parent["scenario_manifest_hash"]
                or row["status"] != summary["status"]
                or row["elapsed_s"] != summary["elapsed_s"]
                or row["cost_s"] != expected
                or row["distance_m"] != summary["distance_m"]
                or row["tacks"] != summary["tack_count"]
                or parent["config_hash"] != task.config.content_hash()
            ):
                raise ValueError("Episode outcome differs from its recorded run")
            rows.append({**row, "run_path": str(run)})
    if sorted(row["index"] for row in rows) != list(range(len(spec.episodes))):
        raise ValueError("Missing or duplicate shard episodes; no partial-study aggregation")
    worlds: dict[int, str] = {}
    for row in rows:
        seed = row["root_seed"]
        if seed in worlds and worlds[seed] != row["world_sha256"]:
            raise ValueError("Paired policies saw different frozen worlds")
        worlds[seed] = row["world_sha256"]
    return sorted(rows, key=lambda row: row["index"])


class LocalJobs:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "jobs.sqlite"
        with self.connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, attempt INTEGER NOT NULL,
                pid INTEGER, updated_utc TEXT NOT NULL, error TEXT)""")

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def directory(self, job_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{64}", job_id):
            raise ValueError("Invalid content-addressed job identifier")
        return self.root / job_id

    def submit(self, spec: JobSpec) -> str:
        spec.verify_runtime()
        job_id = spec.digest()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT id FROM jobs WHERE id=?", (job_id,)).fetchone():
                return job_id
            directory = self.directory(job_id)
            directory.mkdir(exist_ok=False)
            write_json(directory / "job.json", spec.model_dump(mode="json"))
            connection.execute(
                "INSERT INTO jobs VALUES (?, 'queued', 1, NULL, ?, NULL)",
                (job_id, datetime.now(UTC).isoformat()),
            )
        return job_id

    def status(self, job_id: str) -> dict[str, Any]:
        self.directory(job_id)
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise ValueError("Unknown local job")
            if row["status"] in {"running", "cancel_requested"} and row["pid"]:
                try:
                    os.kill(row["pid"], 0)
                except ProcessLookupError:
                    connection.execute(
                        "UPDATE jobs SET status='failed', error=?, updated_utc=? WHERE id=?",
                        (
                            "Worker stopped without a terminal status; partial files retained",
                            datetime.now(UTC).isoformat(),
                            job_id,
                        ),
                    )
                    row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            assert row is not None
            return dict(row)

    def start(self, job_id: str) -> int:
        if self.status(job_id)["status"] != "queued":
            raise ValueError("Only a queued job may start")
        directory = self.directory(job_id)
        with (directory / f"worker-{self.status(job_id)['attempt']}.log").open("ab") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "scripts.bounded_jobs",
                    "worker",
                    "--root",
                    str(self.root),
                    "--job",
                    job_id,
                ],
                cwd=project_root(),
                stdout=log,
                stderr=log,
                start_new_session=True,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                },
            )
        return process.pid

    def cancel(self, job_id: str) -> None:
        directory = self.directory(job_id)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise ValueError("Unknown local job")
            if row["status"] in {"queued", "running", "cancel_requested"}:
                (directory / "cancel").touch()
                state = "cancelled" if row["status"] == "queued" else "cancel_requested"
                connection.execute(
                    "UPDATE jobs SET status=?, updated_utc=? WHERE id=?",
                    (state, datetime.now(UTC).isoformat(), job_id),
                )

    def retry(self, job_id: str) -> None:
        directory = self.directory(job_id)
        spec = JobSpec.model_validate_json((directory / "job.json").read_text())
        spec.verify_runtime()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None or row["status"] not in {"failed", "cancelled"}:
                raise ValueError("Retry requires an inspected failed/cancelled job")
            if row["attempt"] >= spec.max_attempts:
                raise ValueError("Retry attempt budget exhausted")
            (directory / "cancel").unlink(missing_ok=True)
            connection.execute(
                "UPDATE jobs SET status='queued', attempt=attempt+1, pid=NULL, "
                "updated_utc=?, error=NULL WHERE id=?",
                (datetime.now(UTC).isoformat(), job_id),
            )

    def worker(self, job_id: str) -> None:
        directory = self.directory(job_id)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('running', 'cancel_requested')"
            ).fetchone()[0]:
                raise ValueError(
                    "Local registry permits one active worker; start this queued job later"
                )
            cursor = connection.execute(
                "UPDATE jobs SET status='running', pid=?, updated_utc=? "
                "WHERE id=? AND status='queued'",
                (os.getpid(), datetime.now(UTC).isoformat(), job_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Job is no longer queued")
            row = connection.execute("SELECT attempt FROM jobs WHERE id=?", (job_id,)).fetchone()
            assert row is not None
            attempt = row["attempt"]
        state, message = "failed", None
        try:
            spec = JobSpec.model_validate_json((directory / "job.json").read_text())
            evaluate_tasks(
                spec,
                tuple(range(len(spec.episodes))),
                directory / f"attempt-{attempt}",
                directory / "cancel",
            )
            state = "succeeded"
        except Exception as error:
            state = "cancelled" if _cause(error, Cancelled) else "failed"
            message = f"{type(error).__name__}: {error}"
        finally:
            with self.connect() as connection:
                connection.execute(
                    "UPDATE jobs SET status=?, error=?, updated_utc=? WHERE id=?",
                    (state, message, datetime.now(UTC).isoformat(), job_id),
                )


def prepare_demo(output: Path) -> JobSpec:
    policies = {
        "B1_belief": PolicyConfig(wind_input="belief", layline_margin_s=5, boundary_guard_s=45),
        "B2_mean": PolicyConfig(
            name="deterministic_mpc", wind_input="belief", risk_lambda=0, boundary_guard_s=30
        ),
        "B3_expect": PolicyConfig(
            name="scenario_mpc", wind_input="belief", risk_lambda=0, boundary_guard_s=30
        ),
        "B3_risk": PolicyConfig(
            name="scenario_mpc", wind_input="belief", risk_lambda=0.25, boundary_guard_s=30
        ),
    }
    spec = JobSpec(
        numerical_source_sha256=numerical_source_hash(),
        worker_source_sha256=file_hash(Path(__file__)),
        dependency_versions=dependency_versions(),
        episodes=tuple(
            Episode(label=label, config=scenario_config("ou", 610001, policy))
            for label, policy in policies.items()
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as file:
        file.write(spec.model_dump_json(indent=2) + "\n")
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare-demo",
            "submit",
            "start",
            "status",
            "cancel",
            "retry",
            "worker",
            "shard",
            "merge",
            "check-spec",
        ),
    )
    parser.add_argument("--root", type=Path, default=Path("artifacts/jobs/local"))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--job")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--shards", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "prepare-demo":
        if args.output is None:
            parser.error("prepare-demo requires --output")
        print(prepare_demo(args.output).digest())
        return
    if args.command in {"submit", "shard", "merge", "check-spec"}:
        if args.spec is None:
            parser.error("Command requires --spec")
        spec = JobSpec.model_validate_json(args.spec.read_text())
        if args.command == "check-spec":
            spec.verify_runtime()
            print(
                json.dumps(
                    {
                        "job_sha256": spec.digest(),
                        "episodes": len(spec.episodes),
                        "max_wall_s": spec.max_wall_s,
                        "max_attempts": spec.max_attempts,
                    }
                )
            )
        elif args.command == "shard":
            if args.output is None or args.index is None:
                parser.error("shard requires --output and --index")
            evaluate_tasks(spec, (args.index,), args.output, args.output.parent / "cancel")
        elif args.command == "merge":
            if args.output is None or not args.shards:
                parser.error("merge requires --output and --shards")
            rows = merge_shards(spec, tuple(args.shards))
            with args.output.open("x") as file:
                json.dump(rows, file, indent=2, allow_nan=False)
        else:
            print(LocalJobs(args.root).submit(spec))
        return
    if args.job is None:
        parser.error("Command requires --job")
    registry = LocalJobs(args.root)
    if args.command == "status":
        print(json.dumps(registry.status(args.job), indent=2))
    elif args.command == "start":
        print(registry.start(args.job))
    elif args.command == "cancel":
        registry.cancel(args.job)
    elif args.command == "retry":
        registry.retry(args.job)
    elif args.command == "worker":
        registry.worker(args.job)


if __name__ == "__main__":
    main()
