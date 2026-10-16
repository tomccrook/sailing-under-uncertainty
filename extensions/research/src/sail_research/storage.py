"""Immutable registered evidence and a persistent authority ledger."""

import json
import sqlite3
from pathlib import Path
from typing import Any

from sail_research.contracts import Budget, EvidenceRef, Experiment, digest


def read_json(path: Path) -> Any:
    if path.stat().st_size > 2_000_000:
        raise ValueError("Evidence exceeds the 2 MB JSON limit")
    return json.loads(path.read_bytes())


class DeskStore:
    def __init__(self, root: Path, budget: Budget):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("evidence", "experiments", "reports", "jobs"):
            (self.root / name).mkdir(exist_ok=True)
        self.budget = budget
        self.db = self.root / "authority.sqlite"
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS authority (id INTEGER PRIMARY KEY, budget TEXT, "
                "calls INTEGER, jobs INTEGER, wall INTEGER, tasks INTEGER)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS submissions (request TEXT PRIMARY KEY, status TEXT, "
                "job_id TEXT)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY, tool TEXT, arguments "
                "TEXT, response TEXT)"
            )
            encoded = budget.model_dump_json()
            row = db.execute("SELECT budget FROM authority WHERE id=1").fetchone()
            if row is None:
                db.execute("INSERT INTO authority VALUES (1,?,0,0,0,0)", (encoded,))
            elif row[0] != encoded:
                raise ValueError(
                    "Authority budget cannot change on restart; use a new owner-created workspace"
                )

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db, timeout=10)

    def begin_call(self, tool: str, arguments: dict[str, Any]) -> int:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            calls = db.execute("SELECT calls FROM authority WHERE id=1").fetchone()[0]
            if calls >= self.budget.max_tool_calls:
                raise ValueError("Persistent tool-call budget exhausted")
            db.execute("UPDATE authority SET calls=calls+1 WHERE id=1")
            cursor = db.execute(
                "INSERT INTO audit(tool,arguments) VALUES (?,?)",
                (tool, json.dumps(arguments, sort_keys=True)),
            )
            assert cursor.lastrowid is not None
            return cursor.lastrowid

    def end_call(self, call_id: int, response: dict[str, Any]) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE audit SET response=? WHERE id=?",
                (json.dumps(response, sort_keys=True), call_id),
            )

    def reserve(self, request: str, wall: int, tasks: int) -> tuple[bool, str | None]:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute(
                "SELECT job_id FROM submissions WHERE request=?", (request,)
            ).fetchone()
            if old is not None:
                return False, old[0]
            jobs, total_wall, total_tasks = db.execute(
                "SELECT jobs,wall,tasks FROM authority WHERE id=1"
            ).fetchone()
            if (
                jobs + 1 > self.budget.max_jobs
                or total_wall + wall > self.budget.max_reserved_wall_s
                or total_tasks + tasks > self.budget.max_episode_tasks
            ):
                raise ValueError("Cumulative local job/worker-wall/episode budget exhausted")
            db.execute(
                "UPDATE authority SET jobs=jobs+1,wall=wall+?,tasks=tasks+? WHERE id=1",
                (wall, tasks),
            )
            db.execute("INSERT INTO submissions VALUES (?,'reserved',NULL)", (request,))
            return True, None

    def submitted(self, request: str, job_id: str) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE submissions SET status='submitted',job_id=? WHERE request=?",
                (job_id, request),
            )

    def put(self, data: dict[str, Any], description: str) -> EvidenceRef:
        sha = digest(data)
        path = self.root / "evidence" / f"{sha}.json"
        if path.exists():
            self.get(f"evidence://{sha}")
        else:
            with path.open("x") as file:
                file.write(json.dumps(data, sort_keys=True, indent=2) + "\n")
        return EvidenceRef(uri=f"evidence://{sha}", sha256=sha, description=description)

    def get(self, uri: str) -> dict[str, Any]:
        ref = EvidenceRef(uri=uri, sha256=uri.removeprefix("evidence://"), description="lookup")
        data: dict[str, Any] = read_json(self.root / "evidence" / f"{ref.sha256}.json")
        if digest(data) != ref.sha256:
            raise ValueError("Evidence content hash changed")
        return data

    def register(self, experiment: Experiment) -> EvidenceRef:
        ref = self.put(experiment.model_dump(mode="json"), "Registered experiment")
        path = self.root / "experiments" / f"{experiment.experiment_id}.json"
        if path.exists():
            if read_json(path) != ref.model_dump():
                raise ValueError("Experiment identifier is already immutable")
        else:
            with path.open("x") as file:
                file.write(ref.model_dump_json(indent=2) + "\n")
        return ref

    def experiment(self, experiment_id: str) -> tuple[Experiment, EvidenceRef]:
        # Validate before any filename composition.
        from pydantic import TypeAdapter

        from sail_research.contracts import Identifier

        TypeAdapter(Identifier).validate_python(experiment_id)
        ref = EvidenceRef.model_validate(
            read_json(self.root / "experiments" / f"{experiment_id}.json")
        )
        return Experiment.model_validate(self.get(ref.uri)), ref
