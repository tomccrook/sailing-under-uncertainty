"""Immutable run artefacts, versioned manifests, and a local SQLite catalogue."""

import hashlib
import importlib.metadata
import json
import platform
import sqlite3
import subprocess
import time
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq

from sail_lab.config import Contract, ExperimentConfig
from sail_lab.contracts import EpisodeSummary, ReplayBundle, ReplaySample
from sail_lab.dynamics.simulator import Simulator
from sail_lab.policies import PolarHeuristic, Policy
from sail_lab.streams import seed_hierarchy
from sail_lab.world import FrozenWorld


class RunManifest(Contract):
    schema_version: Literal[1] = 1
    run_id: str
    config_hash: str
    code_revision: str
    dirty_tree: bool
    source_snapshot_hash: str
    dependencies: dict[str, str]
    hardware: dict[str, str]
    device: Literal["cpu"] = "cpu"
    scenario_manifest_hash: str
    training_model_ids: list[str] = []
    seed_hierarchy: dict[str, list[int]]
    started_utc: str
    ended_utc: str
    status: Literal["succeeded", "failed"]
    artifacts: dict[str, str]
    warnings: list[str]
    parent_comparisons: list[str] = []


def simulate(
    config: ExperimentConfig,
    policy: Policy | None = None,
    world: FrozenWorld | None = None,
) -> tuple[EpisodeSummary, list[ReplaySample], list[dict[str, object]]]:
    start = time.perf_counter()
    sim = Simulator(config, world=world)
    if policy is not None:
        agent = policy
    elif config.policy.name == "polar_heuristic":
        agent = PolarHeuristic(config.boat, config.policy)
    else:
        from sail_lab.planning import RecedingHorizonPlanner

        agent = RecedingHorizonPlanner(config)
    agent.reset()
    samples = [sim.sample()]
    decisions: list[dict[str, object]] = []
    compute = 0.0
    next_decision = 0.0
    while sim.state.status == "running":
        if sim.state.elapsed >= next_decision - 1e-8:
            observation = sim.observation()
            before = time.perf_counter()
            action = agent.act(observation)
            elapsed = time.perf_counter() - before
            compute += elapsed
            sim.request(action)
            decisions.append(
                {
                    "elapsed_s": sim.state.elapsed,
                    "observation": observation.model_dump(mode="json"),
                    "action": action.model_dump(mode="json"),
                    "executed_target_rad": sim.state.turn_target,
                    "compute_s": elapsed,
                    "planning": getattr(agent, "last_decision", None),
                }
            )
            next_decision += config.clock.policy_interval_s
        sim.step()
        samples.append(sim.sample())
    state = sim.state
    summary = EpisodeSummary(
        episode_id=f"{config.world.family}-{config.root_seed}",
        policy_id=f"{config.policy.name}/{config.policy.wind_input}",
        status=state.status,
        elapsed_s=state.elapsed,
        distance_m=state.distance,
        tack_count=state.tacks,
        manoeuvre_s=state.manoeuvre_time,
        decisions=len(decisions),
        policy_compute_s=compute,
        simulation_compute_s=time.perf_counter() - start - compute,
    )
    return summary, samples, decisions


def source_metadata() -> tuple[str, bool, str]:
    root = project_root()
    digest = hashlib.sha256()
    for name, content in source_snapshot().items():
        digest.update(name.encode())
        digest.update(content.encode())
    for name in ("uv.lock", "pyproject.toml", "README.md"):
        if (root / name).exists():
            digest.update((root / name).read_bytes())
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
            )
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision, dirty = "unavailable", True
    return revision, dirty, digest.hexdigest()


def project_root() -> Path:
    """Find source checkout or Docker context; installed wheels still snapshot their code."""
    package = Path(__file__).resolve().parent
    for candidate in (Path.cwd(), *package.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src/sail_lab").is_dir():
            return candidate
    return package


def source_snapshot() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {
        f"src/sail_lab/{path.relative_to(package)}": path.read_text()
        for path in sorted(package.rglob("*.py"))
    }


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def record_run(
    config: ExperimentConfig,
    output: Path,
    policy: Policy | None = None,
    policy_label: str | None = None,
    model_id: str | None = None,
) -> Path:
    run_id = f"{config.name}-{uuid.uuid4().hex[:12]}"
    directory = output / run_id
    directory.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC).isoformat()
    revision, dirty, source_hash = source_metadata()
    write_json(directory / "config.json", config.model_dump(mode="json"))
    status: Literal["succeeded", "failed"] = "succeeded"
    warnings = [f"Synthetic uncalibrated polar; stage={config.stage}."]
    if dirty:
        warnings.append("Source tree dirty; snapshot hash and source copy retained.")
    # Preserve numerical source even on faults; a hash alone is not a recoverable snapshot.
    root = project_root()
    write_json(directory / "source-snapshot.json", source_snapshot())
    for name in ("uv.lock", "pyproject.toml", "README.md"):
        if (root / name).exists():
            (directory / name).write_bytes((root / name).read_bytes())
    error: Exception | None = None
    world_hash = hashlib.sha256(config.world.model_dump_json().encode()).hexdigest()
    try:
        world = FrozenWorld(config.world, config.root_seed, config.course.deadline_s)
        world.save(directory / "diagnostic-world.npz")
        world_hash = world.digest
        summary, samples, decisions = (
            simulate(config, world=world)
            if policy is None
            else simulate(config, policy=policy, world=world)
        )
        if policy_label is not None:
            summary = summary.model_copy(update={"policy_id": policy_label})
        write_json(directory / "summary.json", summary.model_dump(mode="json"))
        write_json(directory / "decisions.json", decisions)
        # Diagnostics are in distinct Parquet columns, never passed to policies.
        rows = [
            {
                "run_id": run_id,
                "episode_id": summary.episode_id,
                "policy_id": summary.policy_id,
                **sample.model_dump(mode="json"),
            }
            for sample in samples
        ]
        pq.write_table(pa.Table.from_pylist(rows), directory / "trajectory.parquet")
        pq.write_table(
            pa.Table.from_pylist([summary.model_dump(mode="json")]), directory / "episodes.parquet"
        )
        replay = ReplayBundle(
            run_id=run_id,
            config_hash=config.content_hash(),
            policy_label=summary.policy_id,
            policy_parameters=None if model_id is not None else config.policy,
            course=config.course,
            world_family=config.world.family,
            stage=config.stage,
            summary=summary,
            samples=samples,
        )
        write_json(directory / "replay.json", replay.model_dump(mode="json"))
        from sail_lab.plots import plot_run

        plot_run(config, samples, directory)
    except Exception as exc:
        error = exc
        status = "failed"
        warnings.append(f"Implementation error: {type(exc).__name__}: {exc}")
        (directory / "error.txt").write_text(traceback.format_exc())
    artifacts = {
        path.name: file_hash(path) for path in sorted(directory.iterdir()) if path.is_file()
    }
    manifest = RunManifest(
        run_id=run_id,
        config_hash=config.content_hash(),
        code_revision=revision,
        dirty_tree=dirty,
        source_snapshot_hash=source_hash,
        dependencies={
            name: importlib.metadata.version(name)
            for name in ("numpy", "pydantic", "pyarrow", "matplotlib", "typer", "pyyaml")
        },
        hardware={
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        scenario_manifest_hash=world_hash,
        training_model_ids=[model_id] if model_id is not None else [],
        seed_hierarchy=seed_hierarchy(config.root_seed),
        started_utc=started,
        ended_utc=datetime.now(UTC).isoformat(),
        status=status,
        artifacts=artifacts,
        warnings=warnings,
    )
    write_json(directory / "manifest.json", manifest.model_dump(mode="json"))
    with sqlite3.connect(output / "catalogue.sqlite") as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, "
            "manifest_path TEXT NOT NULL, status TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?)", (run_id, str(directory / "manifest.json"), status)
        )
    if error is not None:
        raise RuntimeError(f"Experiment failed; evidence retained in {directory}") from error
    return directory
