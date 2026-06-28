"""Verify recorded evidence and publish a bounded, static portfolio catalogue."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from sail_lab.config import ExperimentConfig
from sail_lab.contracts import ReplayBundle
from sail_lab.experiments import file_hash, project_root, write_json
from sail_lab.locked_evaluation import verify_manifest
from scripts.demo_contracts import Asset, Catalogue, Details, Recording, Scenario


def pairing_hash(config: ExperimentConfig) -> str:
    """Includes course, clocks, world root, instruments, forecast and boat dynamics."""
    value = config.model_dump(mode="json")
    for key in ("name", "policy"):
        value.pop(key)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def asset(path: Path, output: Path) -> Asset:
    return Asset(url="/demo/" + path.relative_to(output).as_posix(), sha256=file_hash(path))


def publish_recording(
    entry: dict[str, Any], row: dict[str, Any], output: Path
) -> tuple[Recording, str]:
    directory = Path(entry["run_directory"])
    if file_hash(directory / "manifest.json") != entry["manifest_hash"]:
        raise ValueError("recording manifest changed")
    verify_manifest(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    config = ExperimentConfig.model_validate_json((directory / "config.json").read_text())
    replay = ReplayBundle.model_validate_json((directory / "replay.json").read_text())
    status = replay.summary.status
    if status not in ("finished", "deadline_exceeded", "boundary_violation"):
        raise ValueError("implementation faults cannot be presented as sailing outcomes")
    if (
        replay.config_hash != config.content_hash()
        or replay.config_hash != row["config_hash"]
        or manifest["config_hash"] != replay.config_hash
        or replay.run_id != manifest["run_id"]
        or replay.stage != "locked"
        or replay.policy_label != row["variant"]
        or replay.course != config.course
        or manifest["scenario_manifest_hash"] != row["world_hash"]
        or replay.summary.status != row["status"]
        or abs(replay.summary.elapsed_s - row["elapsed_s"]) > 1e-8
        or abs(replay.summary.distance_m - row["distance_m"]) > 1e-8
        or replay.summary.tack_count != row["tack_count"]
        or entry["cost_s"] != row["cost_s"]
    ):
        raise ValueError("recording disagrees with locked outcome or configuration")
    if len(replay.samples) > 5000:
        raise ValueError("public replay exceeds sample budget")
    model_ids = manifest["training_model_ids"]
    details = Details(
        run_id=replay.run_id,
        world_hash=row["world_hash"],
        variant=row["variant"],
        model_hash=model_ids[0] if model_ids else None,
        config=config,
        decisions=json.loads((directory / "decisions.json").read_text()),
        probabilities=entry["probabilities"],
    )
    if len(details.decisions) != replay.summary.decisions:
        raise ValueError("missing recorded decisions")
    if details.decisions[-1].elapsed_s >= replay.summary.elapsed_s:
        raise ValueError("decision after terminal outcome")
    destination = output / replay.run_id
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("replay.json", "config.json", "manifest.json", "summary.json"):
        if (directory / name).stat().st_size > 12_000_000:
            raise ValueError("public file exceeds byte budget")
        shutil.copyfile(directory / name, destination / name)
    write_json(destination / "details.json", details.model_dump(mode="json"))
    return (
        Recording(
            variant=row["variant"],
            run_id=replay.run_id,
            status=status,
            elapsed_s=replay.summary.elapsed_s,
            cost_s=row["cost_s"],
            replay=asset(destination / "replay.json", output),
            details=asset(destination / "details.json", output),
            manifest=asset(destination / "manifest.json", output),
        ),
        pairing_hash(config),
    )


def export(
    study: Path, recordings: Path, output: Path, public: Path, calibration: Path, figures: Path
) -> None:
    # Historical publication verifies saved source/lock hashes; it does not execute that source.
    lock = json.loads((study / "lock.json").read_text())
    lock_hash = file_hash(study / "lock.json")
    if lock_hash != (study / "lock.sha256").read_text().strip():
        raise ValueError("lock changed")
    for name, key in (
        ("scenarios.json", "scenario_manifest_hash"),
        ("source-snapshot.json", "source_snapshot_hash"),
    ):
        if file_hash(study / name) != lock[key]:
            raise ValueError("locked evidence changed")
    verify_manifest(study / "evaluation")
    verify_manifest(recordings)
    metadata = json.loads((recordings / "manifest.json").read_text())
    if metadata["lock_hash"] != lock_hash or metadata["evaluation_manifest_hash"] != file_hash(
        study / "evaluation/manifest.json"
    ):
        raise ValueError("recordings refer to a different evaluation")
    entries = json.loads((recordings / "recordings.json").read_text())
    selection = json.loads((recordings / "selection.json").read_text())
    rows = pq.read_table(study / "evaluation/episodes.parquet").to_pylist()
    index = {(r["scenario_id"], r["variant"]): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate evaluated episode")
    output.mkdir(parents=True, exist_ok=False)
    scenarios = []
    models = {m["variant"]: m["model_hash"] for m in lock["models"]}
    for scenario_id, (label, reason) in selection.items():
        selected = [e for e in entries if e["scenario_id"] == scenario_id]
        expected = {r["variant"] for r in rows if r["scenario_id"] == scenario_id}
        if len(selected) != len(expected) or {e["variant"] for e in selected} != expected:
            raise ValueError("missing or duplicate controller recording")
        for entry in selected:
            recorded_manifest = json.loads(
                (Path(entry["run_directory"]) / "manifest.json").read_text()
            )
            expected_models = [models[entry["variant"]]] if entry["variant"] in models else []
            if recorded_manifest["training_model_ids"] != expected_models:
                raise ValueError("recording uses a different frozen model")
        published = [
            publish_recording(e, index[(scenario_id, e["variant"])], output) for e in selected
        ]
        worlds = {index[(scenario_id, e["variant"])]["world_hash"] for e in selected}
        pairs = {p for _, p in published}
        slices = {index[(scenario_id, e["variant"])]["slice"] for e in selected}
        if len(worlds) != 1 or len(pairs) != 1 or len(slices) != 1:
            raise ValueError("controllers do not share the same exogenous scenario")
        scenarios.append(
            Scenario(
                scenario_id=scenario_id,
                label=label,
                selection_reason=reason,
                slice=slices.pop(),
                world_hash=worlds.pop(),
                pairing_hash=pairs.pop(),
                recordings=[r for r, _ in published],
            )
        )
    documents = {}
    for name in (
        "research-report",
        "data-card",
        "model-card",
        "interview-notes",
        "portfolio-demo",
        "measurement-support",
        "bounded-jobs",
        "aws-support",
        "mcp-investigation",
        "planner-replication",
        "compute-followup",
        "recorded-validation",
    ):
        destination = output / f"{name}.md"
        shutil.copyfile(project_root() / f"docs/{name}.md", destination)
        documents[name] = asset(destination, output)
    fields = [
        "scenario_id",
        "world_hash",
        "variant",
        "slice",
        "status",
        "cost_s",
        "elapsed_s",
        "tack_count",
        "distance_m",
        "manoeuvre_s",
        "action_switch_count",
        "config_hash",
    ]
    with (output / "outcomes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["scenario_id"], r["variant"])))
    documents["all-locked-outcomes"] = asset(output / "outcomes.csv", output)
    plot_manifest = json.loads((figures / "manifest.json").read_text())
    if plot_manifest["source_report_hash"] != file_hash(study / "evaluation/report.json"):
        raise ValueError("scientific figures refer to a different report")
    for name in ("paired-learning", "mean-tail"):
        for extension in ("svg", "pdf"):
            filename = f"{name}.{extension}"
            if file_hash(figures / filename) != plot_manifest["artifacts"][filename]:
                raise ValueError("scientific figure changed")
            shutil.copyfile(figures / filename, output / filename)
            documents[filename] = asset(output / filename, output)
    calibration_manifest = json.loads((calibration / "manifest.json").read_text())
    if file_hash(calibration / "report.json") != calibration_manifest["artifacts"]["report.json"]:
        raise ValueError("calibration report changed")
    shutil.copyfile(calibration / "report.json", output / "calibration.json")
    catalogue = Catalogue(
        lock_hash=lock_hash,
        evaluation_manifest_hash=file_hash(study / "evaluation/manifest.json"),
        default_scenario_id=scenarios[0].scenario_id,
        scenarios=scenarios,
        documents=documents,
        calibration=asset(output / "calibration.json", output),
    )
    write_json(output / "catalogue.json", catalogue.model_dump(mode="json"))
    write_json(
        output / "manifest.json",
        {
            "status": "succeeded",
            "recording_manifest_hash": file_hash(recordings / "manifest.json"),
            "calibration_manifest_hash": file_hash(calibration / "manifest.json"),
            "figure_manifest_hash": file_hash(figures / "manifest.json"),
            "publisher_hash": file_hash(Path(__file__)),
            "contract_hash": file_hash(Path(__file__).with_name("demo_contracts.py")),
            "artifacts": {
                p.relative_to(output).as_posix(): file_hash(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            },
        },
    )
    # Replace the complete static directory only after validation succeeds.
    public.mkdir(parents=True, exist_ok=True)
    temporary = public / "demo-next"
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(output, temporary)
    if (public / "demo").exists():
        shutil.rmtree(public / "demo")
    temporary.rename(public / "demo")
    shutil.copyfile(output / "catalogue.json", public / "catalogue.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public", type=Path, default=Path("web/public"))
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    export(args.study, args.recordings, args.output, args.public, args.calibration, args.figures)


if __name__ == "__main__":
    main()
