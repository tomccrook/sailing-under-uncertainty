"""Verified precomputed research exports, with no job-submission service."""

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from sail_lab.classical_study import paired_report, plot_study
from sail_lab.config import PolicyConfig
from sail_lab.contracts import ReplayBundle
from sail_lab.experiments import file_hash, source_metadata, source_snapshot, write_json


def export_study(
    study: Path, output: Path, public: Path, scenario_id: str = "ou-42004", variant: str = "B3_risk"
) -> dict[str, Any]:
    manifest = json.loads((study / "manifest.json").read_text())
    if manifest["stage"] != "development_pilot" or manifest["status"] != "succeeded":
        raise ValueError("only successful development studies can use this export")
    for name in ("report.json", "episodes.parquet"):
        if manifest["artifacts"][name] != file_hash(study / name):
            raise ValueError(f"study artefact hash mismatch: {name}")
    rows = pq.read_table(study / "episodes.parquet").to_pylist()
    report: dict[str, Any] = json.loads((study / "report.json").read_text())
    report["paired_comparisons"] = {
        reference: paired_report(rows, reference)
        for reference in ("B1_belief", "B2_mean", "B3_mean")
    }
    selected = [r for r in rows if r["scenario_id"] == scenario_id and r["variant"] == variant]
    if len(selected) != 1:
        raise ValueError("demo selection must identify one retained episode")
    folder = Path(selected[0]["run_directory"])
    run_manifest = json.loads((folder / "manifest.json").read_text())
    for name in ("replay.json", "config.json", "decisions.json"):
        if run_manifest["artifacts"][name] != file_hash(folder / name):
            raise ValueError(f"recorded run artefact hash mismatch: {name}")
    payload = json.loads((folder / "replay.json").read_text())
    parameters = json.loads((folder / "config.json").read_text())["policy"]
    if parameters["name"] == "deterministic_mpc" and "deterministic_sensor_model" not in parameters:
        parameters["deterministic_sensor_model"] = "sampled"
    expected_name = (
        "polar_heuristic"
        if variant.startswith("B1_")
        else "deterministic_mpc"
        if variant.startswith("B2_")
        else "scenario_mpc"
    )
    if parameters["name"] != expected_name:
        raise ValueError("variant disagrees with recorded controller")
    if run_manifest["scenario_manifest_hash"] != selected[0]["world_hash"]:
        raise ValueError("selected replay does not match paired world")
    if (
        payload["summary"]["elapsed_s"] != selected[0]["elapsed_s"]
        or payload["summary"]["status"] != selected[0]["status"]
    ):
        raise ValueError("selected replay does not match retained outcome")
    payload["policy_parameters"] = PolicyConfig.model_validate(parameters).model_dump(mode="json")
    payload["policy_label"] = variant
    replay = ReplayBundle.model_validate(payload)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "results.json", report)
    write_json(output / "replay.json", replay.model_dump(mode="json"))
    write_json(output / "source-snapshot.json", source_snapshot())
    plot_study(output, report)
    revision, dirty, source_hash = source_metadata()
    write_json(
        output / "manifest.json",
        {
            "stage": "development_export",
            "parent_manifest_hash": file_hash(study / "manifest.json"),
            "source_episode_manifest_hash": file_hash(folder / "manifest.json"),
            "code_revision": revision,
            "dirty_tree": dirty,
            "source_snapshot_hash": source_hash,
            "demo_scenario_id": scenario_id,
            "demo_variant": variant,
            "derived_display_fields": ["policy_label", "policy_parameters"],
            "artifacts": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()},
        },
    )
    public.mkdir(parents=True, exist_ok=True)
    for name in ("results.json", "replay.json"):
        (public / name).write_bytes((output / name).read_bytes())
    return report
