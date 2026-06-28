import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sail_lab.classical_study import paired_report
from sail_lab.config import ExperimentConfig
from sail_lab.experiments import file_hash, record_run, write_json
from sail_lab.presentation import export_study


def test_verified_export_recomputes_comparisons_and_rejects_tampering(tmp_path):
    parent = tmp_path / "study"
    parent.mkdir()
    rows = []
    for seed in (1, 2):
        for variant, name, risk in (
            ("B1_belief", "polar_heuristic", 0),
            ("B2_mean", "deterministic_mpc", 0),
            ("B3_mean", "scenario_mpc", 0),
            ("B3_risk", "scenario_mpc", 0.25),
        ):
            cfg = ExperimentConfig.model_validate(
                {
                    "stage": "development",
                    "root_seed": seed,
                    "course": {"deadline_s": 3},
                    "world": {"wind_speed_m_s": 0},
                    "policy": {
                        "name": name,
                        "wind_input": "belief",
                        "planning_horizon_s": 10,
                        "scenario_count": 2,
                        "risk_lambda": risk,
                    },
                }
            )
            directory = record_run(cfg, parent / "runs")
            summary = json.loads((directory / "summary.json").read_text())
            manifest = json.loads((directory / "manifest.json").read_text())
            decisions = json.loads((directory / "decisions.json").read_text())
            rows.append(
                {
                    **summary,
                    "scenario_id": f"constant-{seed}",
                    "variant": variant,
                    "world_hash": manifest["scenario_manifest_hash"],
                    "cost_s": 2400,
                    "run_directory": str(directory),
                    "latencies_s": [d["compute_s"] for d in decisions],
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), parent / "episodes.parquet")
    write_json(parent / "report.json", paired_report(rows))
    write_json(
        parent / "manifest.json",
        {
            "stage": "development_pilot",
            "status": "succeeded",
            "artifacts": {
                name: file_hash(parent / name) for name in ("report.json", "episodes.parquet")
            },
        },
    )
    result = export_study(parent, tmp_path / "export", tmp_path / "public", "constant-1", "B3_risk")
    assert result["paired_comparisons"]["B2_mean"]["policies"]["B3_risk"]["completions"] == 0
    assert json.loads((tmp_path / "public/replay.json").read_text())["policy_label"] == "B3_risk"
    assert (tmp_path / "export/manifest.json").exists()
    (parent / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        export_study(parent, tmp_path / "bad-export", tmp_path / "public", "constant-1", "B3_risk")
