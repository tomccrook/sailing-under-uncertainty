"""Reproduce the registered +02 clock-offset sensitivity without changing base methods.

Use the support environment. Whole-voyage splits and source-local day groups stay fixed.
Only relative-time/component conclusions are assessed; absolute UTC stays unverified.
"""

import argparse
import copy
import json
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from scripts.empirical_validation import digest, track_validation, verify, write

METRICS = (
    "raw_records",
    "provenance_excluded_rows",
    "processed_rows",
    "accepted_rows",
    "rejected_rows",
    "projection_relative_speed_error",
    "projected_minus_source_sog_m_s",
    "projection_gate_passed",
)


def shift_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    shifted = copy.deepcopy(records)
    for row in shifted:
        if row["timestamp_utc_assumed"]:
            row["timestamp_utc_assumed"] = (
                datetime.fromisoformat(row["timestamp_utc_assumed"]) - timedelta(hours=1)
            ).isoformat()
    return shifted


def run_sensitivity(run: Path, output: Path) -> dict[str, Any]:
    registration, plan, _ = verify(run)
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "succeeded" or manifest["artifacts"]["report.json"] != digest(
        run / "report.json"
    ):
        raise ValueError("baseline report is incomplete or changed")
    records = pq.read_table(run / "gps-source-records.parquet").to_pylist()
    baseline = json.loads((run / "report.json").read_text())["gps"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "wrapper-source.py").write_bytes(Path(__file__).read_bytes())
    start = time.perf_counter()
    try:
        result = track_validation(shift_records(records), registration, output, plan)
        # The frozen base importer has +01 provenance text. Correct that descriptive
        # metadata for this sensitivity; coordinates, times, flags and metrics are untouched.
        for path in (output / "tracks").glob("*/track.json"):
            track = json.loads(path.read_text())
            track["device_metadata"]["clock"] = (
                "assumed fixed +02 sensitivity; source-local groups preserved; UTC unverified"
            )
            write(path, track)
            manifest_path = path.parent / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"]["track.json"] = digest(path)
            write(manifest_path, manifest)
        comparison = {
            "registered_sensitivity": (
                "fixed UTC+02 instead of +01, same source-day groups and voyage splits"
            ),
            "registration_sha256": digest(run / "registration.json"),
            "source_records_sha256": digest(run / "gps-source-records.parquet"),
            "baseline_report_sha256": digest(run / "report.json"),
            "methods": registration["methods"],
            "wrapper_sha256": digest(Path(__file__)),
            "metrics": {
                k: {
                    "unchanged": result[k] == baseline[k],
                    "original": baseline[k],
                    "offset_plus02": result[k],
                }
                for k in METRICS
            },
            "absolute_UTC_verified": False,
        }
        write(output / "comparison.json", comparison)
        if not all(v["unchanged"] for v in comparison["metrics"].values()):
            raise ValueError("constant-offset sensitivity changed a registered relative metric")
        verify(run)
    except Exception:
        (output / "error.txt").write_text(traceback.format_exc())
        write(output / "manifest.json", {"status": "failed", "wall_s": time.perf_counter() - start})
        raise
    write(
        output / "manifest.json",
        {
            "status": "succeeded",
            "wall_s": time.perf_counter() - start,
            "artifacts": {
                p.relative_to(output).as_posix(): digest(p)
                for p in output.rglob("*")
                if p.is_file()
            },
        },
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=Path("artifacts/measurements/recorded-validation-v2")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/measurements/registered-clock-sensitivity")
    )
    args = parser.parse_args()
    comparison = run_sensitivity(args.run, args.output)
    print(
        json.dumps(
            {"all_metrics_unchanged": all(v["unchanged"] for v in comparison["metrics"].values())}
        )
    )


if __name__ == "__main__":
    main()
