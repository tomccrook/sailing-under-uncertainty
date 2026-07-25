import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from test_cloud import MemoryS3

from sail_support.cloud import S3Store
from sail_support.cloud_worker import run_worker


def test_batch_entrypoint_round_trips_a_real_failed_sailing_episode_locally(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    numerical_python = root / ".venv/bin/python"
    if not numerical_python.exists():
        pytest.skip("Run the documented root environment sync for the cross-runtime worker check")
    code = """from pathlib import Path
import sys
from scripts.bounded_jobs import prepare_demo, JobSpec
spec = prepare_demo(Path(sys.argv[1]))
data = spec.model_dump()
data['episodes'] = data['episodes'][:1]
data['episodes'][0]['config']['course']['deadline_s'] = 20
Path(sys.argv[2]).write_text(JobSpec.model_validate(data).model_dump_json())
"""
    source = tmp_path / "job.json"
    subprocess.run(
        [str(numerical_python), "-c", code, str(tmp_path / "full.json"), str(source)],
        cwd=root,
        check=True,
        timeout=30,
    )
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    memory = MemoryS3()
    store = S3Store(memory, "private")
    key = f"jobs/specs/{digest}.json"
    store.put(key, raw)
    environment = {
        "SAIL_SPEC_KEY": key,
        "SAIL_SPEC_SHA256": digest,
        "SAIL_JOB_SHA256": digest,
        "AWS_BATCH_JOB_ARRAY_INDEX": "0",
        "AWS_BATCH_JOB_ATTEMPT": "1",
    }
    assert run_worker(store, numerical_python, tmp_path / "attempt", environment) == 0
    bundle_key = f"jobs/results/{digest}/0/attempt-1/bundle.json"
    bundle_hash = hashlib.sha256(memory.files[bundle_key]).hexdigest()
    store.download_directory(bundle_key, bundle_hash, tmp_path / "download")
    for path in (tmp_path / "attempt").rglob("*"):
        if path.is_file():
            assert (
                tmp_path / "download" / path.relative_to(tmp_path / "attempt")
            ).read_bytes() == path.read_bytes()
    subprocess.run(
        [
            str(numerical_python),
            "-m",
            "scripts.bounded_jobs",
            "merge",
            "--spec",
            str(source),
            "--shards",
            str(tmp_path / "download/shard"),
            "--output",
            str(tmp_path / "merged.json"),
        ],
        cwd=root,
        check=True,
        timeout=30,
    )
    row = json.loads((tmp_path / "merged.json").read_text())[0]
    assert row["status"] == "deadline_exceeded" and row["cost_s"] == 2400
    assert json.loads((tmp_path / "download/attempt.json").read_text())["worker_exit_code"] == 0
