"""CPU Batch entrypoint: immutable input, indexed episode, manifest-preserving attempts."""

import hashlib
import json
import os
import signal
import subprocess
import sys
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError

from sail_support.cloud import S3Store


def run_worker(
    store: S3Store, numerical_python: Path, output: Path, environment: Mapping[str, str]
) -> int:
    raw = store.get(environment["SAIL_SPEC_KEY"], environment["SAIL_SPEC_SHA256"], 5_000_000)
    output.mkdir(parents=True, exist_ok=False)
    (output / "cloud-worker-source.py").write_bytes(Path(__file__).read_bytes())
    spec_path = output / "input-job.json"
    spec_path.write_bytes(raw)
    info = json.loads(
        subprocess.check_output(
            [
                str(numerical_python),
                "-m",
                "scripts.bounded_jobs",
                "check-spec",
                "--spec",
                str(spec_path),
            ],
            text=True,
            timeout=30,
        )
    )
    job_hash = environment["SAIL_JOB_SHA256"]
    index = int(environment.get("AWS_BATCH_JOB_ARRAY_INDEX", "0"))
    attempt = int(environment.get("AWS_BATCH_JOB_ATTEMPT", "1"))
    if (
        info["job_sha256"] != job_hash
        or not 0 <= index < info["episodes"]
        or not 1 <= attempt <= info["max_attempts"]
    ):
        raise ValueError("Batch index/attempt or job identity differs from bounded input")
    cancel = output / "cancel"
    old_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda _signal, _frame: cancel.touch())
    result = 2
    try:
        with (output / "worker.log").open("wb") as log:
            process = subprocess.Popen(
                [
                    str(numerical_python),
                    "-m",
                    "scripts.bounded_jobs",
                    "shard",
                    "--spec",
                    str(spec_path),
                    "--index",
                    str(index),
                    "--output",
                    str(output / "shard"),
                ],
                stdout=log,
                stderr=log,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                },
            )
            try:
                result = process.wait(timeout=info["max_wall_s"] + 30)
            except subprocess.TimeoutExpired:
                cancel.touch()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                result = 2
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        (output / "attempt.json").write_text(
            json.dumps(
                {
                    "job_sha256": job_hash,
                    "index": index,
                    "attempt": attempt,
                    "worker_exit_code": result,
                    "cancel_requested": cancel.exists(),
                    "cloud_worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "support_versions": {
                        name: version(name) for name in ("boto3", "botocore", "pydantic")
                    },
                },
                indent=2,
            )
            + "\n"
        )
        key = f"jobs/results/{job_hash}/{index}/attempt-{attempt}/bundle.json"
        digest = store.upload_directory(output, key)
        print(json.dumps({"bundle_key": key, "bundle_sha256": digest, "worker_exit_code": result}))
    return 0 if result == 0 else 2


def main() -> None:
    for name in ("SAIL_ARTIFACT_BUCKET", "SAIL_SPEC_KEY", "SAIL_SPEC_SHA256", "SAIL_JOB_SHA256"):
        if not os.environ.get(name):
            raise SystemExit(f"Missing immutable job environment: {name}")
    client = boto3.client(
        "s3",
        config=Config(
            retries={"max_attempts": 2, "mode": "standard"}, connect_timeout=5, read_timeout=15
        ),
    )
    store = S3Store(client, os.environ["SAIL_ARTIFACT_BUCKET"])
    try:
        with TemporaryDirectory(prefix="sail-batch-") as temporary:
            result = run_worker(
                store, Path("/app/.venv/bin/python"), Path(temporary) / "attempt", os.environ
            )
    except EndpointConnectionError as error:
        print(f"Transient artifact transport failure: {error}", file=sys.stderr)
        result = 75
    except ClientError as error:
        code = error.response["Error"]["Code"]
        print(f"Artifact transport failure: {code}", file=sys.stderr)
        result = (
            75
            if code in {"SlowDown", "InternalError", "ServiceUnavailable", "RequestTimeout"}
            else 2
        )
    except Exception as error:
        print(f"Worker validation/fault: {type(error).__name__}: {error}", file=sys.stderr)
        result = 2
    raise SystemExit(result)


if __name__ == "__main__":
    main()
