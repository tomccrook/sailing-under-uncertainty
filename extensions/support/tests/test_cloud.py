import hashlib
import io
import json
import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import Any

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.stub import ANY, Stubber

from sail_support.cloud import (
    CloudConfig,
    PriceQuote,
    S3Store,
    WorkPlan,
    estimate_cost,
    json_hash,
    submit_batch,
)


class MemoryS3:
    """Protocol fixture; no network or AWS measurements."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.puts = 0

    def put_object(self, **values: Any) -> dict[str, str]:
        assert values["IfNoneMatch"] == "*" and values["ServerSideEncryption"] == "AES256"
        self.puts += 1
        key = values["Key"]
        if key in self.files:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.files[key] = values["Body"]
        return {}

    def get_object(self, **values: Any) -> dict[str, Any]:
        return {"Body": io.BytesIO(self.files[values["Key"]])}


def quote() -> PriceQuote:
    return PriceQuote.model_validate_json(
        (Path(__file__).resolve().parents[3] / "configs/support/price-quote.json").read_text()
    )


def authorised(plan: WorkPlan, budget: float = 10) -> CloudConfig:
    return CloudConfig(
        enabled=True,
        authorised_budget_usd=budget,
        authorisation_reference="mock-test only; no external authorisation",
        expires_utc=quote().checked_utc + timedelta(days=1),
        approved_cost_sha256=json_hash(estimate_cost(plan, quote())),
        account_id="111122223333",
        artifact_bucket="sail-private-test",
        queue_arn="arn:aws:batch:us-east-1:111122223333:job-queue/test",
        job_definition_arn="arn:aws:batch:us-east-1:111122223333:job-definition/test:1",
    )


def spec(plan: WorkPlan, *, seed: int = 610001) -> tuple[bytes, str]:
    data = {
        "purpose": "development_compatibility",
        "max_attempts": plan.attempts,
        "max_wall_s": plan.worker_wall_s,
        "workers": 1,
        "episodes": [
            {"config": {"stage": "development", "root_seed": seed + index}}
            for index in range(plan.tasks)
        ],
    }
    raw = json.dumps(data, separators=(",", ":")).encode()
    return raw, hashlib.sha256(raw).hexdigest()


def test_dated_prices_include_attempts_and_storage_without_a_free_tier_claim() -> None:
    result = estimate_cost(WorkPlan(tasks=4), quote())
    assert result["components_usd"]["fargate_cpu"] == pytest.approx(5760 * 2 * 0.000011244)
    assert result["components_usd"]["fargate_memory"] == pytest.approx(5760 * 8 * 0.000001235)
    assert result["reservation_usd"] == 2.86
    assert result["authorised_spend_usd"] == 0
    assert result["work"]["public_hosting"] is False
    one = estimate_cost(WorkPlan(tasks=4, attempts=1), quote())
    assert one["components_usd"]["fargate_cpu"] == result["components_usd"]["fargate_cpu"] / 2


def test_content_addressed_transfer_round_trips_every_byte_and_rejects_paths(
    tmp_path: Path,
) -> None:
    memory = MemoryS3()
    store = S3Store(memory, "private")
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text('{"status":"succeeded"}\n')
    (source / "nested").mkdir()
    (source / "nested/episodes.parquet").write_bytes(b"contract bytes\x00\x01")
    digest = store.upload_directory(source, "jobs/results/example/bundle.json")
    assert store.upload_directory(source, "jobs/results/example/bundle.json") == digest
    store.download_directory("jobs/results/example/bundle.json", digest, tmp_path / "download")
    for original in source.rglob("*"):
        if original.is_file():
            assert (
                tmp_path / "download" / original.relative_to(source)
            ).read_bytes() == original.read_bytes()
    with pytest.raises(ValueError, match="changed"):
        store.get("jobs/results/example/bundle.json", "0" * 64)
    with pytest.raises(ValueError, match="Unsafe"):
        store.put("../private", b"x")
    malicious = json.dumps(
        {
            "schema_version": "artifact-bundle-v1",
            "files": [
                {
                    "path": "../escape",
                    "key": "artifacts/sha256/" + "a" * 64,
                    "sha256": "a" * 64,
                    "bytes": 1,
                }
            ],
        }
    ).encode()
    bad_hash = store.put("jobs/bad.json", malicious)
    with pytest.raises(ValueError, match="Unsafe"):
        store.download_directory("jobs/bad.json", bad_hash, tmp_path / "unsafe")
    assert not (tmp_path / "unsafe").exists()


def test_real_sdk_validates_immutable_encrypted_put_without_network() -> None:
    client = boto3.client(
        "s3", region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    raw = b"immutable fixture"
    digest = hashlib.sha256(raw).hexdigest()
    with Stubber(client) as stub:
        stub.add_response(
            "put_object",
            {},
            {
                "Bucket": "private",
                "Key": "artifacts/test",
                "Body": raw,
                "Metadata": {"sha256": digest},
                "ServerSideEncryption": "AES256",
                "IfNoneMatch": "*",
            },
        )
        assert S3Store(client, "private").put("artifacts/test", raw) == digest
        stub.assert_no_pending_responses()


@pytest.mark.parametrize(
    "reason", ["disabled", "budget", "expired", "stale", "changed", "final", "hash"]
)
def test_budget_source_and_split_gates_precede_network_access(tmp_path: Path, reason: str) -> None:
    plan = WorkPlan(tasks=2)
    config = authorised(plan, budget=0.01 if reason == "budget" else 10)
    current = quote().checked_utc + timedelta(seconds=1)
    q = quote()
    if reason == "disabled":
        config = CloudConfig()
    elif reason == "expired":
        config = config.model_copy(update={"expires_utc": current})
    elif reason == "stale":
        q = q.model_copy(update={"checked_utc": current - timedelta(days=8)})
    elif reason == "changed":
        config = config.model_copy(update={"approved_cost_sha256": "0" * 64})
    raw, digest = spec(plan, seed=310000 if reason == "final" else 610001)
    if reason == "hash":
        digest = "0" * 64
    memory = MemoryS3()
    with pytest.raises(ValueError):
        submit_batch(
            config,
            plan,
            q,
            job_sha256=digest,
            spec_raw=raw,
            ledger=tmp_path / "ledger.sqlite",
            store=S3Store(memory, config.artifact_bucket),
            client=None,
            now=current,
        )
    assert memory.puts == 0


def test_submit_array_is_bounded_and_duplicate_submission_is_durable(tmp_path: Path) -> None:
    plan = WorkPlan(tasks=2)
    config = authorised(plan)
    raw, digest = spec(plan)
    memory = MemoryS3()
    store = S3Store(memory, config.artifact_bucket)
    client = boto3.client(
        "batch",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    with Stubber(client) as stub:
        # The SDK validates the complete request shape; no AWS call occurs.
        stub.add_response(
            "submit_job",
            {"jobName": f"sail-{digest[:24]}", "jobId": "mock-job-id"},
            {
                "jobName": f"sail-{digest[:24]}",
                "jobQueue": config.queue_arn,
                "jobDefinition": config.job_definition_arn,
                "arrayProperties": {"size": 2},
                "timeout": {"attemptDurationSeconds": 720},
                "retryStrategy": {
                    "attempts": 2,
                    "evaluateOnExit": [
                        {"action": "RETRY", "onExitCode": "75"},
                        {"action": "RETRY", "onStatusReason": "Task failed to start*"},
                        {"action": "EXIT", "onReason": "*"},
                    ],
                },
                "containerOverrides": {
                    "resourceRequirements": [
                        {"type": "VCPU", "value": "2"},
                        {"type": "MEMORY", "value": "8192"},
                    ],
                    "environment": ANY,
                },
            },
        )
        result = submit_batch(
            config,
            plan,
            quote(),
            job_sha256=digest,
            spec_raw=raw,
            ledger=tmp_path / "ledger.sqlite",
            store=store,
            client=client,
            now=quote().checked_utc + timedelta(seconds=1),
        )
        assert result["aws_job_id"] == "mock-job-id"
        again = submit_batch(
            config,
            plan,
            quote(),
            job_sha256=digest,
            spec_raw=raw,
            ledger=tmp_path / "ledger.sqlite",
            store=store,
            client=client,
            now=quote().checked_utc + timedelta(seconds=1),
        )
        assert again["aws_job_id"] == "mock-job-id" and memory.puts == 1
        stub.assert_no_pending_responses()


def test_cumulative_budget_uses_integer_cents_and_blocks_a_second_job(tmp_path: Path) -> None:
    class CaptureClient:
        calls = 0

        def submit_job(self, **kwargs: Any) -> dict[str, str]:
            self.calls += 1
            return {"jobId": "mock-job-id"}

    plan = WorkPlan(tasks=1)
    reserved = estimate_cost(plan, quote())["reservation_usd"]
    config = authorised(plan, budget=reserved)
    store = S3Store(MemoryS3(), config.artifact_bucket)
    client = CaptureClient()
    raw, digest = spec(plan)
    submit_batch(
        config,
        plan,
        quote(),
        job_sha256=digest,
        spec_raw=raw,
        ledger=tmp_path / "ledger.sqlite",
        store=store,
        client=client,
        now=quote().checked_utc + timedelta(seconds=1),
    )
    raw, digest = spec(plan, seed=610002)
    with pytest.raises(ValueError, match="Persistent"):
        submit_batch(
            config,
            plan,
            quote(),
            job_sha256=digest,
            spec_raw=raw,
            ledger=tmp_path / "ledger.sqlite",
            store=store,
            client=client,
            now=quote().checked_utc + timedelta(seconds=1),
        )
    assert client.calls == 1


def test_ambiguous_submission_stays_reserved_and_cannot_repeat(tmp_path: Path) -> None:
    class TimeoutClient:
        calls = 0

        def submit_job(self, **kwargs: Any) -> None:
            self.calls += 1
            raise TimeoutError("Mocked lost response; remote state unknown")

    plan = WorkPlan(tasks=1)
    config = authorised(plan)
    raw, digest = spec(plan)
    client = TimeoutClient()
    store = S3Store(MemoryS3(), config.artifact_bucket)
    with pytest.raises(TimeoutError):
        submit_batch(
            config,
            plan,
            quote(),
            job_sha256=digest,
            spec_raw=raw,
            ledger=tmp_path / "ledger.sqlite",
            store=store,
            client=client,
            now=quote().checked_utc + timedelta(seconds=1),
        )
    result = submit_batch(
        config,
        plan,
        quote(),
        job_sha256=digest,
        spec_raw=raw,
        ledger=tmp_path / "ledger.sqlite",
        store=store,
        client=client,
        now=quote().checked_utc + timedelta(seconds=1),
    )
    assert result["status"] == "submission_unknown" and client.calls == 1
    with sqlite3.connect(tmp_path / "ledger.sqlite") as db:
        assert db.execute("SELECT SUM(usd_cents) FROM reservations").fetchone()[0] > 0
