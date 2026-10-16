"""Private content-addressed artifacts and an opt-in, durably reserved Batch budget."""

import hashlib
import json
import math
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from botocore.exceptions import ClientError
from pydantic import Field, field_validator, model_validator

from sail_support.contracts import Contract, utc


class PriceQuote(Contract):
    region: Literal["us-east-1"] = "us-east-1"
    checked_utc: datetime
    cpu_usd_per_vcpu_second: float = Field(gt=0)
    memory_usd_per_gb_second: float = Field(gt=0)
    ipv4_usd_per_hour: float = Field(gt=0)
    s3_usd_per_gb_month: float = Field(gt=0)
    s3_put_usd_per_request: float = Field(gt=0)
    s3_get_usd_per_request: float = Field(gt=0)
    ecr_usd_per_gb_month: float = Field(gt=0)
    logs_ingest_usd_per_gb: float = Field(gt=0)
    logs_store_usd_per_gb_month: float = Field(gt=0)
    sources: dict[str, str]
    s3_price_list_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("checked_utc")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        return utc(value)


class WorkPlan(Contract):
    tasks: int = Field(ge=1, le=8)
    attempts: int = Field(default=2, ge=1, le=2)
    worker_wall_s: int = Field(default=600, ge=1, le=600)
    startup_stop_allowance_s: int = Field(default=120, ge=120, le=600)
    vcpus_per_task: Literal[2] = 2
    memory_gb_per_task: Literal[8] = 8
    max_vcpus: Literal[4] = 4
    image_gb: float = Field(default=8, gt=0)
    artifacts_gb: float = Field(default=2, gt=0)
    logs_gb: float = Field(default=0.05, gt=0)
    storage_hours: int = Field(default=168, ge=1, le=720)
    put_requests: int = Field(default=2000, ge=1)
    get_requests: int = Field(default=5000, ge=1)
    network_control_reserve_usd: float = Field(default=1, gt=0)
    safety_factor: float = Field(default=2, ge=2)
    public_hosting: Literal[False] = False


def estimate_cost(plan: WorkPlan, quote: PriceQuote) -> dict[str, Any]:
    seconds = plan.tasks * plan.attempts * (plan.worker_wall_s + plan.startup_stop_allowance_s)
    month = plan.storage_hours / 730
    components = {
        "fargate_cpu": seconds * plan.vcpus_per_task * quote.cpu_usd_per_vcpu_second,
        "fargate_memory": seconds * plan.memory_gb_per_task * quote.memory_usd_per_gb_second,
        "public_ipv4": seconds / 3600 * quote.ipv4_usd_per_hour,
        "s3_storage": plan.artifacts_gb * month * quote.s3_usd_per_gb_month,
        "s3_put_list": plan.put_requests * quote.s3_put_usd_per_request,
        "s3_get_head": plan.get_requests * quote.s3_get_usd_per_request,
        "ecr_storage": plan.image_gb * month * quote.ecr_usd_per_gb_month,
        "logs_ingestion": plan.logs_gb * quote.logs_ingest_usd_per_gb,
        "logs_storage": plan.logs_gb * month * quote.logs_store_usd_per_gb_month,
        "network_and_control_allowance": plan.network_control_reserve_usd,
    }
    total = sum(components.values())
    return {
        "currency": "USD",
        "region": quote.region,
        "components_usd": components,
        "estimated_all_attempts_usd": total,
        "reservation_usd": math.ceil(total * plan.safety_factor * 100) / 100,
        "quote": quote.model_dump(mode="json"),
        "work": plan.model_dump(),
        "authorised_spend_usd": 0,
        "status": "planning_only",
        "limits": [
            "Usage allowances and image size are assumptions, not cloud measurements or hard caps",
            "All attempts included; no free-tier credit assumed; tax/currency conversion excluded",
            "Image pull/stop time can exceed the allowance; billing alerts do not cap spend",
            "No NAT gateway, GPU, paid forecast access or public hosting in this plan",
            "Network/control reserve is an explicit allowance, not a measured unit-price charge",
        ],
    }


class CloudConfig(Contract):
    enabled: bool = False
    authorised_budget_usd: float = Field(default=0, ge=0)
    authorisation_reference: str = ""
    expires_utc: datetime | None = None
    approved_cost_sha256: str = ""
    region: Literal["us-east-1"] = "us-east-1"
    account_id: str = Field(default="000000000000", pattern=r"^\d{12}$")
    artifact_bucket: str = "not-configured"
    queue_arn: str = ""
    job_definition_arn: str = ""

    @field_validator("expires_utc")
    @classmethod
    def aware(cls, value: datetime | None) -> datetime | None:
        return utc(value) if value is not None else None

    @model_validator(mode="after")
    def disabled_by_default(self) -> "CloudConfig":
        if self.enabled:
            if (
                self.authorised_budget_usd <= 0
                or not self.authorisation_reference
                or self.expires_utc is None
                or not re.fullmatch(r"[a-f0-9]{64}", self.approved_cost_sha256)
                or self.account_id == "000000000000"
            ):
                raise ValueError(
                    "Cloud execution needs an explicit budget, reference, expiry "
                    "and approved cost hash"
                )
            prefix = f"arn:aws:batch:{self.region}:{self.account_id}:"
            if not self.queue_arn.startswith(prefix + "job-queue/") or not re.fullmatch(
                re.escape(prefix) + r"job-definition/[\w-]+:\d+", self.job_definition_arn
            ):
                raise ValueError(
                    "Queue and versioned definition must belong to the approved region/account"
                )
            if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", self.artifact_bucket):
                raise ValueError("Invalid private artifact bucket")
        return self


def json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def approved_cost(
    config: CloudConfig, plan: WorkPlan, quote: PriceQuote, now: datetime | None = None
) -> dict[str, Any]:
    current = utc(now or datetime.now(UTC))
    if not config.enabled or config.authorised_budget_usd <= 0:
        raise ValueError("Cloud submission disabled; no authorised spending budget")
    if config.expires_utc is None or config.expires_utc <= current:
        raise ValueError("Cloud authorisation expired")
    if not timedelta(0) <= current - quote.checked_utc <= timedelta(days=7):
        raise ValueError("Price quote is stale or future-dated")
    cost = estimate_cost(plan, quote)
    if config.approved_cost_sha256 != json_hash(cost):
        raise ValueError("Work or quote differs from the approved cost plan")
    return cost


class S3Store:
    """SDK client injected for tests; nothing constructs a cloud client during local imports."""

    def __init__(self, client: Any, bucket: str):
        self.client, self.bucket = client, bucket

    def put(self, key: str, raw: bytes) -> str:
        if (
            not re.fullmatch(r"[a-zA-Z0-9/_.-]+", key)
            or ".." in PurePosixPath(key).parts
            or key.startswith("/")
        ):
            raise ValueError("Unsafe artifact key")
        digest = hashlib.sha256(raw).hexdigest()
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=raw,
                Metadata={"sha256": digest},
                ServerSideEncryption="AES256",
                IfNoneMatch="*",
            )
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"PreconditionFailed", "412"}:
                raise
            if self.get(key, digest, len(raw)) != raw:
                raise ValueError("Existing immutable artifact differs") from error
        return digest

    def get(self, key: str, digest: str, max_bytes: int = 100_000_000) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        body = response["Body"]
        try:
            raw: bytes = body.read(max_bytes + 1)
        finally:
            body.close()
        if len(raw) > max_bytes or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Downloaded artifact exceeds its bound or has changed")
        return raw

    def upload_directory(self, source: Path, bundle_key: str) -> str:
        entries = []
        total = 0
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError("Artifact bundles cannot contain symlinks")
            if not path.is_file():
                continue
            raw = path.read_bytes()
            total += len(raw)
            if len(raw) > 100_000_000 or total > 500_000_000:
                raise ValueError("Artifact bundle exceeds its byte budget")
            digest = hashlib.sha256(raw).hexdigest()
            key = f"artifacts/sha256/{digest}"
            self.put(key, raw)
            entries.append(
                {
                    "path": path.relative_to(source).as_posix(),
                    "key": key,
                    "sha256": digest,
                    "bytes": len(raw),
                }
            )
        bundle = json.dumps(
            {"schema_version": "artifact-bundle-v1", "files": entries}, sort_keys=True
        ).encode()
        return self.put(bundle_key, bundle)

    def download_directory(self, key: str, digest: str, output: Path) -> None:
        bundle = json.loads(self.get(key, digest, 5_000_000))
        if bundle.get("schema_version") != "artifact-bundle-v1":
            raise ValueError("Unsupported artifact bundle")
        entries = bundle["files"]
        if len(entries) > 10_000 or sum(entry["bytes"] for entry in entries) > 500_000_000:
            raise ValueError("Bundle size limit exceeded")
        names = set()
        for entry in entries:
            path = PurePosixPath(entry["path"])
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.as_posix() in names
                or not 0 <= entry["bytes"] <= 100_000_000
                or entry["key"] != f"artifacts/sha256/{entry['sha256']}"
            ):
                raise ValueError("Unsafe or duplicate artifact path/key")
            names.add(path.as_posix())
        output.mkdir(parents=True, exist_ok=False)
        for entry in entries:
            raw = self.get(entry["key"], entry["sha256"], entry["bytes"])
            if len(raw) != entry["bytes"]:
                raise ValueError("Artifact length differs from its manifest")
            target = output / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)


def submit_batch(
    config: CloudConfig,
    plan: WorkPlan,
    quote: PriceQuote,
    *,
    job_sha256: str,
    spec_raw: bytes,
    ledger: Path,
    store: S3Store,
    client: Any,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Reserve before network access; ambiguous submissions never automatically repeat."""
    cost = approved_cost(config, plan, quote, now)
    if not re.fullmatch(r"[a-f0-9]{64}", job_sha256):
        raise ValueError("Invalid job hash")
    data = json.loads(spec_raw)
    # Full JobSpec/source validation is done by the frozen numerical CLI before submission.
    if (
        data.get("purpose") != "development_compatibility"
        or len(data.get("episodes", [])) != plan.tasks
        or data.get("max_attempts") != plan.attempts
        or data.get("max_wall_s") != plan.worker_wall_s
        or data.get("workers") != 1
    ):
        raise ValueError("Job workload differs from the costed plan")
    if any(
        episode.get("config", {}).get("stage") != "development"
        or not 600000 <= episode.get("config", {}).get("root_seed", -1) < 700000
        for episode in data["episodes"]
    ):
        raise ValueError("Cloud jobs cannot use final/locked data")
    if (
        hashlib.sha256(
            json.dumps(data, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        ).hexdigest()
        != job_sha256
    ):
        raise ValueError("Job hash does not match the supplied specification")
    if store.bucket != config.artifact_bucket:
        raise ValueError("Artifact storage differs from approved infrastructure")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(ledger, timeout=10) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS reservations (job TEXT PRIMARY KEY, usd_cents INTEGER, "
            "status TEXT, aws_job_id TEXT, cost_sha256 TEXT)"
        )
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute("SELECT * FROM reservations WHERE job=?", (job_sha256,)).fetchone()
        if existing:
            return {
                "job_sha256": existing[0],
                "reservation_usd": existing[1] / 100,
                "status": existing[2],
                "aws_job_id": existing[3],
            }
        used = db.execute("SELECT COALESCE(SUM(usd_cents), 0) FROM reservations").fetchone()[0]
        reservation_cents = int(Decimal(str(cost["reservation_usd"])) * 100)
        budget_cents = int(Decimal(str(config.authorised_budget_usd)) * 100)
        if used + reservation_cents > budget_cents:
            raise ValueError("Persistent reservation ledger would exceed the authorised budget")
        db.execute(
            "INSERT INTO reservations VALUES (?, ?, 'reserved', NULL, ?)",
            (job_sha256, reservation_cents, json_hash(cost)),
        )
    try:
        spec_hash = hashlib.sha256(spec_raw).hexdigest()
        key = f"jobs/specs/{spec_hash}.json"
        store.put(key, spec_raw)
        request: dict[str, Any] = {
            "jobName": f"sail-{job_sha256[:24]}",
            "jobQueue": config.queue_arn,
            "jobDefinition": config.job_definition_arn,
            "timeout": {"attemptDurationSeconds": plan.worker_wall_s + 120},
            "retryStrategy": {
                "attempts": plan.attempts,
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
                "environment": [
                    {"name": name, "value": value}
                    for name, value in {
                        "SAIL_ARTIFACT_BUCKET": config.artifact_bucket,
                        "SAIL_SPEC_KEY": key,
                        "SAIL_SPEC_SHA256": spec_hash,
                        "SAIL_JOB_SHA256": job_sha256,
                    }.items()
                ],
            },
        }
        if plan.tasks > 1:
            request["arrayProperties"] = {"size": plan.tasks}
        response = client.submit_job(**request)
    except Exception:
        with sqlite3.connect(ledger) as db:
            db.execute(
                "UPDATE reservations SET status='submission_unknown' WHERE job=?", (job_sha256,)
            )
        raise
    with sqlite3.connect(ledger) as db:
        db.execute(
            "UPDATE reservations SET status='submitted', aws_job_id=? WHERE job=?",
            (response["jobId"], job_sha256),
        )
    return {
        "job_sha256": job_sha256,
        "status": "submitted",
        "aws_job_id": response["jobId"],
        "reservation_usd": cost["reservation_usd"],
    }
