"""Typed, immutable publication of a complete independent planner replication."""

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq
from pydantic import Field, model_validator

from sail_lab.config import Contract
from sail_lab.experiments import file_hash, write_json
from sail_lab.locked_evaluation import verify_manifest
from scripts.planner_replication import SLICES, VARIANTS, validate_rows, verify

Variant = Literal["B1_belief", "B2_mean", "B2_budget", "B3_mean", "B3_risk"]
Reference = Literal["B1_belief", "B2_mean", "B2_budget", "B3_mean"]
SliceName = Literal[
    "nominal", "sensor_delay", "delay_tack", "oscillating", "ou", "regime", "spatial"
]


class Contrast(Contract):
    reference: Reference
    mean_difference_s: float
    mean_difference_95: tuple[float, float]
    cvar95_difference_s: float
    cvar95_difference_95: tuple[float, float]

    @model_validator(mode="after")
    def ordered(self) -> "Contrast":
        if any(
            bounds[0] > bounds[1] for bounds in (self.mean_difference_95, self.cvar95_difference_95)
        ):
            raise ValueError("reversed confidence interval")
        return self


class PolicyResult(Contract):
    name: Variant
    worlds: int = Field(ge=1)
    finished: int = Field(ge=0)
    deadline_exceeded: int = Field(ge=0)
    boundary_violation: int = Field(ge=0)
    mean_cost_s: float = Field(ge=0)
    cvar95_s: float = Field(ge=0)
    decision_latency_mean_s: float = Field(ge=0)
    decision_latency_p95_s: float = Field(ge=0)
    model_steps: int = Field(ge=0)
    model_step_calls_including_splits: int = Field(ge=0)
    allocated_model_step_ceilings: int = Field(ge=0)
    model_rollouts: int = Field(ge=0)
    zero_failure_upper95: float | None = Field(ge=0, le=1)

    @model_validator(mode="after")
    def counts(self) -> "PolicyResult":
        if self.finished + self.deadline_exceeded + self.boundary_violation != self.worlds:
            raise ValueError("policy denominator excludes outcomes")
        if (
            not self.model_steps <= self.allocated_model_step_ceilings
            or self.model_step_calls_including_splits < self.model_steps
        ):
            raise ValueError("invalid actual/allocated work")
        if self.finished == self.worlds:
            if self.zero_failure_upper95 is None or not math.isclose(
                self.zero_failure_upper95, 1 - 0.05 ** (1 / self.worlds), abs_tol=1e-12
            ):
                raise ValueError("missing zero-failure uncertainty")
        elif self.zero_failure_upper95 is not None:
            raise ValueError("zero-failure bound applied to failures")
        return self


class CohortResult(Contract):
    name: SliceName
    worlds: int = Field(ge=1)
    tail_effective_worlds: float = Field(gt=0)
    policies: tuple[PolicyResult, ...]
    contrasts: tuple[Contrast, ...]

    @model_validator(mode="after")
    def complete(self) -> "CohortResult":
        if len(self.policies) != 5 or {p.name for p in self.policies} != set(VARIANTS):
            raise ValueError("missing controller")
        if any(p.worlds != self.worlds for p in self.policies):
            raise ValueError("unpaired controller denominators")
        if len(self.contrasts) != 4 or {c.reference for c in self.contrasts} != set(VARIANTS[:-1]):
            raise ValueError("missing registered contrast")
        if not math.isclose(self.tail_effective_worlds, 0.05 * self.worlds, abs_tol=1e-10):
            raise ValueError("incorrect tail sample")
        return self


class ReplicationPublication(Contract):
    schema_version: Literal[1] = 1
    evidence_level: Literal["independent_locked_synthetic_replication"] = (
        "independent_locked_synthetic_replication"
    )
    lock_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    original_lock_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    numerical_source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    method_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    episodes_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    episodes_url: Literal["/replication/episodes.parquet"] = "/replication/episodes.parquet"
    report_url: Literal["/replication/report.json"] = "/replication/report.json"
    total_worlds: int = Field(ge=1)
    retained_episodes: int = Field(ge=1)
    selected_deterministic_horizon_s: Literal[60, 120, 240, 480]
    evaluation_wall_s: float = Field(gt=0)
    workers: int = Field(ge=1, le=4)
    bootstrap_count: int = Field(ge=200)
    analysis_seed: Literal[885300] = 885300
    precision_target_met: bool = Field(
        description=(
            "Nominal sample-count target only: at least 1024 worlds. "
            "Does not assert narrow intervals or close the scientific precision gate."
        )
    )
    cohorts: tuple[CohortResult, ...]
    nominal_families: tuple[CohortResult, ...]
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def denominators(self) -> "ReplicationPublication":
        if len(self.cohorts) != 3 or {c.name for c in self.cohorts} != set(SLICES):
            raise ValueError("missing stress cohort")
        if (
            sum(c.worlds for c in self.cohorts) != self.total_worlds
            or self.retained_episodes != 5 * self.total_worlds
        ):
            raise ValueError("incomplete episode denominators")
        nominal = next(c for c in self.cohorts if c.name == "nominal")
        if len(self.nominal_families) != 4 or {c.name for c in self.nominal_families} != {
            "oscillating",
            "ou",
            "regime",
            "spatial",
        }:
            raise ValueError("missing nominal family")
        if sum(c.worlds for c in self.nominal_families) != nominal.worlds:
            raise ValueError("nominal family denominator mismatch")
        if self.precision_target_met != (nominal.worlds >= 1024):
            raise ValueError("unearned nominal precision target")
        return self


def cohort(name: str, value: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "worlds": value["worlds"],
        "tail_effective_worlds": value["tail_effective_worlds"],
        "policies": [
            {
                "name": p,
                "worlds": v["worlds"],
                **v["failure_counts"],
                **{
                    k: v[k]
                    for k in (
                        "mean_cost_s",
                        "cvar95_s",
                        "decision_latency_mean_s",
                        "decision_latency_p95_s",
                        "model_steps",
                        "model_step_calls_including_splits",
                        "allocated_model_step_ceilings",
                        "model_rollouts",
                        "zero_failure_upper95",
                    )
                },
            }
            for p, v in value["policies"].items()
        ],
        "contrasts": [
            {
                "reference": ref,
                **{
                    k: v[k]
                    for k in (
                        "mean_difference_s",
                        "mean_difference_95",
                        "cvar95_difference_s",
                        "cvar95_difference_95",
                    )
                },
            }
            for ref, v in value["B3_risk_minus"].items()
        ],
    }


def publish(run: Path, output: Path, public: Path) -> ReplicationPublication:
    lock, plan, registered = verify(run)
    verify_manifest(run)
    evaluation = verify_manifest(run / "evaluation")
    rows = pq.read_table(run / "evaluation/episodes.parquet").to_pylist()
    validate_rows(rows, registered, lock["policies"])
    data = json.loads((run / "report.json").read_text())
    if (
        data["stage"] != "independent_locked_replication"
        or data["lock_sha256"] != file_hash(run / "lock.json")
        or data["episodes"] != len(rows)
    ):
        raise ValueError("report identity/count mismatch")
    value = ReplicationPublication.model_validate(
        {
            "lock_sha256": file_hash(run / "lock.json"),
            "original_lock_sha256": lock["parent_lock_sha256"],
            "numerical_source_sha256": lock["numerical_source_sha256"],
            "method_sha256": lock["method_hashes"]["scripts/planner_replication.py"],
            "report_sha256": file_hash(run / "report.json"),
            "episodes_sha256": file_hash(run / "evaluation/episodes.parquet"),
            "total_worlds": len(registered),
            "retained_episodes": len(rows),
            "selected_deterministic_horizon_s": lock["policies"]["B2_budget"]["planning_horizon_s"],
            "evaluation_wall_s": evaluation["wall_s"],
            "workers": plan.workers,
            "bootstrap_count": plan.bootstrap_count,
            "precision_target_met": data["precision_target_met"],
            "cohorts": [cohort(name, v) for name, v in data["slices"].items()],
            "nominal_families": [cohort(name, v) for name, v in data["nominal_families"].items()],
            "limitations": data["limitations"],
        }
    )
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(run / "report.json", output / "report.json")
    shutil.copyfile(run / "evaluation/episodes.parquet", output / "episodes.parquet")
    for name in ("scenarios.json", "plan.json", "source-snapshot.json"):
        shutil.copyfile(run / name, output / name)
    write_json(
        output / "registration.json",
        {
            key: entry
            for key, entry in lock.items()
            if key not in ("parent_path", "validation_path")
        },
    )
    plot_comparisons(value, output)
    write_json(output / "catalogue.json", value.model_dump(mode="json"))
    write_json(
        output / "manifest.json",
        {
            "status": "succeeded",
            "source_manifest_sha256": file_hash(run / "manifest.json"),
            "artifacts": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()},
        },
    )
    destination = public / "replication"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(output, destination)
    write_json(public / "replication.json", value.model_dump(mode="json"))
    return value


def plot_comparisons(value: ReplicationPublication, output: Path) -> None:
    """Export research figures directly from validated paired statistics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "B1_belief": "Tuned B1 belief",
        "B2_mean": "Original B2 (60 s)",
        "B2_budget": f"Allowance-selected B2 ({value.selected_deterministic_horizon_s} s)",
        "B3_mean": "B3 expected cost",
    }
    for group in value.cohorts:
        figure, axes = plt.subplots(1, 2, figsize=(11, 3.5))
        for axis, endpoint in zip(axes, ("mean", "cvar95"), strict=True):
            for position, contrast in enumerate(group.contrasts):
                point = getattr(contrast, f"{endpoint}_difference_s")
                low, high = getattr(contrast, f"{endpoint}_difference_95")
                axis.hlines(position, low, high, color="#256072", linewidth=2)
                axis.scatter([point], [position], color="#173f4b", s=28, zorder=3)
            axis.axvline(0, color="#666", linestyle="--", linewidth=1)
            axis.set_yticks(range(4), [labels[c.reference] for c in group.contrasts])
            axis.invert_yaxis()
            axis.set_xlabel("B3 risk minus reference cost (s)")
            axis.set_title("Mean" if endpoint == "mean" else "CVaR95")
            axis.grid(axis="x", alpha=0.2)
        figure.suptitle(
            f"{group.name}: {group.worlds} paired worlds, "
            f"{group.tail_effective_worlds:.1f} effective tail worlds"
        )
        figure.text(
            0.5,
            0.01,
            "Pointwise 95% stratified paired bootstrap; failures retained. "
            "Synthetic existing families; equal work ceilings do not mean equal runtime.",
            ha="center",
            fontsize=8,
        )
        figure.tight_layout(rect=(0, 0.06, 1, 0.94))
        for suffix in ("svg", "png", "pdf"):
            figure.savefig(output / f"{group.name}.{suffix}", dpi=160)
        plt.close(figure)


def export_schema(project: Path) -> None:
    schema: dict[str, Any] = ReplicationPublication.model_json_schema()
    schema["$schema"] = "http://json-schema.org/draft-07/schema#"

    def lower(value: Any) -> None:
        if isinstance(value, dict):
            if "prefixItems" in value:
                value["items"] = value.pop("prefixItems")
            for child in value.values():
                lower(child)
        elif isinstance(value, list):
            for child in value:
                lower(child)

    lower(schema)
    for path in (
        project / "schemas/replication.schema.json",
        project / "web/replication.schema.json",
    ):
        write_json(path, schema)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/exports/m5-replication"))
    parser.add_argument("--public", type=Path, default=Path("web/public"))
    args = parser.parse_args()
    export_schema(Path.cwd())
    if args.run:
        print(publish(args.run, args.output, args.public).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
