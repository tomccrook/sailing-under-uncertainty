"""Verified static publication of measured-work studies and recorded components."""

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
from scripts.compute_followup import VARIANTS, validate_rows, verify

Variant = Literal[
    "B1_belief", "B2_matched", "B3_mean_matched", "B3_risk_matched", "B3_risk_original"
]


class Difference(Contract):
    reference: Variant
    mean_difference_s: float
    mean_difference_95: tuple[float, float]
    cvar95_difference_s: float
    cvar95_difference_95: tuple[float, float]

    @model_validator(mode="after")
    def intervals(self) -> "Difference":
        if (
            self.mean_difference_95[0] > self.mean_difference_95[1]
            or self.cvar95_difference_95[0] > self.cvar95_difference_95[1]
        ):
            raise ValueError("reversed interval")
        return self


class PolicySummary(Contract):
    name: Variant
    finished: int = Field(ge=0)
    boundary_violation: int = Field(ge=0)
    deadline_exceeded: int = Field(ge=0)
    mean_cost_s: float = Field(ge=0)
    cvar95_s: float = Field(ge=0)
    steps: int = Field(ge=0)
    steps_per_decision: float = Field(ge=0)
    decision_latency_p95_s: float = Field(ge=0)
    latency_mean_s: float = Field(ge=0)
    recursive_step_calls: int = Field(ge=0)
    zero_failure_upper95: float | None = Field(default=None, ge=0, le=1)


class WorkCohort(Contract):
    name: Literal[
        "nominal",
        "new_family",
        "oscillating",
        "ou",
        "regime",
        "spatial",
        "gust_front",
        "persistent_trend",
    ]
    worlds: int = Field(ge=1)
    tail_effective_worlds: float = Field(gt=0)
    work_matching_passed: bool
    matching_tolerance: float = Field(gt=0, le=0.15)
    policies: tuple[PolicySummary, ...]
    contrasts: tuple[Difference, ...]

    @model_validator(mode="after")
    def complete(self) -> "WorkCohort":
        if len(self.policies) != 5 or {p.name for p in self.policies} != set(VARIANTS):
            raise ValueError("missing/duplicated policy")
        for p in self.policies:
            if p.finished + p.boundary_violation + p.deadline_exceeded != self.worlds:
                raise ValueError("excluded outcomes")
            if p.recursive_step_calls < p.steps:
                raise ValueError("invalid work accounting")
            if p.finished == self.worlds:
                if p.zero_failure_upper95 is None or not math.isclose(
                    p.zero_failure_upper95, 1 - 0.05 ** (1 / self.worlds), abs_tol=1e-12
                ):
                    raise ValueError("invalid zero-failure bound")
            elif p.zero_failure_upper95 is not None:
                raise ValueError("zero bound with failures")
        policies = {p.name: p for p in self.policies}
        denom = policies["B2_matched"].steps_per_decision
        matched = denom > 0 and all(
            abs(policies[v].steps_per_decision / denom - 1) <= self.matching_tolerance
            for v in ("B3_mean_matched", "B3_risk_matched")
        )
        if matched != self.work_matching_passed:
            raise ValueError("unsupported matching claim")
        if len(self.contrasts) != 4 or {d.reference for d in self.contrasts} != set(VARIANTS) - {
            "B3_risk_matched"
        }:
            raise ValueError("missing contrast")
        if not math.isclose(self.tail_effective_worlds, 0.05 * self.worlds, abs_tol=1e-10):
            raise ValueError("tail count mismatch")
        return self


class ComputePublication(Contract):
    version: Literal["measured-work-publication-v1"] = "measured-work-publication-v1"
    evidence_level: Literal["locked_synthetic_measured_work_and_new_processes"] = (
        "locked_synthetic_measured_work_and_new_processes"
    )
    lock_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    episodes_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    worlds: int = Field(ge=1)
    episodes: int = Field(ge=1)
    horizon_s: Literal[110, 120, 130, 140, 150]
    matched_samples: Literal[2] = 2
    wall_s: float = Field(gt=0)
    cohorts: tuple[WorkCohort, ...]
    families: tuple[WorkCohort, ...]
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def complete(self) -> "ComputePublication":
        if (
            len(self.cohorts) != 2
            or {g.name for g in self.cohorts} != {"nominal", "new_family"}
            or sum(g.worlds for g in self.cohorts) != self.worlds
            or self.episodes != 5 * self.worlds
        ):
            raise ValueError("incomplete cohort table")
        if len(self.families) != 6 or {g.name for g in self.families} != {
            "oscillating",
            "ou",
            "regime",
            "spatial",
            "gust_front",
            "persistent_trend",
        }:
            raise ValueError("missing families")
        for c in self.cohorts:
            names = (
                {"gust_front", "persistent_trend"}
                if c.name == "new_family"
                else {"oscillating", "ou", "regime", "spatial"}
            )
            if sum(g.worlds for g in self.families if g.name in names) != c.worlds:
                raise ValueError("family denominator mismatch")
        return self


class EmpiricalPredictor(Contract):
    name: Literal["AR1", "persistence", "climatology"]
    heldout_pairs: int = Field(ge=1)
    vector_rmse_m_s: float = Field(ge=0)
    speed_rmse_m_s: float = Field(ge=0)
    direction_rmse_deg_at_speed_ge_0_5: float | None = Field(default=None, ge=0)
    coverage95: float = Field(ge=0, le=1)
    gaussian_nll_mean: float


class EmpiricalPublication(Contract):
    version: Literal["recorded-publication-v1"] = "recorded-publication-v1"
    evidence_level: Literal["recorded_GPS_and_buoy_component_validation"] = (
        "recorded_GPS_and_buoy_component_validation"
    )
    registration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    raw_gps_rows: int = Field(ge=1)
    excluded_gps_rows: int = Field(ge=0)
    accepted_gps_rows: int = Field(ge=0)
    rejected_gps_rows: int = Field(ge=0)
    registered_train_identifiers: int = Field(ge=1)
    registered_heldout_identifiers: int = Field(ge=1)
    train_voyages: int = Field(ge=1)
    heldout_voyages: int = Field(ge=1)
    projection_heldout_pairs: int = Field(ge=1)
    projection_heldout_p99_relative: float = Field(ge=0)
    projection_gate_passed: bool
    wind_train_days: int = Field(ge=1)
    wind_heldout_days: int = Field(ge=1)
    wind_heldout_pairs: int = Field(ge=1)
    predictors: tuple[EmpiricalPredictor, ...]
    vector_rmse_difference_m_s: float
    vector_rmse_difference95: tuple[float, float]
    attribution: dict[str, str]
    limitations: tuple[str, ...]
    boat_calibrated: Literal[False] = False

    @model_validator(mode="after")
    def complete(self) -> "EmpiricalPublication":
        if (
            self.train_voyages > self.registered_train_identifiers
            or self.heldout_voyages > self.registered_heldout_identifiers
        ):
            raise ValueError("unregistered voyage identifier")
        if (
            self.raw_gps_rows
            != self.excluded_gps_rows + self.accepted_gps_rows + self.rejected_gps_rows
        ):
            raise ValueError("missing GPS rows")
        if (
            len(self.predictors) != 3
            or {p.name for p in self.predictors} != {"AR1", "persistence", "climatology"}
            or any(p.heldout_pairs != self.wind_heldout_pairs for p in self.predictors)
        ):
            raise ValueError("unequal wind targets")
        if self.vector_rmse_difference95[0] > self.vector_rmse_difference95[1]:
            raise ValueError("reversed empirical interval")
        if self.projection_gate_passed != (self.projection_heldout_p99_relative <= 0.01):
            raise ValueError("invalid projection gate")
        scores = {p.name: p.vector_rmse_m_s for p in self.predictors}
        if not math.isclose(
            scores["AR1"] - scores["persistence"], self.vector_rmse_difference_m_s, abs_tol=1e-12
        ):
            raise ValueError("incorrect observed comparison")
        return self


def work_cohort(name: str, data: dict[str, Any]) -> dict[str, Any]:
    keys = tuple(PolicySummary.model_fields)
    return {
        "name": name,
        **{
            k: data[k]
            for k in (
                "worlds",
                "tail_effective_worlds",
                "work_matching_passed",
                "matching_tolerance",
            )
        },
        "policies": [
            {
                "name": p,
                **v["failure_counts"],
                **{k: v[k] for k in keys if k != "name" and k not in v["failure_counts"]},
            }
            for p, v in data["policies"].items()
        ],
        "contrasts": [
            {"reference": p, **{k: v[k] for k in Difference.model_fields if k != "reference"}}
            for p, v in data["contrasts"]["B3_risk_matched"].items()
        ],
    }


def finish(output: Path, public: Path, name: str, value: Contract) -> None:
    write_json(output / "catalogue.json", value.model_dump(mode="json"))
    write_json(
        output / "manifest.json",
        {
            "status": "succeeded",
            "publisher_sha256": file_hash(Path(__file__)),
            "artifacts": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()},
        },
    )
    public.mkdir(parents=True, exist_ok=True)
    destination = public / name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(output, destination)
    write_json(public / (name + ".json"), value.model_dump(mode="json"))


def plots(value: ComputePublication, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "B1_belief": "Tuned B1",
        "B2_matched": f"B2 ({value.horizon_s} s work target)",
        "B3_mean_matched": "B3 expected cost (2 paths)",
        "B3_risk_original": "Original B3 risk (8 paths; unmatched)",
    }
    for c in value.cohorts:
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
        for ax, metric in zip(axes, ("mean", "cvar95"), strict=True):
            for i, d in enumerate(c.contrasts):
                low, high = getattr(d, f"{metric}_difference_95")
                point = getattr(d, f"{metric}_difference_s")
                ax.hlines(i, low, high, color="#286173", lw=2)
                ax.scatter(point, i, color="#173f4b", s=25)
            ax.set_yticks(range(4), [labels[d.reference] for d in c.contrasts])
            ax.invert_yaxis()
            ax.axvline(0, ls="--", color="#777")
            ax.grid(axis="x", alpha=0.2)
            ax.set_title("Mean" if metric == "mean" else "CVaR95")
            ax.set_xlabel("Two-path B3 risk minus reference cost (s)")
        fig.suptitle(
            f"{c.name}: {c.worlds} paired worlds; {c.tail_effective_worlds:.1f} tail mass; "
            f"measured-work gate {'passes' if c.work_matching_passed else 'fails'}"
        )
        fig.text(
            0.5,
            0.01,
            "Pointwise 95% stratified paired bootstrap. All failures retained. "
            "Work tolerance is per-decision cohort average, not equal runtime.",
            ha="center",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.94))
        for ext in ("svg", "pdf", "png"):
            fig.savefig(output / f"{c.name}.{ext}", dpi=160)
        plt.close(fig)


def publish_compute(run: Path, output: Path, public: Path) -> ComputePublication:
    lock, plan, registered = verify(run)
    verify_manifest(run)
    manifest = verify_manifest(run / "evaluation")
    rows = pq.read_table(run / "evaluation/episodes.parquet").to_pylist()
    validate_rows(rows, registered, lock["policies"])
    report = json.loads((run / "report.json").read_text())
    if report["episodes"] != len(rows) or report["lock_sha256"] != file_hash(run / "lock.json"):
        raise ValueError("report identity mismatch")
    value = ComputePublication.model_validate(
        {
            "lock_sha256": file_hash(run / "lock.json"),
            "report_sha256": file_hash(run / "report.json"),
            "episodes_sha256": file_hash(run / "evaluation/episodes.parquet"),
            "worlds": len(registered),
            "episodes": len(rows),
            "horizon_s": int(lock["policies"]["B2_matched"]["planning_horizon_s"]),
            "wall_s": manifest["wall_s"],
            "cohorts": [work_cohort(n, v) for n, v in report["cohorts"].items()],
            "families": [work_cohort(n, v) for n, v in report["families"].items()],
            "limitations": report["limitations"],
        }
    )
    output.mkdir(parents=True, exist_ok=False)
    for name in ("report.json", "plan.json", "scenarios.json", "source-snapshot.json"):
        shutil.copyfile(run / name, output / name)
    shutil.copyfile(run / "evaluation/episodes.parquet", output / "episodes.parquet")
    write_json(
        output / "registration.json",
        {k: v for k, v in lock.items() if k not in ("parent_path", "validation_path")},
    )
    plots(value, output)
    finish(output, public, "compute-followup", value)
    return value


def publish_empirical(run: Path, output: Path, public: Path) -> EmpiricalPublication:
    verify_manifest(run)
    registration = json.loads((run / "registration.json").read_text())
    if file_hash(run / "registration.json") != (run / "registration.sha256").read_text().strip():
        raise ValueError("empirical registration changed")
    for name, digest in registration["methods"].items():
        if file_hash(Path(name)) != digest:
            raise ValueError("empirical method changed")
    for name, digest in registration["sources"].items():
        if file_hash(Path(registration["source_path"]) / name) != digest:
            raise ValueError("recorded source changed")
    report = json.loads((run / "report.json").read_text())
    g, w = report["gps"], report["wind"]
    quality = json.loads((run / "evaluation/gps-quality-by-day.json").read_text())
    primary_sessions = {r["session"] for r in quality}
    if report["registration_sha256"] != file_hash(run / "registration.json"):
        raise ValueError("empirical report mismatch")
    value = EmpiricalPublication.model_validate(
        {
            "registration_sha256": file_hash(run / "registration.json"),
            "report_sha256": file_hash(run / "report.json"),
            "raw_gps_rows": g["raw_records"],
            "excluded_gps_rows": g["provenance_excluded_rows"],
            "accepted_gps_rows": g["accepted_rows"],
            "rejected_gps_rows": g["rejected_rows"],
            "registered_train_identifiers": sum(v == "train" for v in g["session_splits"].values()),
            "registered_heldout_identifiers": sum(
                v == "heldout" for v in g["session_splits"].values()
            ),
            "train_voyages": sum(g["session_splits"][s] == "train" for s in primary_sessions),
            "heldout_voyages": sum(g["session_splits"][s] == "heldout" for s in primary_sessions),
            "projection_heldout_pairs": g["projection_relative_speed_error"]["heldout"]["pairs"],
            "projection_heldout_p99_relative": g["projection_relative_speed_error"]["heldout"][
                "p99"
            ],
            "projection_gate_passed": g["projection_gate_passed"],
            "wind_train_days": w["train_days"],
            "wind_heldout_days": w["heldout_days"],
            "wind_heldout_pairs": w["heldout_pairs"],
            "predictors": [
                {"name": n, **{k: v[k] for k in EmpiricalPredictor.model_fields if k != "name"}}
                for n, v in w["predictors"].items()
            ],
            "vector_rmse_difference_m_s": w["AR1_minus_persistence_vector_rmse_m_s"],
            "vector_rmse_difference95": w["difference95"],
            "attribution": report["attribution"],
            "limitations": g["limitations"] + w["limitations"] + [report["boat_validation_status"]],
        }
    )
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(run / "report.json", output / "report.json")
    write_json(
        output / "registration.json", {k: v for k, v in registration.items() if k != "source_path"}
    )
    for name in (
        "gps-processed-points.parquet",
        "gps-quality-by-day.json",
        "wind-heldout-predictions.parquet",
        "wind-fit.json",
    ):
        shutil.copyfile(run / "evaluation" / name, output / name)
    (output / "LICENSE-DATA.md").write_text(
        "# Recorded data attribution and licence\n\n"
        + report["attribution"]["sailing"]
        + "\n\nGPS adaptations (Parquet, observed-track plots and GPS summary) are "
        "licensed CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/ . "
        "No diary text or crew names is exported.\n\n"
        + report["attribution"]["wind"]
        + " Public US government observations; preserve NOAA attribution.\n"
    )
    empirical_plots(value, run, output, registration)
    finish(output, public, "empirical", value)
    return value


def empirical_plots(
    value: EmpiricalPublication, run: Path, output: Path, registration: dict[str, Any]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    names = [p.name for p in value.predictors]
    axes[0].bar(names, [p.vector_rmse_m_s for p in value.predictors], color="#286173")
    axes[0].set_ylabel("Vector RMSE (m/s)")
    axes[1].bar(names, [100 * p.coverage95 for p in value.predictors], color="#286173")
    axes[1].axhline(95, ls="--", color="#777")
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Nominal 95% ellipse coverage (%)")
    fig.suptitle(
        f"NOAA 44013: {value.wind_heldout_days} held-out days, "
        f"{value.wind_heldout_pairs} transitions"
    )
    fig.text(
        0.5,
        0.01,
        "Point estimates; same buoy supplies observed targets. "
        "Retrospective observations, not available forecasts or instantaneous truth.",
        ha="center",
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    for ext in ("svg", "pdf", "png"):
        fig.savefig(output / f"wind-validation.{ext}", dpi=160)
    plt.close(fig)
    frame = pq.read_table(run / "evaluation/gps-processed-points.parquet").to_pylist()
    heldout = [s for s, v in registration["gps_sessions"].items() if v == "heldout"]
    session = max(heldout, key=lambda s: registration["gps_session_start_dates"][s])
    day = min(r["source_day"] for r in frame if r["session"] == session)
    track_path = run / "evaluation/tracks" / f"{session}-{day}-0" / "track.json"
    selected = [r for r in json.loads(track_path.read_text())["points"] if r["accepted"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in selected:
        groups.setdefault(row["segment"], []).append(row)
    for segment in groups.values():
        ax.plot(
            [r["east_m"] for r in segment],
            [r["north_m"] for r in segment],
            color="#286173",
            lw=1,
            marker=".",
            markersize=2,
        )
    ax.set_aspect("equal")
    ax.set_xlabel("Local east (m)")
    ax.set_ylabel("Local north (m)")
    ax.set_title(
        f"Observed GPS: voyage {session}, {day}\nLatest held-out voyage, first recorded day"
    )
    fig.text(
        0.5,
        0.01,
        "Greer Jarrett / Zenodo 17340505 / CC BY-SA 4.0. "
        "Gaps are disconnected; clock assumed CET. No counterfactual route.",
        ha="center",
        fontsize=6,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("svg", "pdf", "png"):
        fig.savefig(output / f"observed-track.{ext}", dpi=160)
    plt.close(fig)
    write_json(
        output / "track-selection.json",
        {
            "rule": (
                "latest held-out voyage by start date, first recorded day/chunk; "
                "post-evaluation diagnostic, not efficacy selection"
            ),
            "session": session,
            "source_day": day,
            "rows": len(selected),
            "segments": len(groups),
            "source_sha256": file_hash(run / "evaluation/gps-processed-points.parquet"),
            "track_sha256": file_hash(track_path),
        },
    )


def export_schemas() -> None:
    for name, model in (
        ("compute-followup", ComputePublication),
        ("empirical", EmpiricalPublication),
    ):
        schema: dict[str, Any] = model.model_json_schema()
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
        for directory in ("schemas", "web"):
            write_json(Path(directory) / (name + ".schema.json"), schema)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--empirical", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public", type=Path, default=Path("web/public"))
    args = parser.parse_args()
    export_schemas()
    if args.study:
        print(
            publish_compute(args.study, args.output / "compute", args.public).model_dump_json(
                indent=2
            )
        )
    if args.empirical:
        print(
            publish_empirical(
                args.empirical, args.output / "empirical", args.public
            ).model_dump_json(indent=2)
        )


if __name__ == "__main__":
    main()
