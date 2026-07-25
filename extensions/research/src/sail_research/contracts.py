"""Shared strict contracts for MCP, adapters and labelled investigation scoring."""

import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")]
Issue = Literal[
    "sensor_mismatch",
    "polar_mismatch",
    "unequal_compute",
    "hidden_state_leakage",
    "omitted_failures",
    "unsupported_claim",
    "unpaired_worlds",
    "invalid_cost",
]
Variation = Literal["baseline", "delay", "tack", "delay_tack"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class EvidenceRef(Contract):
    uri: str = Field(pattern=r"^evidence://[a-f0-9]{64}$")
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    description: str


class Controller(Contract):
    label: Identifier
    information_sha256: str
    polar_sha256: str
    compute_units: float = Field(ge=0)
    observation_fields: tuple[str, ...]


class Outcome(Contract):
    root_seed: int = Field(ge=0)
    label: Identifier
    world_sha256: str
    status: Literal["finished", "boundary_violation", "deadline_exceeded"]
    elapsed_s: float = Field(ge=0)
    cost_s: float = Field(ge=0)


class Experiment(Contract):
    experiment_id: Identifier
    stage: Literal["development", "locked_test"]
    evidence_level: Literal["recorded_simulation", "synthetic_audit_fixture"]
    controllers: tuple[Controller, ...] = Field(min_length=2, max_length=4)
    expected_roots: tuple[int, ...] = Field(min_length=1, max_length=1024)
    outcomes: tuple[Outcome, ...] = Field(max_length=4096)
    failure_cost_s: float = Field(default=2400, gt=0)
    proposed_claim: Literal["exploratory", "universal_improvement", "equal_computation"] = (
        "exploratory"
    )
    notes: str = Field(default="", max_length=2000)
    provenance: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_keys(self) -> "Experiment":
        if len(set(self.expected_roots)) != len(self.expected_roots):
            raise ValueError("Duplicate expected world")
        if len({c.label for c in self.controllers}) != len(self.controllers):
            raise ValueError("Duplicate controller")
        keys = [(r.root_seed, r.label) for r in self.outcomes]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate outcome")
        if any(r.label not in {c.label for c in self.controllers} for r in self.outcomes):
            raise ValueError("Unknown controller outcome")
        return self


class Comparison(Contract):
    experiment_id: str
    reference: str
    candidate: str
    comparable: bool
    issues: tuple[Issue, ...]
    worlds: int
    mean_difference_s: float | None
    mean_difference_95: tuple[float, float] | None
    cvar95_difference_s: float | None
    cvar95_difference_95: tuple[float, float] | None
    failures: dict[str, int]
    limitations: tuple[str, ...]


class Budget(Contract):
    max_tool_calls: int = Field(default=100, ge=1, le=1000)
    max_jobs: int = Field(default=2, ge=0, le=4)
    max_reserved_wall_s: int = Field(default=300, ge=0, le=600)
    max_episode_tasks: int = Field(default=16, ge=0, le=32)


class ExperimentRequest(Contract):
    variation: Variation
    roots: tuple[int, ...] = (624000, 624001, 624002, 624003)
    max_wall_s: int = Field(default=150, ge=10, le=150)
    stage: Literal["development"] = "development"

    @model_validator(mode="after")
    def fresh_roots(self) -> "ExperimentRequest":
        if not 1 <= len(self.roots) <= 4 or len(set(self.roots)) != len(self.roots):
            raise ValueError("One to four unique paired development roots required")
        if any(not 624000 <= seed < 625000 for seed in self.roots):
            raise ValueError("M8 development roots must be 624000–624999")
        return self


class ReportDraft(Contract):
    experiment_id: Identifier
    reference: Identifier
    candidate: Identifier
    issues: tuple[Issue, ...] = ()
    verdict: Literal["qualified", "invalid_comparison", "insufficient_evidence"]
    evidence_uris: tuple[str, ...] = Field(min_length=1, max_length=10)
    followup: Variation | None = None
    explanation: str = Field(max_length=1600)


class Action(Contract):
    kind: Literal["tool", "report"]
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    report: ReportDraft | None = None

    @model_validator(mode="after")
    def one_action(self) -> "Action":
        if self.kind == "tool" and (self.tool is None or self.report is not None):
            raise ValueError("Tool action requires a tool and no report")
        if self.kind == "report" and (
            self.report is None or self.tool is not None or self.arguments
        ):
            raise ValueError("Report action requires only a report")
        return self


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
