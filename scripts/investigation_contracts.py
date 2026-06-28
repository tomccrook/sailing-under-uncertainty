"""Typed static presentation of M8 audit evidence, outside frozen sailing science."""

from typing import Literal

from pydantic import Field, model_validator

from sail_lab.config import Contract

StrategyName = Literal["fixed_script", "local_assistant", "researcher_critic"]


class StrategySummary(Contract):
    name: StrategyName
    tasks: int = Field(ge=1)
    accepted_reports: int = Field(ge=0)
    exact_issue_sets: int = Field(ge=0)
    exact_rate_wilson95: tuple[float, float]
    false_accusations: int = Field(ge=0)
    missed_issues: int = Field(ge=0)
    unsupported_structured_claims: int = Field(ge=0)
    numerical_correct_reports: int = Field(ge=0)
    budget_compliant_tasks: int = Field(ge=0)
    wall_s: float = Field(ge=0)
    model_calls: int = Field(ge=0)
    generated_tokens: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)


class TaskScore(Contract):
    task_id: str = Field(pattern=r"^evaluation-[0-9]{2}$")
    strategy: StrategyName
    status: Literal["succeeded", "failed"]
    expected_issues: tuple[str, ...]
    predicted_issues: tuple[str, ...]
    exact_issue_set: bool
    accepted_report: bool
    evidence_links_correct: bool
    numerical_correct: bool
    error: str | None
    explanation: str | None


class InvestigationCatalogue(Contract):
    schema_version: Literal[1] = 1
    evidence_level: Literal["synthetic_audit_fixture"] = "synthetic_audit_fixture"
    model_repository: str
    model_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    protocol_version: str
    strategies: tuple[StrategySummary, ...]
    tasks: tuple[TaskScore, ...]
    paid_service_cost_usd: Literal[0] = 0
    result_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    evidence_url: Literal["/investigation-evidence.json"] = "/investigation-evidence.json"
    limits: str

    @model_validator(mode="after")
    def complete_denominators(self) -> "InvestigationCatalogue":
        names = {"fixed_script", "local_assistant", "researcher_critic"}
        if {r.name for r in self.strategies} != names or len(self.strategies) != 3:
            raise ValueError("All three complete strategy cohorts required")
        ids = {r.task_id for r in self.tasks}
        if len(self.tasks) != 3 * len(ids) or len(
            {(r.task_id, r.strategy) for r in self.tasks}
        ) != len(self.tasks):
            raise ValueError("Investigation outcomes missing or duplicated")
        for summary in self.strategies:
            rows = [r for r in self.tasks if r.strategy == summary.name]
            if (
                summary.tasks != len(rows)
                or summary.accepted_reports != sum(r.accepted_report for r in rows)
                or summary.exact_issue_sets != sum(r.exact_issue_set for r in rows)
            ):
                raise ValueError("Investigation denominator or summary differs from case records")
        return self
