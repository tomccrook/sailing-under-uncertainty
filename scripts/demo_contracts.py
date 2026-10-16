"""Read-only presentation contracts; independent of the frozen numerical package."""

import math
from typing import Literal, Self

from pydantic import Field, model_validator

from sail_lab.config import Contract, ExperimentConfig
from sail_lab.contracts import Action, Observation
from sail_lab.planning import upper_cvar


class Candidate(Contract):
    heading_rad: float
    sample_costs_s: list[float] = Field(min_length=1, max_length=64)
    mean_s: float = Field(ge=0)
    cvar_s: float = Field(ge=0)
    objective_s: float = Field(ge=0)


class Planning(Contract):
    candidates: list[Candidate] = Field(min_length=1, max_length=32)
    selected_index: int = Field(ge=0)
    conditional_paths_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    planner_tick: int = Field(ge=0)
    sample_count: int = Field(ge=1, le=64)
    horizon_s: float = Field(gt=0)
    shared_first_action: Literal[True]
    recourse: str


class Decision(Contract):
    elapsed_s: float = Field(ge=0)
    observation: Observation
    action: Action
    executed_target_rad: float
    compute_s: float = Field(ge=0)
    planning: Planning | None = None


class Probability(Contract):
    elapsed_s: float = Field(ge=0)
    headings_rad: list[float] = Field(min_length=8, max_length=8)
    probabilities: list[float] = Field(min_length=8, max_length=8)
    selected_index: int = Field(ge=0, lt=8)

    @model_validator(mode="after")
    def distribution(self) -> Self:
        if any(p < 0 or p > 1 for p in self.probabilities):
            raise ValueError("invalid categorical probabilities")
        if not math.isclose(sum(self.probabilities), 1, abs_tol=1e-6):
            raise ValueError("categorical probabilities do not sum to one")
        if self.probabilities[self.selected_index] != max(self.probabilities):
            raise ValueError("deterministic PPO did not choose maximum probability")
        return self


class Details(Contract):
    version: Literal[1] = 1
    run_id: str
    world_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    variant: str
    model_hash: str | None = None
    config: ExperimentConfig
    decisions: list[Decision] = Field(min_length=1, max_length=500)
    probabilities: list[Probability] = Field(default_factory=list, max_length=500)

    @model_validator(mode="after")
    def causal_scores(self) -> Self:
        previous = -1.0
        for d in self.decisions:
            if d.elapsed_s <= previous or d.observation.elapsed_s != d.elapsed_s:
                raise ValueError("decisions must have matching, increasing public times")
            previous = d.elapsed_s
            if d.observation.forecast and d.observation.forecast.available_s > d.elapsed_s:
                raise ValueError("forecast unavailable at decision")
            if d.planning:
                p = d.planning
                if p.selected_index >= len(p.candidates):
                    raise ValueError("invalid chosen candidate")
                if not math.isclose(
                    p.candidates[p.selected_index].heading_rad,
                    d.action.target_heading_rad,
                    abs_tol=1e-8,
                ):
                    raise ValueError("candidate and recorded action disagree")
                if p.selected_index != min(
                    range(len(p.candidates)), key=lambda i: p.candidates[i].objective_s
                ):
                    raise ValueError("recorded candidate is not the objective minimiser")
                for c in p.candidates:
                    weight = (
                        self.config.policy.risk_lambda
                        if self.config.policy.name == "scenario_mpc"
                        else 0
                    )
                    if (
                        len(c.sample_costs_s) != p.sample_count
                        or any(v < 0 for v in c.sample_costs_s)
                        or not math.isclose(c.mean_s, sum(c.sample_costs_s) / p.sample_count)
                        or not math.isclose(
                            c.cvar_s, upper_cvar(c.sample_costs_s, self.config.policy.cvar_alpha)
                        )
                        or not math.isclose(
                            c.objective_s, (1 - weight) * c.mean_s + weight * c.cvar_s
                        )
                    ):
                        raise ValueError("candidate scores disagree with recorded samples")
        if bool(self.probabilities) != bool(self.model_hash):
            raise ValueError("learned models require measured action distributions")
        if self.probabilities:
            if len(self.probabilities) != len(self.decisions):
                raise ValueError("missing learned action distribution")
            for d, probability in zip(self.decisions, self.probabilities, strict=True):
                if d.elapsed_s != probability.elapsed_s or not math.isclose(
                    d.action.target_heading_rad,
                    probability.headings_rad[probability.selected_index],
                    abs_tol=1e-8,
                ):
                    raise ValueError("learned distribution disagrees with decision")
        return self


class Asset(Contract):
    url: str = Field(pattern=r"^/demo/[a-zA-Z0-9_./-]+$")
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class Recording(Contract):
    variant: str
    run_id: str
    status: Literal["finished", "deadline_exceeded", "boundary_violation"]
    elapsed_s: float = Field(gt=0)
    cost_s: float = Field(gt=0)
    replay: Asset
    details: Asset
    manifest: Asset


class Scenario(Contract):
    scenario_id: str
    label: str
    selection_reason: str
    slice: str
    world_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    pairing_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    recordings: list[Recording] = Field(min_length=2, max_length=32)


class Catalogue(Contract):
    version: Literal[1] = 1
    stage: Literal["locked_diagnostic"] = "locked_diagnostic"
    lock_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    evaluation_manifest_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    default_scenario_id: str
    scenarios: list[Scenario] = Field(min_length=1, max_length=16)
    documents: dict[str, Asset]
    calibration: Asset | None = None

    @model_validator(mode="after")
    def identifiers(self) -> Self:
        ids = [s.scenario_id for s in self.scenarios]
        if len(set(ids)) != len(ids) or self.default_scenario_id not in ids:
            raise ValueError("ambiguous scenario selection")
        run_ids = [r.run_id for s in self.scenarios for r in s.recordings]
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("duplicate recording")
        for s in self.scenarios:
            variants = [r.variant for r in s.recordings]
            if len(set(variants)) != len(variants):
                raise ValueError("duplicate policy in paired world")
        return self
