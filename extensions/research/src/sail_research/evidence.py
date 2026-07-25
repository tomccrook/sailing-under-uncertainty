"""Failure-aware paired statistics and explicit comparability checks."""

import numpy as np

from sail_research.contracts import Comparison, Experiment, Issue


def cvar(values: np.ndarray, alpha: float = 0.95) -> float:
    ordered = np.sort(values)[::-1]
    mass = len(ordered) * (1 - alpha)
    count = int(np.floor(mass))
    return float(
        (ordered[:count].sum() + (mass - count) * ordered[min(count, len(ordered) - 1)]) / mass
    )


def information_issues(experiment: Experiment, reference: str, candidate: str) -> list[Issue]:
    indexed = {c.label: c for c in experiment.controllers}
    if reference == candidate or reference not in indexed or candidate not in indexed:
        raise ValueError("Select two distinct registered controllers")
    left, right = indexed[reference], indexed[candidate]
    issues: list[Issue] = []
    if left.information_sha256 != right.information_sha256:
        issues.append("sensor_mismatch")
    if any(
        field.startswith(("diagnostic_", "hidden_", "future_", "realised_future"))
        for controller in (left, right)
        for field in controller.observation_fields
    ):
        issues.append("hidden_state_leakage")
    return issues


def compare(experiment: Experiment, reference: str, candidate: str) -> Comparison:
    issues = information_issues(experiment, reference, candidate)
    indexed = {c.label: c for c in experiment.controllers}
    left, right = indexed[reference], indexed[candidate]
    if left.polar_sha256 != right.polar_sha256:
        issues.append("polar_mismatch")
    if left.compute_units != right.compute_units:
        issues.append("unequal_compute")
    if experiment.proposed_claim == "universal_improvement" or (
        experiment.proposed_claim == "equal_computation"
        and left.compute_units != right.compute_units
    ):
        issues.append("unsupported_claim")
    rows = {(r.root_seed, r.label): r for r in experiment.outcomes}
    expected = {
        (root, label) for root in experiment.expected_roots for label in (reference, candidate)
    }
    actual = {key for key in rows if key[1] in (reference, candidate)}
    if expected != actual:
        issues.append("omitted_failures")
    if any(
        rows[(seed, reference)].world_sha256 != rows[(seed, candidate)].world_sha256
        for seed in experiment.expected_roots
        if (seed, reference) in rows and (seed, candidate) in rows
    ):
        issues.append("unpaired_worlds")
    if any(
        abs(row.cost_s - (row.elapsed_s if row.status == "finished" else experiment.failure_cost_s))
        > 1e-8
        for row in experiment.outcomes
    ):
        issues.append("invalid_cost")
    blockers = set(issues) - {"unequal_compute", "unsupported_claim"}
    comparable = not blockers
    failures = {
        label: sum(r.label == label and r.status != "finished" for r in experiment.outcomes)
        for label in (reference, candidate)
    }
    mean: float | None = None
    tail: float | None = None
    mean_ci: tuple[float, float] | None = None
    tail_ci: tuple[float, float] | None = None
    if comparable:
        a = np.array([rows[(seed, reference)].cost_s for seed in experiment.expected_roots])
        b = np.array([rows[(seed, candidate)].cost_s for seed in experiment.expected_roots])
        differences = b - a
        mean, tail = float(differences.mean()), cvar(b) - cvar(a)
        draws = np.random.default_rng(880024).integers(len(a), size=(1000, len(a)))
        mean_q = np.quantile(differences[draws].mean(axis=1), [0.025, 0.975])
        tail_q = np.quantile([cvar(b[d]) - cvar(a[d]) for d in draws], [0.025, 0.975])
        mean_ci = (float(mean_q[0]), float(mean_q[1]))
        tail_ci = (float(tail_q[0]), float(tail_q[1]))
    return Comparison(
        experiment_id=experiment.experiment_id,
        reference=reference,
        candidate=candidate,
        comparable=comparable,
        issues=tuple(issues),
        worlds=len(experiment.expected_roots),
        mean_difference_s=mean,
        mean_difference_95=mean_ci,
        cvar95_difference_s=tail,
        cvar95_difference_95=tail_ci,
        failures=failures,
        limitations=(
            f"Only {len(experiment.expected_roots) * 0.05:.2f} effective CVaR95 tail worlds; "
            "intervals are exploratory.",
            (
                "Whole-world paired bootstrap, 1000 draws, analysis seed 880024; no training-seed "
                "uncertainty here."
            ),
            (
                "Declared controller compute differs; no matched-compute claim follows."
                if "unequal_compute" in issues
                else "Declared compute units match; actual runtime equality is unverified."
            ),
            "Synthetic audit numbers test investigation only."
            if experiment.evidence_level == "synthetic_audit_fixture"
            else "Fresh development worlds; no confirmatory claim or policy tuning.",
        ),
    )
