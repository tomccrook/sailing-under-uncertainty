import numpy as np
import pytest

from sail_lab.learning_study import learning_report


def table() -> list[dict[str, object]]:
    return [
        {
            "scenario_id": str(i),
            "world_hash": str(i),
            "variant": v,
            "status": "boundary_violation" if v.endswith("2") else "finished",
            "elapsed_s": 400 + i,
            "cost_s": 2400 if v.endswith("2") else 400 + i,
            "latencies_s": [0.01],
            "tack_count": 2,
            "distance_m": 1400,
            "manoeuvre_s": 20,
            "action_switch_count": 4,
        }
        for i in range(5)
        for v in ("B1_belief", "PPO_history_fixed/seed-1", "PPO_history_fixed/seed-2")
    ]


def test_hierarchical_bootstrap_counts_worlds_and_between_training_variation() -> None:
    report = learning_report(table(), bootstrap_count=200)
    group = report["learned_groups"]["PPO_history_fixed"]
    assert group["independent_worlds"] == 5 and group["training_seeds"] == 2
    assert group["tail_independent_worlds"] == pytest.approx(0.25)
    assert group["between_training_sd_mean_cost_s"] == pytest.approx(np.std([402, 2400], ddof=1))
    assert group["hierarchical_paired_mean_95"][0] <= 0
    assert group["hierarchical_paired_mean_95"][1] >= 1900
    assert report["policies"]["PPO_history_fixed/seed-2"]["completions"] == 0


def test_report_rejects_missing_duplicates_mismatches_and_dropped_failures() -> None:
    for rows in (table()[:-1], [*table(), table()[0]]):
        with pytest.raises(ValueError, match="paired table"):
            learning_report(rows)
    rows = table()
    rows[0]["world_hash"] = "another"
    with pytest.raises(ValueError, match="worlds differ"):
        learning_report(rows)
    rows = table()
    rows[-1]["cost_s"] = 0
    with pytest.raises(ValueError, match="sailing cost"):
        learning_report(rows)
