"""Reproduce registered diagnostic worlds with frozen policies, without fitting."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from sail_lab.config import ExperimentConfig, PolicyConfig
from sail_lab.contracts import Action, Observation
from sail_lab.experiments import file_hash, record_run, write_json
from sail_lab.locked_evaluation import _classical_agent, verify_lock, verify_manifest
from sail_lab.rl_env import headings
from sail_lab.training import PPOPolicy

SELECTION = {
    "ou-310000": (
        "Nominal OU · first registered world",
        "First nominal OU index; outcome independent",
    ),
    "ou-310026": (
        "Nominal OU · planner boundary failure",
        "Previously reported adverse planner run",
    ),
    "ou-490003": (
        "Wider no-go · PPO deadline failure",
        "Previously reported adverse learned-policy run",
    ),
}


class DistributionRecorder(PPOPolicy):
    """Record the categorical policy on exactly the features used for inference."""

    def reset(self) -> None:
        super().reset()
        self.distributions: list[dict[str, Any]] = []

    def act(self, observation: Observation) -> Action:
        features = self.encoder.encode(observation)
        action, _ = self.model.predict(features, deterministic=True)
        with torch.no_grad():
            tensor, _ = self.model.policy.obs_to_tensor(features)
            distribution = self.model.policy.get_distribution(tensor)
            categorical = distribution.distribution
            if not isinstance(categorical, torch.distributions.Categorical):
                raise TypeError("expected discrete categorical PPO")
            probabilities = categorical.probs.cpu().numpy().ravel().tolist()
        choices = headings(observation)
        selected = int(action)
        if selected != int(np.argmax(probabilities)):
            raise ValueError("recorded distribution disagrees with deterministic inference")
        self.distributions.append(
            {
                "elapsed_s": observation.elapsed_s,
                "headings_rad": list(choices),
                "probabilities": probabilities,
                "selected_index": selected,
            }
        )
        return Action(target_heading_rad=float(choices[selected]))


def record(study: Path, output: Path) -> None:
    lock, scenarios = verify_lock(study)
    verify_manifest(study / "evaluation")
    output.mkdir(parents=True, exist_ok=False)
    rows = pq.read_table(study / "evaluation/episodes.parquet").to_pylist()
    configs = {f"{s['config']['world']['family']}-{s['config']['root_seed']}": s for s in scenarios}
    models = {m["variant"]: m for m in lock["models"]}
    policies = {v: PolicyConfig.model_validate(p) for v, p in lock["policies"].items()}
    records: list[dict[str, Any]] = []
    write_json(output / "selection.json", SELECTION)
    status = "failed"
    try:
        for scenario_id in SELECTION:
            selected = sorted(
                (r for r in rows if r["scenario_id"] == scenario_id), key=lambda r: r["variant"]
            )
            if len(selected) != len(models) + len(policies):
                raise ValueError("selected world is missing a registered controller")
            for row in selected:
                variant = row["variant"]
                config = ExperimentConfig.model_validate(configs[scenario_id]["config"])
                learned = (
                    DistributionRecorder(Path(models[variant]["path"]))
                    if variant in models
                    else None
                )
                if learned is None:
                    config = config.model_copy(update={"policy": policies[variant]})
                agent = learned or _classical_agent(config, variant, policies[variant])
                directory = record_run(
                    config,
                    output / "runs",
                    policy=agent,
                    policy_label=variant,
                    model_id=learned.model_id if learned else None,
                )
                summary = json.loads((directory / "summary.json").read_text())
                manifest = json.loads((directory / "manifest.json").read_text())
                if (
                    summary["status"] != row["status"]
                    or abs(summary["elapsed_s"] - row["elapsed_s"]) > 1e-8
                    or manifest["scenario_manifest_hash"] != row["world_hash"]
                    or abs(summary["distance_m"] - row["distance_m"]) > 1e-8
                    or summary["tack_count"] != row["tack_count"]
                ):
                    raise ValueError("diagnostic reproduction disagrees with retained evaluation")
                probabilities = learned.distributions if learned else []
                records.append(
                    {
                        "scenario_id": scenario_id,
                        "slice": row["slice"],
                        "variant": variant,
                        "run_directory": str(directory.resolve()),
                        "manifest_hash": file_hash(directory / "manifest.json"),
                        "cost_s": row["cost_s"],
                        "probabilities": probabilities,
                    }
                )
                write_json(output / "recordings.json", records)
                print(scenario_id, variant, summary["status"], flush=True)
        verify_lock(study)
        status = "succeeded"
    except Exception:
        status = "failed"
        raise
    finally:
        write_json(
            output / "manifest.json",
            {
                "status": status,
                "lock_hash": file_hash(study / "lock.json"),
                "evaluation_manifest_hash": file_hash(study / "evaluation/manifest.json"),
                "recordings": len(records),
                "artifacts": {
                    n: file_hash(output / n)
                    for n in ("selection.json", "recordings.json")
                    if (output / n).exists()
                },
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record(args.study, args.output)


if __name__ == "__main__":
    main()
