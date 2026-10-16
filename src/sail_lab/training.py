"""Bounded local PPO trials, explicit device checks and immutable provenance.

SB3 implements clipped PPO; this module owns the research protocol, not a second
RL optimiser. Checkpoints include model/Adam state. Resumption is operational:
the rollout buffer, live physics and RNG state are not restored by SB3 load.
"""

import importlib.metadata
import json
import os
import platform
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

import torch
from pydantic import Field, model_validator
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from sail_lab.classical_study import scenario_config
from sail_lab.config import Contract, ExperimentConfig, PolicyConfig
from sail_lab.contracts import Action, Observation
from sail_lab.experiments import (
    file_hash,
    project_root,
    source_metadata,
    source_snapshot,
    write_json,
)
from sail_lab.rl_env import FeatureEncoder, FeatureSpec, SailingEnv, headings
from sail_lab.streams import keyed_rng, seed_hierarchy

Device = Literal["cpu", "cuda", "mps"]
FAMILIES = ("oscillating", "ou", "regime", "spatial")


class TrainingConfig(Contract):
    version: Literal["expected-cost-ppo-v1"] = "expected-cost-ppo-v1"
    device: Device = "cpu"
    seed: int = Field(default=71001, ge=0)
    features: FeatureSpec = FeatureSpec()
    total_steps: int = Field(default=65536, ge=128)
    n_envs: int = Field(default=4, ge=1, le=32)
    rollout_steps: int = Field(default=256, ge=32)
    batch_size: int = Field(default=256, ge=2)
    epochs: int = Field(default=5, ge=1, le=20)
    learning_rate: float = Field(default=0.0003, gt=0)
    gamma: float = Field(default=1, ge=1, le=1)
    gae_lambda: float = Field(default=0.95, gt=0, le=1)
    entropy_coefficient: float = Field(default=0.01, ge=0)
    shaping: bool = True
    torch_threads: int = Field(default=1, ge=1, le=16)
    hidden_units: Literal[64] = 64
    max_wall_s: float = Field(default=1800, gt=0)

    @model_validator(mode="after")
    def batches_align(self) -> Self:
        size = self.n_envs * self.rollout_steps
        if size % self.batch_size or self.total_steps % size:
            raise ValueError("batch size and total steps must divide whole rollout sizes")
        return self


def check_device(device: Device) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no device fallback")
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable; no device fallback")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise RuntimeError("disable PYTORCH_ENABLE_MPS_FALLBACK for labelled MPS workloads")


def synchronise(device: Device) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def make_scenarios(
    split: Literal["train", "validation"], count: int = 128, randomised: bool = False
) -> tuple[ExperimentConfig, ...]:
    if not 1 <= count <= 1000:
        raise ValueError("worlds per family must be in [1,1000]")
    base = 100000 if split == "train" else 200000
    results = []
    for family_index, family in enumerate(FAMILIES):
        for index in range(count):
            root = base + family_index * 10000 + index
            config = scenario_config(family, root, PolicyConfig())
            if randomised:
                rng = keyed_rng(root, "analysis", 31)
                data = config.model_dump()
                data["world"].update(
                    direction_sigma_deg_sqrt_s=float(rng.uniform(0.8, 1.8)),
                    direction_kappa_s_inv=float(rng.uniform(0.01, 0.04)),
                )
                data["sensor"].update(
                    latency_s=float(rng.choice([2, 4, 6])),
                    dropout_probability=float(rng.uniform(0, 0.15)),
                )
                data["boat"].update(
                    minimum_tack_s=float(rng.uniform(6, 12)),
                    polar_scale=float(rng.uniform(0.9, 1.1)),
                    no_go_deg=float(rng.uniform(32, 38)),
                )
                data["forecast"]["uncertainty_scale"] = float(rng.uniform(0.8, 1.2))
                config = ExperimentConfig.model_validate(data)
            results.append(config)
    return tuple(results)


def scenario_manifest(scenarios: tuple[ExperimentConfig, ...], split: str) -> dict[str, Any]:
    roots = [c.root_seed for c in scenarios]
    if len(set(roots)) != len(roots):
        raise ValueError("duplicate independent world roots")
    return {
        "version": 1,
        "split": split,
        "independent_worlds": len(scenarios),
        "scenarios": [c.model_dump(mode="json") for c in scenarios],
    }


class EvidenceCallback(BaseCallback):
    def __init__(self, output: Path, environments: list[SailingEnv], max_wall_s: float):
        super().__init__()
        self.output, self.environments, self.max_wall_s = output, environments, max_wall_s
        self.started = time.perf_counter()
        self.episodes: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        if time.perf_counter() - self.started > self.max_wall_s:
            raise TimeoutError("training wall budget exceeded; incomplete trial retained")
        for env in self.environments:
            self.episodes.extend(
                {**row, "training_step": self.num_timesteps} for row in env.completed
            )
            env.completed.clear()
        return True

    def _on_rollout_end(self) -> None:
        write_json(self.output / "training-episodes.json", self.episodes)
        # Before the next update: optimiser and policy are a consistent snapshot.
        self.model.save(self.output / "checkpoint.zip")


def train_trial(
    output: Path,
    config: TrainingConfig,
    scenarios: tuple[ExperimentConfig, ...],
    split: str = "train",
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    revision, dirty, source_hash = source_metadata()
    manifest: dict[str, Any] = {
        "version": 1,
        "stage": "development_training",
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "code_revision": revision,
        "dirty_tree": dirty,
        "source_snapshot_hash": source_hash,
        "config": config.model_dump(mode="json"),
        "seed_hierarchy": seed_hierarchy(config.seed),
        "hardware": {"platform": platform.platform(), "machine": platform.machine()},
        "dependencies": {
            n: importlib.metadata.version(n)
            for n in ("torch", "gymnasium", "stable-baselines3", "numpy")
        },
        "resume": "operational only; SB3 model/optimizer saved, live env/buffer/RNG not restored",
        "objective": "expected unconditional cost, gamma=1; not CVaR training",
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "scenario-manifest.json", scenario_manifest(scenarios, split))
    write_json(
        output / "features.json",
        {**config.features.model_dump(mode="json"), "names": config.features.names},
    )
    write_json(output / "source-snapshot.json", source_snapshot())
    (output / "uv.lock").write_bytes((project_root() / "uv.lock").read_bytes())
    started = time.perf_counter()
    environments: list[SailingEnv] = []
    callback: EvidenceCallback | None = None
    vector = None
    try:
        check_device(config.device)
        torch.set_num_threads(config.torch_threads)
        environments = [
            SailingEnv(scenarios, config.features, config.shaping) for _ in range(config.n_envs)
        ]

        def factory(env: SailingEnv) -> Callable[[], SailingEnv]:
            return lambda: env

        vector = DummyVecEnv([factory(env) for env in environments])
        model = PPO(
            "MlpPolicy",
            vector,
            device=config.device,
            seed=config.seed,
            n_steps=config.rollout_steps,
            batch_size=config.batch_size,
            n_epochs=config.epochs,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            ent_coef=config.entropy_coefficient,
            policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        )
        if model.device.type != config.device:
            raise RuntimeError(f"device mismatch: requested {config.device}, got {model.device}")
        model.set_logger(configure(str(output), ["csv"]))
        callback = EvidenceCallback(output, environments, config.max_wall_s)
        synchronise(config.device)
        model.learn(total_timesteps=config.total_steps, callback=callback)
        synchronise(config.device)
        model.save(output / "model.zip")
        manifest.update(
            status="succeeded",
            actual_steps=model.num_timesteps,
            parameter_count=sum(p.numel() for p in model.policy.parameters()),
        )
    except BaseException:
        manifest["status"] = "failed"
        (output / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        if callback is not None:
            write_json(output / "training-episodes.json", callback.episodes)
        if vector is not None:
            vector.close()
        manifest.update(
            ended_utc=datetime.now(UTC).isoformat(), wall_s=time.perf_counter() - started
        )
        manifest["artifacts"] = {
            p.name: file_hash(p)
            for p in sorted(output.iterdir())
            if p.is_file() and p.name != "manifest.json"
        }
        write_json(output / "manifest.json", manifest)
    return manifest


class PPOPolicy:
    def __init__(self, trial: Path):
        manifest = json.loads((trial / "manifest.json").read_text())
        if manifest["status"] != "succeeded":
            raise ValueError("cannot evaluate an incomplete training trial")
        for name in ("model.zip", "features.json", "scenario-manifest.json"):
            if file_hash(trial / name) != manifest["artifacts"][name]:
                raise ValueError(f"training artifact hash mismatch: {name}")
        self.config = TrainingConfig.model_validate(manifest["config"])
        self.encoder = FeatureEncoder(self.config.features)
        torch.set_num_threads(1)
        self.model = PPO.load(trial / "model.zip", device="cpu")
        self.model_id = file_hash(trial / "model.zip")

    def reset(self) -> None:
        self.encoder.reset()

    def act(self, observation: Observation) -> Action:
        action, _ = self.model.predict(self.encoder.encode(observation), deterministic=True)
        return Action(target_heading_rad=headings(observation)[int(action)])


def benchmark(output: Path, steps: int = 2048) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    scenarios = make_scenarios("train", 4)
    results = []
    for device in ("cpu", "mps", "cuda"):
        for threads in (1, 4) if device == "cpu" else (1,):
            name = f"{device}-threads{threads}"
            config = TrainingConfig(device=device, total_steps=steps, torch_threads=threads)
            try:
                check_device(config.device)
            except RuntimeError as exc:
                results.append({"trial": name, "status": "unavailable", "reason": str(exc)})
                continue
            try:
                result = train_trial(output / name, config, scenarios, "benchmark_only")
                results.append(
                    {
                        "trial": name,
                        "status": result["status"],
                        "wall_s": result["wall_s"],
                        "decision_steps_s": result["actual_steps"] / result["wall_s"],
                    }
                )
            except Exception as exc:
                results.append({"trial": name, "status": "failed", "reason": str(exc)})
    report = {
        "stage": "workload_benchmark",
        "steps_per_trial": steps,
        "trials": results,
        "scope": "four sequential vector envs, finite physics/instruments and PPO updates",
        "selection": "choose successful actual workload; no accelerated fallback",
    }
    write_json(output / "report.json", report)
    return report
