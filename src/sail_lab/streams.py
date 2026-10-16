"""Stable named stream lineage; counter-keyed exogenous instrument errors."""

import numpy as np

STREAM_NAMES = ("world", "sensors", "policy", "planner", "optimiser", "analysis", "forecasts")


def seed_hierarchy(root: int) -> dict[str, list[int]]:
    streams = np.random.SeedSequence(root).spawn(len(STREAM_NAMES))
    return {
        name: [int(value) for value in stream.generate_state(4)]
        for name, stream in zip(STREAM_NAMES, streams, strict=True)
    }


def stream_seed(root: int, name: str) -> np.random.SeedSequence:
    if name not in STREAM_NAMES:
        raise ValueError(f"unknown random stream: {name}")
    return np.random.SeedSequence(root).spawn(len(STREAM_NAMES))[STREAM_NAMES.index(name)]


def keyed_rng(root: int, name: str, tick: int) -> np.random.Generator:
    if tick < 0:
        raise ValueError("counter must be nonnegative")
    words = [int(value) for value in stream_seed(root, name).generate_state(4)]
    return np.random.default_rng(np.random.SeedSequence([*words, tick]))
