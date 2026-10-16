"""Run a small synthetic episode without training or downloading observations."""

from sail_lab.config import ExperimentConfig
from sail_lab.experiments import simulate


def main() -> None:
    summary, samples, decisions = simulate(ExperimentConfig())
    print(summary.model_dump_json(indent=2))
    print(f"Recorded {len(samples)} states and {len(decisions)} policy decisions.")


if __name__ == "__main__":
    main()
