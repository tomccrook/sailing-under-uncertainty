"""Reproducible local research entry points."""

from pathlib import Path

import typer

from sail_lab.config import load_config
from sail_lab.experiments import record_run

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Sailing research commands (CPU/local by default)."""


@app.command()
def validate_config(path: Path) -> None:
    """Validate an experiment's resolved SI configuration."""
    config = load_config(path)
    typer.echo(f"Valid schema v{config.schema_version}; config_sha256={config.content_hash()}")


@app.command()
def run(
    config: Path = Path("configs/experiments/smoke.yaml"), output: Path = Path("artifacts/runs")
) -> None:
    """Run a local episode and record summary, Parquet, replay and provenance."""
    directory = record_run(load_config(config), output)
    typer.echo(str(directory.resolve()))


@app.command()
def validate_simulator(output: Path = Path("artifacts/validation/m1.json")) -> None:
    """Check the analytical fixture, reflection symmetry and timestep convergence."""
    from sail_lab.validation import validate_simulator as validate

    report = validate(output)
    typer.echo(f"M1 numerical checks passed; evidence: {output}")
    typer.echo(str(report["relative_finish_time_difference"]))


@app.command()
def demo(config: Path = Path("configs/experiments/smoke.yaml")) -> None:
    """Generate a genuine simulator replay for the local browser application."""
    from shutil import copyfile

    directory = record_run(load_config(config), Path("artifacts/runs"))
    destination = Path("web/public/replay.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    copyfile(directory / "replay.json", destination)
    typer.echo(f"Replay: {destination}; run: {directory}")


@app.command()
def tune_fixture(output: Path = Path("artifacts/tuning/b1-fixture")) -> None:
    """Record all trials in a tiny deterministic B1 parameter search."""
    from sail_lab.validation import tune_fixture as tune

    report = tune(output)
    typer.echo(f"Selected fixture margin: {report['selected_margin_s']} s; evidence: {output}")


@app.command()
def diagnose_observations(output: Path = Path("artifacts/diagnostics/m2"), worlds: int = 8) -> None:
    """Score estimator and forecast calibration on independent development worlds."""
    from sail_lab.diagnostics import observation_diagnostics

    observation_diagnostics(output, worlds)
    typer.echo(f"M2 diagnostics: {output}")


@app.command()
def study_classical(
    output: Path = Path("artifacts/studies/m3"), worlds: int = 8, legacy_b2: bool = False
) -> None:
    """Tune baselines on validation roots, freeze, then run a paired development pilot."""
    from sail_lab.classical_study import StudyConfig, run_study

    run_study(
        output,
        StudyConfig(
            pilot_worlds_per_family=worlds,
            deterministic_sensor_model="sampled" if legacy_b2 else "expected",
        ),
        progress=typer.echo,
    )
    typer.echo(f"M3 development study: {output}")


@app.command()
def revise_classical(
    parent: Path = Path("artifacts/studies/m3"),
    output: Path = Path("artifacts/studies/m3-mean-instruments"),
) -> None:
    """Preserve the original pilot and add B2 with deterministic future instruments."""
    from sail_lab.classical_study import revise_mean_instruments

    revise_mean_instruments(parent, output, progress=typer.echo)
    typer.echo(f"Exploratory amendment: {output}")


@app.command()
def export_classical(
    study: Path = Path("artifacts/studies/m3-mean-instruments"),
    output: Path = Path("artifacts/exports/m3"),
    scenario: str = "ou-42004",
    variant: str = "B3_risk",
) -> None:
    """Verify evidence hashes and export a precomputed comparison and selected replay."""
    from sail_lab.presentation import export_study

    export_study(study, output, Path("web/public"), scenario, variant)
    typer.echo(f"Verified static export: {output}; browser data: web/public")


@app.command()
def benchmark_training(output: Path = Path("artifacts/benchmarks/m4"), steps: int = 2048) -> None:
    """Measure actual finite-simulator PPO rollouts and updates on available devices."""
    from sail_lab.training import benchmark

    typer.echo(str(benchmark(output, steps)))


@app.command()
def study_learning(
    output: Path = Path("artifacts/studies/m4"), steps: int = 65536, threads: int = 1
) -> None:
    """Train all registered seeds/representations, select on validation, compare randomisation."""
    from sail_lab.learning_study import run_learning_study

    report = run_learning_study(output, steps, threads, progress=typer.echo)
    typer.echo(str(report["selection"]))


@app.command()
def lock_evaluation(
    parent: Path = Path("artifacts/studies/m4"),
    classical: Path = Path("artifacts/studies/m3-mean-instruments"),
    output: Path = Path("artifacts/studies/m5"),
) -> None:
    """Extend selected groups to five seeds and lock untouched ID/OOD cohorts."""
    from sail_lab.locked_evaluation import EvaluationConfig, prepare_lock

    typer.echo(prepare_lock(parent, classical, output, EvaluationConfig(), progress=typer.echo))


@app.command()
def evaluate_locked(study: Path = Path("artifacts/studies/m5")) -> None:
    """Verify the lock, record access and evaluate once without tuning."""
    from sail_lab.locked_evaluation import run_locked

    report = run_locked(study, progress=typer.echo)
    typer.echo(f"Retained {report['episodes']} episodes")


@app.command()
def export_learning(
    study: Path = Path("artifacts/studies/m4"), output: Path = Path("artifacts/exports/m4")
) -> None:
    """Verify and export static learning evidence; training curves stay labelled."""
    from sail_lab.locked_evaluation import export_learning as export

    export(study, output, Path("web/public"))
    typer.echo(str(output))


@app.command()
def replay_worst(
    study: Path = Path("artifacts/studies/m5"), output: Path = Path("artifacts/replays/m5-worst")
) -> None:
    """Record and verify each policy's worst locked outcome for explanation."""
    from sail_lab.locked_evaluation import record_worst

    typer.echo(str(record_worst(study, output, Path("web/public"))))


if __name__ == "__main__":
    app()
