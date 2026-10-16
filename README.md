# Sailing under uncertainty

A sailing simulator and policy comparison built around one question: **does looking further ahead help when wind observations are noisy and changing tack costs time?**

[Interactive demo](https://tomcrook.dev/demos/sailing) · [Project study](https://tomcrook.dev/projects/sailing) · [Research report](docs/research-report.md)

The simulator models a 1 km upwind leg with finite turning, acceleration, tack recovery, a boat-speed polar and course boundaries. A rule-based controller, deterministic and scenario planners, and PPO all sail through the same frozen wind worlds. Policies receive only information available at their decision time.

## Start here

Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/) are required. The Python dependency lock is committed.

```bash
uv sync --locked
uv run python examples/one_episode.py
uv run sail-lab validate-simulator
uv run python -m pytest
```

The example runs one synthetic episode and prints its outcome. It does not train a model, fetch sailing tracks or start cloud resources. Tests write temporary fixtures; generated runs belong in `artifacts/`.

For the local replay, install Node 22.14 or later, then:

```bash
uv run sail-lab demo
npm ci --prefix web
npm run dev --prefix web
```

Open `http://127.0.0.1:5187`. This generates a small simulator replay. The richer portfolio demo is linked above; its saved training/evaluation runs are not bundled into a fresh checkout.

## What the experiments found

The strong sailing heuristic was difficult to beat. The original locked study found belief-feature PPO had **62.74 seconds more mean cost** than the tuned heuristic on nominal worlds, with a joint 95% interval of **26.71–114.17 seconds**. A later study matched planners' numerical work and retained 10,240 episodes across 2,048 worlds. Its two-path risk planner also underperformed the heuristic.

That result changed the interpretation: good uncertainty estimates, a credible simple controller and the approximation used at the end of a planning horizon matter as much as the sophistication of the optimiser. Boundary exits and missed deadlines remain failures in the score rather than disappearing from the analysis.

These are results from the original completed studies. They are synthetic comparisons, not evidence of faster real-world sailing. The quick-start example does not reproduce the full study. See [the model card](docs/model-card.md), [locked evaluation](docs/locked-evaluation.md) and [matched-work follow-up](docs/compute-followup.md) for assumptions and reproduction commands.

## How the pieces fit

| Area | Where to read | Main decision |
|---|---|---|
| Boat and wind | [`src/sail_lab/dynamics`](src/sail_lab/dynamics), [`world.py`](src/sail_lab/world.py) | Keep weather independent of policy calls; model manoeuvres explicitly. |
| Information | [`observations.py`](src/sail_lab/observations.py), [`beliefs.py`](src/sail_lab/beliefs.py) | Separate available instruments from hidden diagnostic truth. |
| Controllers | [`policies.py`](src/sail_lab/policies.py), [`planning.py`](src/sail_lab/planning.py), [`training.py`](src/sail_lab/training.py) | Compare strong rules, lookahead and learning on the same dynamics. |
| Evaluation | [`locked_evaluation.py`](src/sail_lab/locked_evaluation.py), [`scripts/compute_followup.py`](scripts/compute_followup.py) | Freeze choices before testing; pair worlds and retain failures. |
| Evidence | [`experiments.py`](src/sail_lab/experiments.py), [`schemas`](schemas), [`web`](web) | Connect recorded decisions to configuration, source and outcome hashes. |

The support and research extensions have separate dependency locks under `extensions/`. AWS definitions are optional infrastructure code; nothing in the quick start provisions resources. Field axes are east/north in metres, headings run clockwise from north, and reported wind direction is where wind comes from.

## Checks and limits

```bash
uv run ruff check src scripts tests
uv run mypy src scripts
uv run python -m pytest
```

Tests cover analytical geometry, timestep convergence, repeated seeded runs, instrument timing, causal policy inputs, rollout accounting, failure recording and publication contracts. [Numerical validation](docs/simulator-validation.md) and [observation diagnostics](docs/observation-validation.md) explain the checks in more detail.

The boat polar and manoeuvre model are illustrative. Separate public GPS/buoy checks validate supported components; they do not calibrate the full boat model. Measured tracks, training weights, run databases, cloud state and credentials are excluded. Publication details and current validation status are in [the release notes](docs/public-release.md).
