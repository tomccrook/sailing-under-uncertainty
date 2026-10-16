# Independent classical planner replication

This follow-up is designed after the first M5 outcomes and M8 demonstration. It does not
amend the original lock or retune its controllers. The frozen numerical package and root
dependency lock remain unchanged; the new orchestration lives in `scripts`.

## Preaccess design

H2 remains uncertainty-aware versus deterministic planning, with the strong B1 belief
reference and B3 mean versus risk ablation. Primary contrasts are B3 risk minus the
validation-selected deterministic work-budget baseline, original B2 mean, B3 mean, and
B1 belief. Primary cohort is a balanced mixture of four existing nominal families.
Mean failure-aware cost and CVaR95 differences receive paired whole-world bootstrap
intervals; family membership is preserved in resampling. Intervals are pointwise,
not simultaneous across contrasts or endpoints. Improvement requires an interval below
zero for the stated endpoint; an unresolved interval is an unresolved result. All failures
cost 2400 s, with 1800/3600 s sensitivity. Zero-failure bounds and tail counts are reported.

Default target is 256 nominal worlds per family (1024 total, 51.2 effective tail worlds),
plus 32 worlds per family for each of sensor delay and joint delay/tack stress (128 each).
Secondary stress/per-family tails remain small. No empirical weather-frequency or
new-process-family claim is made. Do not combine this replication with previously opened
M5 worlds to claim preregistered confirmation.

Fresh validation roots start at 720000 + family index × 1000, two per family. Nominal
roots start at 800000 + family index × 10000; delay at 850000 + family index × 10000;
joint stress at 900000 + family index × 10000. Validation selects deterministic horizon
60/120/240/480 s by lowest mean cost, breaking ties toward the shorter horizon. All
candidates inherit original B2 parameters and retain its expected-instrument recourse.
B1 and both B3 policies keep original M5 parameters. Validation also measures these
controllers on the actual local workload. No final outcomes select parameters or size.

Each deterministic candidate has at most 480 simulated seconds per immediate heading,
versus eight 60 s paths for scenario planners. Candidate/action sets, decision clock and
physics resolution are shared. This is a matched **forward-model work ceiling**, not equal
actual work, wall latency, effective lookahead, candidate count or episode computation.
Early finishes/boundaries reduce actual work; longer deterministic horizons hold forecast
values beyond its 120 s packet. The selected baseline may use less than its allowance;
selection never forces a worse, longer horizon just to consume work. Outer integration requests and recursive simulator
step calls at manoeuvre events, rollout counts, ceilings and measured per-decision time
are retained. The ceiling bounds outer requests; split calls are counted separately.
Latency includes instrumentation and concurrent worker contention. B3 mean
versus risk uses identical horizon/sample capacity, isolating the objective more closely.

Default local execution uses four CPU workers, one numerical thread per worker and a
7200 s cooperative wall budget. The owner may register a different sample count after
validation timing but before accessing any final worlds. Budget exhaustion or implementation
faults fail the experiment, retaining every completed episode and error; no partial table
becomes a policy comparison and no automatic outcome-driven expansion/retry is allowed.
Local process cancellation is checked between physics steps and policy actions.

The step counter temporarily binds the planner's simulator constructor to a delegating
subclass within one process's policy call. It never changes the outer simulator or feeds
diagnostic truth to a policy. This instrumentation supports process workers, not concurrent
threads. Equality tests compare instrumented and original decisions/episode outcomes.

This is a synthetic follow-up with known method weaknesses. It can improve tail precision
and assess a resource-ceiling baseline, but cannot validate real sailing or close a
new-family generalisation requirement. Results and actual timing will be added only after
complete hash-verified execution.

## Reproduction

Keep the original M5 artifacts available, and use the original root lock:

```sh
uv sync --locked
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m scripts.planner_replication validate
.venv/bin/python -m scripts.planner_replication prepare \
  --config configs/replication/default.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -u -m scripts.planner_replication run
.venv/bin/python -m scripts.planner_replication verify
.venv/bin/python -m scripts.replication_publication \
  --run artifacts/studies/m5-replication
```

Directories are exclusive; use `--validation`, `--output` and the publication's `--output`
for a fresh replication. `prepare` rejects changed methods or parent results. `run` verifies
its lock before and after execution and preserves faulted attempts. Publication requires
complete registered rows, retains full Parquet and report evidence, exports SVG/PDF/PNG
paired figures, and exposes a separately labelled static browser section. Public files
contain no local workspace paths or job-submission endpoint. The forecast's availability
and observation boundary are unchanged. Run artifacts, generated publications and weights
remain outside Git.

## Recorded validation before replication access

Git `bc04887` froze the implementation. The eight development worlds retained all 56
validation episodes, completing in 42.48 s with four CPU workers. Mean failure-aware costs
were 514.47 / 487.65 / 530.96 / 516.79 s for deterministic horizons 60/120/240/480 s.
The minimum selects 120 s; these are selection evidence, not confirmatory efficacy.
B2 at 120 s had 33.06 ms decision p95; B3 mean/risk had 133.88 / 132.03 ms. The 120 s
baseline uses less than its allowance. It is therefore a resource-ceiling comparison,
not measured runtime equivalence. Eight validation worlds leave selection uncertainty.

Summed episode timings for the five final controllers project about 76 minutes for 1280
worlds at four workers, before stress/thermal variation. The preaccess decision retains
the default 7200 s cap and sample target. Lock
`510d72826b3c835273a59ad13e6a9fd055dbd9dd0e2e26f4ea30271a52f30dff`
was written before final-world execution. No sample-size or policy change is permitted in
response to this run's outcomes. New-family generalisation and empirical calibration remain
separate open gates regardless of this replication's result.

## Completed independent replication

All 1280 registered worlds and 6400 episodes were retained. Four CPU workers on an
Apple M4 with 24 GiB memory took 4960.26 s (82.67 minutes), below the registered 7200 s
cooperative cap. Forty-one sailing failures were boundary violations; there were no
simulation faults, discarded episodes or deadline failures. No policy, sample count or
analysis changed after final-world access. The original numerical source and M5 lock
still verify. These results are separate from the original 192-world planner mixture.

The primary nominal cohort contains 1024 worlds and 51.2 effective CVaR95 tail worlds.
Negative differences favour B3 risk; costs include the registered 2400 s failure penalty.
All intervals below are pointwise 95% whole-world paired bootstrap intervals, stratified
by family, using 4000 draws and analysis seed 885300. They do not control simultaneous
coverage across the four references and two endpoints.

| B3 risk minus reference | Mean difference, s [95% interval] | CVaR95 difference, s [95% interval] |
|---|---|---|
| Validation-selected B2 (120 s) | -14.83 [-28.97, -1.64] | -286.81 [-547.10, -48.52] |
| Original B2 (60 s) | -18.91 [-34.20, -4.77] | -387.43 [-670.69, -126.34] |
| B3 expected cost | -3.54 [-13.27, +6.00] | -70.30 [-251.73, +109.89] |
| Tuned B1 belief | +0.64 [-5.62, +8.26] | +86.91 [-16.47, +222.26] |

This provides pointwise evidence of lower failure-aware mean and tail cost than both
specified deterministic planners in this nominal synthetic mixture. It does not establish
an advantage over the strongest tuned B1 reference, or establish that the risk term improves
on expected-cost scenario planning. The latter is the closer objective ablation: its mean
and tail intervals cross zero. Uncertainty propagation, lookahead, recourse and actual work
remain different across B2/B3; the deterministic comparison alone cannot identify which
component causes the observed benefit. A universally better risk-aware policy is unsupported.

| Nominal controller | Mean cost, s | CVaR95, s | Boundary failures / 1024 | Decision p95, ms | Actual outer model steps |
|---|---:|---:|---:|---:|---:|
| B1_belief | 543.04 | 710.58 | 0 | 0.055 | 0 |
| B2_mean | 562.58 | 1184.92 | 14 | 17.821 | 23,542,568 |
| B2_budget | 558.50 | 1084.30 | 11 | 34.680 | 43,794,072 |
| B3_mean | 547.22 | 867.79 | 4 | 138.690 | 190,662,925 |
| B3_risk | 543.67 | 797.49 | 3 | 138.861 | 190,043,719 |

B3 risk used about 4.34 times the selected B2's actual outer model steps, with about
4.00 times its measured decision p95. These include instrumentation and concurrent local
worker contention; they are not an isolated hardware benchmark. B1 had no failures here,
but its nominal one-sided 95% zero-failure upper bound is 0.292%, not proof of zero risk.
The nominal B3-risk versus selected-B2 mean difference remains negative at failure penalties
1800/2400/3600 s (−10.14/−14.83/−24.20 s); tail differences are
−193.06/−286.81/−474.31 s. Sensitivity entries are point estimates, not new intervals.

## Registered stress and family diagnostics

Each stress cohort has 128 worlds and only 6.4 effective tail worlds. Under sensor delay,
B3 risk is worse than every reference in both pointwise endpoints. Against selected B2,
mean difference is +36.41 s [+2.10,+83.81] and CVaR95 difference +617.32 s
[+15.80,+1370.73]; against B1 they are +45.13 s [+2.46,+100.01] and
+889.67 s [+95.84,+1692.24]. Risk has three boundary failures, B2-selected one,
B3-mean one, B1 and original B2 none. Thus nominal improvement is not robust to this
registered incomplete-observation stress.

Under joint delay/tack stress, selected-B2 differences are −17.35 s [−46.15,+2.39]
and −294.83 s [−798.88,−1.73]. The mean is unresolved; the tail interval is only just
below zero in a small secondary cohort. Risk and B1 have zero failures; original/selected
B2 have one each, B3-mean two. Against B1, both intervals cross zero. The apparent
sensor-only/joint difference is not an estimated causal tack interaction: the two cohorts
use distinct independent roots, not within-world paired treatments. The M8 model did not
perform this joint-stress evaluation; this is a separately registered manual follow-up.

Per-family nominal cohorts each have 256 worlds / 12.8 effective tail worlds. Against
selected B2, mean differences are −17.61/−26.81/+8.87/−23.76 s for
oscillating/OU/regime/spatial; only OU has both pointwise intervals below zero. These are
secondary multiple comparisons with small family tails, not proof of a universal family
advantage. No CVaR95 here saturates at the failure penalty, but saturation remains possible
when failures comprise at least 5% of outcomes; such a plateau does not measure unfinished
durations. The full report retains all contrasts, failure categories, penalty sensitivities,
actual and allocated work, recursive split calls, timing and zero-failure uncertainty.

## Acceptance and remaining evidence

The independent execution/publication gate is complete: registered source/dependency/world
identities verify; full outcomes, SVG/PDF/PNG paired figures and immutable manifests are
published; desktop/mobile inspections verify full-table/report download hashes. Root checks
include 130 Python tests, Ruff, strict mypy; 48 browser checks plus TypeScript, ESLint,
Prettier and a production build pass. Generated evidence is excluded from Git.

The nominal sample target was met; this does not assert that all scientific precision
requirements are met. Risk-versus-B1 and risk-versus-mean-B3 effects remain unresolved.
Exact matched actual-work/runtime, new-process-family generalisation and real instrumented
sailing validation remain open. Further policy adjustment would require fresh validation
and untouched final worlds, preserving this complete result. Public results are static;
no cloud resources, paid services or model training were used in this follow-up.

## Subsequent registered work

The later [measured-work/new-process study](compute-followup.md) completes aggregate work
matching and evaluates two unseen families on independent roots. It retains this study's
earlier uncertainties rather than pooling or rewriting them. Its two-path controllers are
different estimands; original eight-path risk remains an unmatched reference. Supported
[public recorded components](recorded-validation.md) are also now measured, while empirical
boat polar/manoeuvre calibration remains unidentifiable from those sources.
