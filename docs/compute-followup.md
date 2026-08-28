# Measured-work and unseen-process follow-up

## Registration before validation and final access

This new study preserves the original M5 and replication sources, environments and results.
All work uses the CPU numerical package. No policy sees a realised future wind path. Two
new wind processes are implemented outside the frozen package; the standard simulator and
observation/forecast boundary receive them through the existing wind-field protocol.

The matched-work resource point is two conditional 60 s paths for B3, compared with a
validation-selected deterministic horizon. The original eight-path B3 risk policy remains
an explicitly unmatched reference, and the original tuned B1 remains a strong baseline.
Reducing scenario count is a declared resource trade-off, not a replacement of previous
results or a claim that two samples accurately estimate a planning tail: CVaR95 is the
maximum with both two and eight samples. Risk and expected-cost B3 at two samples share
horizon, sample paths, candidates and recourse, and differ only in the objective.

Eight fresh nominal validation worlds (two per existing family, roots 1000000 + family
index × 10000 + i) test deterministic horizons 110/120/130/140/150 s. Feasibility requires
B3-risk/B2 mean actual outer integration requests per tactical decision within 5%.
Among feasible candidates, lowest validation mean failure-aware cost selects B2; ties
choose the shorter horizon. No new family participates in tuning. If no candidate is
feasible, preserve validation and halt before final access; do not improvise test tuning.

Final registration contains 1024 nominal worlds (256 per old family), plus 512 per new
family, for 2048 worlds / 10240 episodes. Roots are 1100000 + family index × 10000 + i,
with old families followed by gust_front and persistent_trend. Every policy runs on every
world. Matching passes separately for a cohort only if both two-path B3 variants' measured
mean outer requests per decision are within 10% of B2. This is an explicit numerical-work
matching tolerance, not exact equality at every state or matching wall-clock latency.
Actual total work, recursive calls and measured decision latency remain reported, and a
failed match remains visible. Different realised routes/candidate counts/early termination
mean aggregate matching does not identify an equal-work counterfactual at every state.
No extra dummy rollout, discarded sample or test-driven budget adaptation is allowed.

## New-family definitions

- **gust_front:** compound-Poisson arrivals with mean spacing 140 s. Independent signed
  Gaussian direction amplitudes (SD 16°) and log-speed amplitudes (SD 0.12) share arrival
  times; pulses decay exponentially with time constant 45 s. Zero-start transient is
  deliberate. This is shot noise, not OU diffusion or alternating two-state regimes.
- **persistent_trend:** independent Gaussian angular rates (SD 0.18°/s) persist for 90 s,
  with deterministic restoring time 150 s. Speed follows a 300 s log-speed cycle of
  amplitude 0.1 and an independent phase. This is an integrated random slope process,
  not the old sinusoidal direction, OU or alternating regime family.

Both are spatially uniform and freeze on a 2 s grid. Direction is clipped at ±50° and
speed at the original 12 m/s cap. Piecewise-linear interpolation defines intermediate
queries; the arrays are immutable, root-seeded and independent of policy query order.
They are deliberately constructed misspecification tests, not measured weather. The
planner/filter retain old OU/persistence assumptions and receive no family labels.
World stream children 8/9/10 separate events/direction/speed; prior streams are unchanged.
A generic OU base config holds public instrument settings for new worlds; the separately
registered process config and actual world hash identify the realised process, not that
proxy label. Statistical rows use the actual process family.

## Locked inference and uncertainty decision

Primary endpoints are mean failure-aware cost and fractional CVaR95 for matched B3-risk
minus matched B2, tuned B1 and matched B3-mean, in nominal and pooled new-family cohorts.
Each pooled cohort has 51.2 effective tail worlds; individual new families have 25.6,
nominal families 12.8. Use 4000 whole-world paired bootstrap draws stratified by actual
family, seed 887700; intervals are pointwise 95%, not simultaneous. All legitimate
failures cost 2400 s; sensitivities 1800/3600 retain failed episodes. Counts, zero-failure
bounds, estimator coverage, distance, tacks and compute survive in the report. Four CPU
workers use a 10800 s cooperative cap; faults preserve raw records and fail aggregation.

The earlier eight-path risk-versus-B1 and risk-versus-eight-path-mean findings remain
**unresolved**, without pooling, selective sample expansion or equivalence claims. This
explicitly completes the requested resolve-or-retain decision. This new two-path ablation
is a different estimand and may resolve or remain uncertain on its own registered worlds.
No experiment must yield a winner, and a zero-crossing interval is not proof of equality.

## Reproduction

```sh
uv sync --locked
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m scripts.compute_followup validate
.venv/bin/python -m scripts.compute_followup prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -u -m scripts.compute_followup run
.venv/bin/python -m scripts.compute_followup verify
```

Directories are exclusive. Choose fresh `--validation`/`--output` paths for a new run.
No paid service, training, external forecast or empirical boat parameter is introduced.
The completed results below were published only after complete, identity-verified execution.

## Validation selection and measured resource point

Before final access, 72 validation episodes (five B2 horizons plus the four frozen
references on eight nominal worlds) completed in 39.47 s with four CPU workers. The
matching target was 857.603 outer requests per decision for two-path B3 risk. Every
candidate and legitimate failure remains in the validation table.

| Deterministic horizon | Mean validation cost (s) | Outer requests / decision | B3-risk / B2 work ratio | Within registered ±5%? |
|---|---:|---:|---:|---|
| 110 s | 590.412 | 746.935 | 1.14816 | No |
| 120 s | 625.565 | 808.980 | 1.06010 | No |
| **130 s, selected** | **557.334** | **861.508** | **0.99547** | **Yes** |
| 140 s | 574.254 | 891.039 | 0.96248 | Yes |
| 150 s | 538.138 | 939.184 | 0.91314 | No |

The 150 s candidate had a lower raw validation cost but violated the predeclared work
constraint; selection uses the best feasible baseline. Nothing in the final cohort
chooses the horizon. Validation mean decision latencies were 28.215 ms for selected B2
and 28.866 ms for two-path B3 risk. These measured concurrent/metered timings informed
an approximately 96.6 minute preaccess projection for the registered final workload;
the cooperative local cap was fixed at three hours. Runtime is an estimate, not the
matching criterion. Other local checks can contend for CPU, so final decision timings
are workload measurements, not dedicated-hardware guarantees.

## Locked results — 4 October 2026

All **2048 worlds / 10240 episodes** completed in **5908.21 s (98.47 minutes)** on four
M4 CPU workers, below the registered 10800 s cap. All 67 boundary failures remain;
there were no deadline failures or implementation faults in this run. Methods, parent
identities, registered worlds, raw records and complete Parquet hashes verify. Registration:
`fac4a801474739bd833ec9f29cbaca6303b96c47e73c4fffaf3cbd1650c4f695`.
Report: `e99163c2e0627f4796b3e002ae6b18bd6a96f8748c697ca17657589a9aaa2bc7`.

Each primary cohort has 1024 paired worlds and 51.2 effective tail worlds. Costs below
include failures at 2400 s. All failures were boundary violations.

| Controller | Nominal mean / CVaR95 (s) | Nominal failures | New-family mean / CVaR95 (s) | New-family failures |
|---|---:|---:|---:|---:|
| Tuned B1 belief | 538.107 / 687.766 | 0 | 520.300 / 643.821 | 0 |
| B2, 130 s work target | 556.795 / 1126.897 | 12 | 540.661 / 1093.393 | 13 |
| B3 expected cost, two paths | 545.199 / 867.860 | 5 | 530.316 / 798.431 | 5 |
| B3 risk, two paths | 554.581 / 1036.462 | 10 | 551.751 / 1181.707 | 15 |
| Original B3 risk, eight paths (unmatched) | 544.874 / 894.248 | 5 | 525.297 / 698.208 | 2 |

Actual work matching passes in both pooled cohorts and all six family slices. Nominal
outer requests per decision are 845.889/852.567/851.772 for B2/mean B3/risk B3;
new-family values are 852.686/852.576/851.361. Relative to B2, mean/risk B3 ratios
are 1.00789/1.00695 nominal and 0.99987/0.99845 new-family. This meets the declared
aggregate tolerance, not exact statewise, total-episode or wall-time equality.
Nominal decision p95 is 38.236/36.387/36.136 ms for these three controllers;
new-family p95 is 37.899/35.315/35.372 ms. Original eight-path risk uses about
four times B2's requests per decision and p95 143.147/139.366 ms; tuned B1 is
0.057/0.056 ms. Full recursive-call counts and episode wall sums remain downloadable.

Positive paired differences below mean two-path risk is worse. Intervals are pointwise
stratified whole-world bootstrap 95%, conditional on the frozen controllers/process mixture.

| Cohort / reference | Mean difference (s), 95% interval | CVaR95 difference (s), 95% interval |
|---|---:|---:|
| Nominal / B1 | +16.47 [5.48,28.79] | +348.70 [154.49,574.92] |
| Nominal / B2 | −2.21 [−19.67,14.60] | −90.43 [−410.57,222.50] |
| Nominal / expected-cost B3 | +9.38 [−1.52,21.04] | +168.60 [−33.94,379.16] |
| Nominal / original eight-path risk | +9.71 [−4.38,24.05] | +142.21 [−122.71,411.66] |
| New families / B1 | +31.45 [18.24,46.29] | +537.89 [285.43,820.68] |
| New families / B2 | +11.09 [−6.74,29.58] | +88.31 [−254.31,438.04] |
| New families / expected-cost B3 | +21.43 [9.53,34.79] | +383.28 [150.42,634.54] |
| New families / original eight-path risk | +26.45 [12.39,41.76] | +483.50 [208.63,779.53] |

There is no established H2 benefit at this matched resource point. The two-path risk
objective is worse than the strong sailing heuristic in both primary cohorts and worse
than the matched expected-cost objective on unseen processes. Comparisons with matched
B2 remain unresolved. This is evidence about these implemented controllers, horizons,
instruments and synthetic process mixture, not all uncertainty-aware planning.
Reducing samples changes the estimator as well as the work budget; this study does not
isolate a universal causal effect of compute. The two-sample tail score is a maximum-sample
proxy, and the scenario model remains misspecified under new processes.

Individual new-family risk-minus-B1 mean differences are +25.96 [10.60,44.38] s for
gust fronts and +36.95 [16.42,60.45] s for persistent trends. Their CVaR95 differences
are +420.41 [128.63,780.54] and +659.46 [279.46,1103.13] s respectively. These are
secondary pointwise comparisons with only 25.6 effective tail worlds each; old-family
slices have 12.8. Full tables retain every family and contrast rather than selecting winners.

### Uncertainty retained, with new evidence distinguished

The earlier eight-path risk-versus-B1 and risk-versus-eight-path-mean replication remains
unresolved as registered; it is neither pooled nor expanded after inspection. On this
new nominal cohort, original eight-path risk versus B1 also has an unresolved mean
(+6.77 [−1.10,16.10] s), but a worse tail (+206.48 [64.51,374.98] s).
On new families its mean is slightly worse than B1 (+5.00 [0.43,11.29] s), with an
unresolved tail (+54.39 [−22.88,170.27] s). It improves mean/tail over B2 on the
new-family pool (−15.36 [−29.51,−2.32] and −395.18 [−672.81,−147.52] s), while
using about four times the work. That comparison cannot establish a matched-work advantage.
The new study contains no eight-path expected-cost controller; comparisons with its
two-path counterpart cannot settle the old eight-path objective ablation.

All 1800/2400/3600 s failure-penalty point sensitivities remain in the full report.
Risk-minus-B1 signs remain positive under these penalties. B1's zero failures in each
1024-world primary cohort imply only a one-sided 95% upper bound of 0.292%, not zero
true failure probability. Nominal belief coverage is about 80.6%, new-family about 97.3%;
these diagnostics do not establish that the planning model's conditional tails are calibrated.
No policy, parameter, sample count or numerical source was adjusted after final access.

## Publication and audit

```sh
uv run python -m scripts.followup_publication \
  --study artifacts/studies/compute-followup \
  --empirical artifacts/measurements/recorded-validation-v2 \
  --output artifacts/exports/fresh-followup-publication
```

Current complete results are `artifacts/studies/compute-followup`; verified public files
are `artifacts/exports/followup-final/compute` and `web/public/compute-followup`.
The app shows every controller and all eight cohorts, mean/tail intervals, matching flags,
failures, latency and independent recorded components. Downloads include the complete
10240-row Parquet table, report with sensitivities, registration and SVG/PDF/PNG figures.
Public files are precomputed and contain no weights or job-submission endpoint.
