# M2 observation and stochastic-world acceptance

Computed locally on 3 October 2026. All inputs are synthetic and uncalibrated.
`uv run sail-lab diagnose-observations --output artifacts/diagnostics/m2-v2`
records estimator and forecast Parquet, resolved configs, source snapshot, scenario hashes,
world arrays, random-stream lineage and a machine-readable report. Eight independent worlds
per family, each 600 s; a 60 s burn-in. Families: constant, OU, regime and spatial.
Sensor slices: clean, noisy (6°/0.35 m/s), 6 s delayed, 50% dropout plus delay, and
12°/0.5 m/s biased. The same world roots pair instrument slices; different roots separate families.
Bootstrap intervals resample worlds, not correlated ticks. Fixed estimator settings were
engineering defaults; these diagnostics did not fit parameters or access final tests.

| Slice | Filter direction RMSE ° | Raw RMSE ° | Vector ellipse coverage (nominal 95%) |
|---|---:|---:|---:|
| Constant / noisy | 2.241 | 5.924 | 99.4% |
| OU / noisy | 2.839 | 5.994 | 93.5% |
| Regime / noisy | 4.566 | 5.872 | 89.8% |
| Regime / delayed | 7.840 | 8.813 | 71.7% |
| OU / dropout | 4.152 | 6.769 | 86.7% |
| OU / biased | 12.197 | 13.268 | 1.3% |

The average absolute direction error in the first 20 s after a regime jump was 10.56°
with noisy sensors and 22.02° with delay. This describes lag; it is not a fitted recovery-time law.
Raw errors include held/stale values and the public prior before delivery. Filter likelihood
scores apply to its approximate vector Gaussian, with instrument clipping and angular
linearisation disclosed. Small sample counts limit calibration precision.

Station forecast direction coverage at 60 s lead in OU worlds was 92.65%
(world-bootstrap interval 88.24–97.06%). Adding a 12° forecast bias reduced it to 53.68%
(40.44–66.91%). Forecast calibration is scored separately from estimator covariance.
The arithmetic-speed forecast and angle Gaussian are deliberately imperfect; “nominal”
is a configuration name, not a claim of calibration. Underdispersed forecasts use half
nominal standard deviations.

Acceptance tests verify frozen query-order-independent worlds, exact OU knot transitions,
log-speed mean correction, independent named streams, fixed initial heading, sensor clock
invariance across physics resolutions, acquisition/availability masks, stale timestamps,
publication causality, forecast issue-time-only world queries, identical station forecasts
on different spatial routes, circular angle handling, PSD covariance and duplicate assimilation.
Policies receive immutable observations; offline truth is a separate replay field.

`uv run sail-lab demo --config configs/experiments/stochastic-smoke.yaml` records the actual
OU/noisy/4 s delayed/5% dropout run. Root 20261003 finished in 515.238076 s, one completed
tack, 52 decisions. This illustrates sensing and route recording; it does not establish
policy superiority. The app shows delivery status, age, filtered uncertainty and available
forecast issue times. The original M1 convergence gate still passes.
