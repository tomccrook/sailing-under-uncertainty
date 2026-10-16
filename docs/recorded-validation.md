# Recorded-data component validation

## Sources, scope and registration

This work uses actual public observations after explicit user authorisation. The sailing
source is Greer Jarrett's *From the masthead to the map* dataset, pinned to
[Zenodo record 17340505](https://zenodo.org/records/17340505), archive
`Jarrett_tracks_log_v3.zip`. Its public metadata licences the data under CC BY-SA 4.0.
The adaptation retains attribution, licence, published MD5 and local SHA256 hashes.
The compressed 12 MB archive contains 93,151 original GPS point attributes and voyage
logs from traditional Nordic clinker boats. It also identifies reconstructed tracks,
some rowing/motoring and data loss. It is not Tom's boat or a dinghy-racing benchmark.
No publication includes diary text or crew names. Earlier-version metadata was screened;
validation uses the latest pinned archive rather than its sparse GPS-linked log alone.

The wind source is [NOAA NDBC station 44013](https://www.ndbc.noaa.gov/station_page.php?station=44013),
[2025 historical standard meteorological observations](https://www.ndbc.noaa.gov/data/historical/stdmet/44013h2025.txt.gz).
[Measurement definitions](https://www.ndbc.noaa.gov/faq/measdes.shtml) specify UTC, wind
FROM true north, m/s, and eight-minute averaging for buoy wind speed. These are measured
observations, not forecasts available at historical sailing decisions. Station-level,
ten-minute validation cannot establish minute-scale or spatial racecourse wind fidelity.
All downloaded bytes and source metadata remain local, hashed and outside Git.

Methods, dependencies, source bytes, splits and endpoints lock before held-out diagnostics.
Schema/provenance/time screening precedes registration and is recorded; it does not select
parameters from held-out error metrics. Older synthetic measurement fixtures remain
separate. No measurement fits change the frozen simulator or any policy outcome.

## Sailing GPS quality and independent voyage split

Read only the explicit GPS dBASE archive member, with bounded counts/record sizes and
strict layout/UTF-8 validation; no arbitrary file extraction. Preserve source record IDs,
local clocks, coordinates, original SOG in knots, COG and provenance. Exclude non-primary
or reconstructed-provenance rows from observed-velocity diagnostics, preserving them in
the source-record Parquet and all exclusions/counts. Exclusions break continuity, as do
day/frame/chunk resets, rejected rows and >30 s gaps. Source order is preserved so time
regressions are not hidden by sorting.

Published GPS `DateTime_1` is the corrected local-time character field. Metadata calls its
clock CET but does not resolve all daylight-saving/device conventions. For this relative-
time study, interpret it at fixed UTC+01:00 and label that as assumed. A UTC+02:00 constant-
shift sensitivity leaves elapsed-speed/projection metrics invariant when source-day groups
are preserved. Absolute UTC remains unverified: no forecast or buoy matching is attempted.
GPS bearing is COG, never hull heading; knots convert once to SI. Missing instruments
remain absent. No apparent-wind note becomes a true-wind measurement.

Assign whole voyages chronologically by earliest source local date: the first two-thirds
(rounded up) train, remainder held out. Every day/chunk from a voyage keeps its split.
The 10 s nominal sampling label, 30 s gap and 15 m/s plausibility threshold are fixed
engineering defaults; no test-error-driven threshold fit occurs. Each local WGS84
azimuthal-equidistant frame stays within 100 km, preserving rejected coordinates and
reasons. Chunks are bounded to the existing importer's 10,000-point/5 MB limit.

The numerical validation compares projected finite-difference SOG to WGS84 geodesic
finite-difference SOG on held-out observed intervals with geodesic speed >=0.1 m/s.
Predeclared gate: p99 absolute relative difference <=1%. Both use the same GPS positions;
this checks real-input projection consistency, not independent GPS accuracy. Difference
from the provider's GPS-derived SOG is diagnostic too, not through-water speed accuracy.
Full processed Parquet, per-day quality flags, exclusions and disconnected tracks survive.

## Held-out observed-wind prediction

Use 1 January–1 March 2025 inclusive (60 UTC days) for fitting and 2–31 March (30 UTC days)
for held-out prediction. Keep only consecutive valid 600 s transitions within a day;
missing sentinels, irregular gaps and day boundaries are counted and never bridged.
Convert wind FROM/speed to east/north velocity before averaging or fitting angles.

Fit a scalar vector AR(1) with intercept by training-only least squares, phi clipped to
[0,0.9999]. Compare persistence and training climatology on the same later observations.
Each model's full residual covariance is fitted on training data only, with a 1e-6
numerical diagonal floor. Score vector/speed RMSE, direction RMSE for observed speed
>=0.5 m/s, Gaussian NLL and nominal 95% Mahalanobis coverage. Distributional misspecification
and measurement averaging are explicit; covariance is not hidden instantaneous truth.

Primary comparison is AR(1) minus persistence vector RMSE. Bootstrap whole held-out
calendar days, retaining all transitions within each sampled day, 4000 draws, seed 889910.
Intervals are pointwise and conditional on site/season; serial dependence between days can
reduce effective sample size. A stationary-buoy fit is not a calibrated boat filter or
one-second wind-process parameter estimate. No observations enter final synthetic policies.

## Reproduction and limits

Use the existing independently locked support runtime; root dependencies remain frozen:

```sh
uv sync --directory extensions/support --locked
PYTHONPATH=extensions/support/src extensions/support/.venv/bin/python -m scripts.empirical_validation fetch
PYTHONPATH=extensions/support/src extensions/support/.venv/bin/python -m scripts.empirical_validation prepare --output artifacts/measurements/recorded-validation-v2
PYTHONPATH=extensions/support/src extensions/support/.venv/bin/python -m scripts.empirical_validation run --output artifacts/measurements/recorded-validation-v2
PYTHONPATH=extensions/support/src extensions/support/.venv/bin/python -m scripts.empirical_validation verify --output artifacts/measurements/recorded-validation-v2
```

The bounded fetch command pins the published sailing archive and records exact source hashes. Output directories are exclusive;
choose `--output` for a new registration. Preparation checks the publisher's archive MD5.
The resulting report distinguishes measurement-supported coordinate processing and
observed-wind prediction from unvalidated polar/tack/current/through-water dynamics.
Public data here cannot identify those boat components; claiming full empirical sailing
calibration would be false. That remaining component needs suitably instrumented sessions,
with hull heading, calibrated wind conventions, current/water speed and manoeuvre resolution.
The measured results below follow frozen methods and complete held-out execution.

Preaccess preparation history: the first directory preserved a path-normalisation fault
before any registration or held-out diagnostics. A regression check covers repository-
relative source paths; the repaired method uses a fresh `recorded-validation-v2` directory.

## Recorded execution and findings

Frozen registration `6e53a15211031f30995cc9893383bf9a99441685d1e468effd4cae06464e3ac5`
completed locally in 2.44 s after preparation. The report is
`artifacts/measurements/recorded-validation-v2/report.json`. Publication verifies source,
method, registration and recursive artifact hashes; the static application exposes the
complete processed GPS and held-out wind tables, report, licence and observed-track figure.
No sailing-policy result was recomputed or fitted to these data.

| Recorded GPS accounting | Rows |
|---|---:|
| Original archive records | 93,151 |
| Unique provenance/clock exclusions | 47 |
| Processed, accepted | 91,490 |
| Processed, quality-rejected | 1,614 |

All source rows are accounted for. Exclusion reasons overlap: 47 records have non-primary
provenance and one of those also has an unparseable clock. The processed data cover 112
session-days. Registration contains 18 training and 9 held-out raw voyage identifiers;
one training identifier, `lofotturen`, is a single reconstructed point excluded by the
registered provenance rule. Primary GPS therefore covers 17 training and 9 held-out
voyages. The similarly named actual voyage retains its original independent source ID;
identifiers were neither silently merged nor moved between splits.

On 10,173 held-out contiguous intervals, p99 projection/geodesic relative speed difference
is 0.000749945%, below the registered 1% gate. Provider-SOG agreement on 10,492 intervals
has RMSE 0.0000504 m/s. Both comparisons reuse the same GPS positions: neither estimates
independent position accuracy, hull heading, current or through-water speed. A full
UTC+02:00 sensitivity rerun preserves source-day grouping and every projection, speed and
quality-accounting metric exactly (`artifacts/measurements/registered-clock-sensitivity`).
It does not verify absolute UTC.

The NOAA window contains 12,960 observations, 12 missing-wind rows and 89 gap/day-boundary
events; 8,568 training and 4,278 held-out paired transitions survive the registered rules.
The scalar 600 s AR(1) coefficient is 0.992558. Its continuous-time-equivalent restoring
rate, 0.0000124495 s⁻¹, is a description of these averaged buoy observations, **not** a
one-second racecourse calibration or a parameter transferred into the simulator.

| Predictor | Vector RMSE (m/s) | Speed RMSE (m/s) | Direction RMSE (°) | Nominal 95% coverage | Mean Gaussian NLL |
|---|---:|---:|---:|---:|---:|
| AR(1), training fit | 0.842779 | 0.573646 | 10.49295 | 93.619% | 1.81103 |
| Persistence | 0.843854 | 0.573061 | 10.44971 | 93.782% | 1.81356 |
| Training climatology | 7.641174 | 3.458671 | 99.99341 | 97.452% | 6.09567 |

AR(1) minus persistence vector RMSE is −0.00107468 m/s, with a pointwise whole-day 95%
bootstrap interval [−0.00205857, −0.00004670]. This is about 0.13% of persistence's RMSE;
speed and direction RMSE are slightly worse. The narrow negative vector interval does
not imply practical sailing benefit. These intervals condition on the training-fitted
models; parameter-fitting uncertainty is not resampled. Thirty adjacent held-out days at one site/season,
wind averaging and inter-day dependence limit inference. The nominal ellipses under-cover
for both short-horizon predictors. These are forecasts of later recorded buoy values,
not historical forecasts supplied to a boat or measurements of instantaneous hidden wind.

The requested public recorded-data component validation is complete. Empirical polar,
turn-rate, tack/recovery and water/current calibration remain **unidentifiable with these
inputs**. Those limitations are retained explicitly, rather than fitting boat physics to
GPS ground speed or presenting measured tracks as policy counterfactuals.

```sh
.venv/bin/python -m scripts.followup_publication --empirical artifacts/measurements/recorded-validation-v2 --output artifacts/exports/fresh-recorded-publication
```

The full clock-offset check is independently reproducible:

```sh
PYTHONPATH=extensions/support/src extensions/support/.venv/bin/python -m scripts.recorded_clock_sensitivity --output artifacts/measurements/fresh-clock-sensitivity
```

Its wrapper verifies the baseline report hash, calls the frozen registered importer and
corrects only the sensitivity export's descriptive clock metadata. Original diagnostic
attempts remain preserved; the first ad hoc sensitivity inherited the base +01 metadata,
so the reproducible export supersedes it without changing any numerical finding.

Publication outputs are exclusive; source archives, complete run artifacts and exports
remain ignored by Git. The first preaccess preparation fault remains recorded separately.
