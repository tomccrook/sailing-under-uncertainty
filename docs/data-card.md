# Data card — synthetic policy studies and separate recorded components

All polar, weather, sensor and course inputs are original synthetic modelling assumptions.
The numerical study ingests no personal tracks or external weather; its synthetic
physics is separate from the later public recorded-data component validation below.
No measured boat calibration or boat-class licence is implied. The five weather families are separately labelled.

Worlds are frozen numerical arrays and deterministic position/time queries. Named world,
sensor, forecast, policy, planner, optimiser and analysis streams are independent. Local wind
measurements and station forecasts have issue/acquisition/availability times; their biases,
noise, dropout and latency are configurable. Position, heading, water speed and known current
are synthetic instrument assumptions. Future realised weather is absent from policy observations.

M2 diagnostics use roots 21000–24007; cases pair the same underlying worlds. M3 validation
uses 31000+family×1000+i, i=0,1. Its held-out development pilot uses 41000+family×1000+i,
i=0…7. The correction to mean-
instrument B2 is labelled as an exploratory amendment on the already-open development pilot;
hyperparameters still come from validation. All failures remain in tables and trajectories.

M4 whole-world training roots are 100000–139999, validation 200000–239999, in four
nonoverlapping family blocks. Fixed and randomised training manifests explicitly name
every world/config; the randomisation parameters are original synthetic assumptions.
M4 validation has 32 independent worlds shared across model seeds. M5 nominal roots use
300000–339999; one-factor shifts start at 400000 in separate 10000-root blocks. The
registered cohort contains 1536 unique roots, disjoint from all actual training/validation
manifests. A smaller outcome-independent subset is marked for expensive classical planning.
No measured weather-frequency interpretation is attached to the chosen family/shift mix.
Model repetitions do not increase the independent-world count. Access and amendments are
logged; source/model/dependency/scenario hashes are verified before and after evaluation.
The initial locked table contains 20544 episodes including 1581 sailing failures.
Sixteen worst-controller replays exactly match their locked outcomes. They are diagnostic
selections after evaluation, never inputs to model fitting or favourable efficacy examples.

Parquet stores full-resolution measurements, beliefs, actions and separate truth columns.
JSON feeds the static browser; the browser does not simulate or submit research jobs.
Every run retains source/config/lock/version hashes and random lineage. Public results/replays
are generated only after artefact verification and stay outside Git. The source and schema
contracts are committed; README recreates the evidence. Measured data require a later,
separate quality/availability/licensing and group-held-out calibration workflow.

M7 implements that workflow in the independently locked `extensions/support` package.
Six instrument sessions and a weather API response are explicitly synthetic. UTC/GPS,
instruments, hashes, exclusions, disconnected paths and derived ground speed are retained.
Twelve known transitions support a four-day training / latest-two-day holdout mechanics
check; that fixture itself supplies no empirical support. True ground-frame wind and
ground-current conventions require instrument review. Apparent wind requires preprocessing;
GPS course is not hull heading. Retrospective weather remains context only; individual-run publication is
known with provenance or explicitly assumed. Import never publishes a private track.
M7 roots 600000–699999 are development compatibility roots, not new final-test evidence.

The M6 catalogue contains 48 exact frozen-policy reproductions on three declared post-test
diagnostic worlds. It retains eleven failures; these examples do not add independent test
worlds. Public downloads contain all 20544 locked outcome rows, selected recorded
observations/explicit offline truth, candidate scores, action distributions, configurations,
manifest identities and verified scientific figures. Raw measured tracks and model weights
are not published. Public replay files are limited to 5000 samples / 12 MB each.

M8 adds 24 labelled synthetic audit tasks: twelve development and twelve evaluation cases,
with independent generator streams 880055 and 880056. Fabricated outcome tables test
investigation defects; they are not sailing-policy evidence. All 36 evaluation workflow
records, failures, raw generations and sealed label hashes are retained. Two separate
actual numerical jobs use four fresh M8 roots 624000–624003 with no final-test tuning;
baseline and increased-latency treatments retain all sixteen episodes. No measured track,
external forecast, hidden policy observation or paid model/cloud service is introduced.

Independent planner replication uses eight validation worlds at
720000 + family-index×1000 + i (i=0,1), with 56 candidate/reference episodes. Final
nominal roots are 800000 + family-index×10000 + i (i=0…255); sensor-delay roots
850000 + family-index×10000 + i and joint-stress roots 900000 + family-index×10000 + i
(i=0…31). Four families are balanced within each cohort. All 1280 final worlds and
6400 paired episodes are retained, including 41 boundary failures. World roots are
disjoint from actual earlier training/validation/evaluation runs. Full Parquet, reports,
registration, source snapshots, plots and manifests are publicly downloadable as static
evidence. No model weights, private track or job endpoint is published. Each final world
is paired across five controllers; stress cohorts have distinct roots from one another,
so their difference is not a within-world causal treatment contrast. The primary mixture
is nominal only and differs from the original 192-world planner mixture.

## Public recorded components

Greer Jarrett's *From the masthead to the map* GPS archive is pinned to DOI
10.5281/zenodo.17340505 (`Jarrett_tracks_log_v3.zip`, 12,135,093 bytes). Metadata
licences it CC BY-SA 4.0. Processed GPS Parquet and observed-track adaptations retain
that licence, attribution and source hashes; diary text and crew names are not published.
Traditional clinker voyages include rowing/motoring and reconstructed provenance; they
are not a measured dinghy-performance benchmark. All 93,151 source rows are accounted
for: 47 excluded for provenance/clock, 91,490 accepted and 1,614 quality-rejected.
Whole-voyage chronological registration contains 18/9 training/held-out identifiers,
including one excluded training reconstruction alias; primary GPS covers 17/9 voyages.
Day/chunk resets, excluded/rejected rows and gaps break observed paths. Corrected local
clocks are assumed fixed CET; constant-offset sensitivity leaves relative metrics
unchanged, without verifying absolute UTC. Raw records remain local and ignored by Git.

NOAA NDBC 44013 historical standard meteorological observations for 2025 are the separate
public wind source (801,902 compressed bytes). Wind FROM is true north, m/s, with UTC
clock and buoy averaging. Sixty days ending 1 March fit all three predictors and their
covariances; 30 later days provide 4,278 common held-out ten-minute transitions. Missing
sentinels and day boundaries are never bridged. Historical observations are not available
forecasts and do not enter policy observations or numerical parameter fitting. Full held-out
predictions, fitted parameters, report, registration, published source metadata and
licence/attribution are retained. No private track is published by the importer.

See [registered methods and measured component results](recorded-validation.md). This
evidence supports real-input numerical projection and a narrow observed-wind prediction
comparison. GPS-derived SOG is not independent speed truth, hull heading or water speed;
these sources cannot identify empirical polar or manoeuvre parameters.

## Measured-work and new-process final worlds

Validation roots are 1000000 + family-index×10000 + i (i=0,1), two per old family.
All 72 candidate/reference episodes survive selection. Final roots are 1100000 +
family-index×10000 + i: 256 each old family, 512 each gust_front/persistent_trend,
2048 worlds and 10240 episodes including 67 boundary failures. Five controllers share
every world; no model repetitions inflate independent-world counts. Roots are disjoint
from actual earlier train/validation/final manifests. The new process family definitions
and source hashes are registered outside the frozen numerical core. Children 8/9/10
separate event/direction/speed randomness, and arrays are query-order independent.
No process-family labels or realised future wind enter the policy. Neither public GPS
nor buoy parameters were transferred into these synthetic worlds. Every outcome, cohort,
failure, work count, latency and paired interval remains in static downloads; no weights
or private submission endpoint is published. See compute-followup.md.
