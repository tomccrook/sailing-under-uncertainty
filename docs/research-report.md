# Sailing Under Uncertainty — classical and learned-policy research report

Owner: Tom Crook. Question: under incomplete wind observations and forecast errors,
does uncertainty-aware tactical planning improve unconditional expected finish cost or poor outcomes?

**PPO at the registered budget is worse than the tuned heuristic on independent nominal worlds.**
Fixed belief PPO's mean penalised finish cost is 603.80 s versus B1's 541.06 s;
joint paired difference +62.74 s, interval [26.71,114.17]. The initial locked planner
cohort suggests a mean benefit versus deterministic MPC, without establishing improvement
over strong B1 or stable tail benefit. These are synthetic-model findings, not real-world efficacy.

The earlier classical pilot did not demonstrate improvement over the tuned filtered heuristic.
Risk-aware scenario MPC's mean cost was 547.17 s versus 548.36 s for B1 belief; the paired
mean difference was −1.19 s with a world-bootstrap 95% interval [−23.65, +20.88] s.
Its CVaR95 point estimate was higher. This is an unresolved development finding, not proof
of equivalence or a locked final result. M4 adds the development learned-policy comparison below.
No real-world effectiveness claim is made.

**Latest registered follow-up:** two-path risk MPC is worse than tuned B1 on nominal and
unseen-process pools at the declared matched-work resource point. Against matched B2,
mean/tail differences remain uncertain; on new processes the risk term also underperforms
matched expected-cost planning. Earlier eight-path uncertainties remain intact. Public GPS
and buoy observations validate supported components without identifying boat physics.
The chronological studies below retain their own cohorts and conclusions.

## Method and evidence

The validated simulator uses an original synthetic polar, finite turning/tacks, acceleration,
a 1 km northbound target with 10 m finish circle, ±600 m lateral bounds and 1800 s deadline.
M1's independent ideal reference is 471.404521 s; the finite fixture finishes at 497.541222 s.
Across 1/0.5/0.25 s physics steps its finish-time deviation is below 0.016%.
These validate specific numerical properties, not empirical boat realism.

M2 freezes constant, oscillating, OU, regime and spatial worlds. Sensors are acquired at 2 s,
decisions at 10 s, and station forecasts issued at 30 s. Policy observations carry delivery
times, ages/masks, vector beliefs and available conditional forecasts. Future/diagnostic truth
is excluded. Sensors/forecasts/planning use independent counter-keyed streams. Known uniform
current and synthetic position/heading/water-speed instruments are disclosed assumptions.

M2 diagnostics on eight worlds per family show noisy OU direction RMSE 5.99° raw versus
2.84° filtered. Nominal 95% vector coverage drops to 71.7% for delayed regime changes and
1.3% for biased OU instruments. Confidence is unreliable under model error; uncertainty-aware
control cannot assume that its probability model is calibrated. Details: `observation-validation.md`.

M3 fitted each B1 processing variant on the same eight validation worlds using 16 settings
for layline margin, commitment and boundary lookahead. B2/B3 inherited tuned recourse settings,
searched 40/60 s horizons and selected 60 s. Settings were frozen before the first pilot.
Pilot roots are disjoint from validation and M2 diagnostics: eight worlds per oscillating/OU/regime/
spatial family, 32 paired worlds. Wind fields, sensor innovations, initial state and course match.
B1 belief is feasible in the common immediate candidate set. B2/B3 call the same finite-motion
simulator; B3 uses eight conditional paths and a shared first action, followed by causal approximate
heuristic recourse. No evaluation-world future supplies those paths. A24–A28 disclose model approximations.

Code review identified a single future-instrument-noise draw in the original mean-wind B2.
A certainty-equivalent mean-instrument version retains latency and was subsequently added,
with horizon fitted on validation only. The pilot was already open: this amendment is explicitly
exploratory, and every original outcome remains as `B2_sensor_draw`. See A30 and the revision plan.
Current fresh studies start with the corrected version. No final-test outcomes have been used.

## Computed development results

Finish cost is actual finish time; boundary/deadline failure cost is 2400 s. All 192 combined
episodes remain in the complete table, including five boundary failures. There were no recorded
implementation faults in these studies. Failed-run handling is separately exercised by injected tests.

| Policy | Finished | Mean cost s | CVaR95 s | Difference vs B1 belief s, paired 95% interval | Compute p95 ms/decision |
|---|---:|---:|---:|---|---:|
| B1 raw | 32/32 | 554.17 | 666.39 | +5.81 [−13.40, +23.78] | 0.050 |
| B1 belief | 32/32 | 548.36 | 696.31 | 0 | 0.048 |
| B2 mean instruments | 31/32 | 588.66 | 1748.10 | +40.30 [−27.06, +160.70] | 13.62 |
| B2 single sensor draw, retained | 29/32 | 708.34 | 2400.00 | +159.98 [−12.75, +389.42] | 13.84 |
| B3 expected-cost | 31/32 | 592.40 | 1744.99 | +44.04 [−27.51, +171.55] | 109.51 |
| B3 risk, λ=0.25 | 32/32 | 547.17 | 738.07 | −1.19 [−23.65, +20.88] | 108.74 |

Risk-aware versus corrected B2 mean cost difference was −41.49 s [−159.34, +26.00]; versus
B3 expected-cost it was −45.22 s [−157.85, +15.82]. Neither interval excludes zero.
All intervals resample independent paired worlds with fixed policies, using 4,000 draws;
they describe the pooled empirical family mix. Multiple exploratory contrasts are disclosed,
without a confirmatory significance claim. Per-family tables and 1800/2400/3600 s failure-penalty
sensitivity are in the complete results JSON.

CVaR95 uses fractional empirical tail mass. With 32 worlds there are only 1.6 effective tail
outcomes per policy. B1 belief's bootstrap tail interval was [633.16, 698.86] s and B3 risk's
[596.12, 764.33] s. These wide, discrete estimates cannot justify a strong tail conclusion.
Eight planning samples also make the internal 95% tail score a worst-sample proxy.

Actual CPU timings were measured on this M4 Mac using recorded policy calls. Mean risk-planner
latency was 94.95 ms, versus 0.039 ms for B1 belief. Both fit the 10 s decision clock; the extra
computation has no demonstrated benefit over B1 in this pilot. Total pipeline wall times were
532.91 s for the original study and 35.81 s for the amendment, including recording/analysis.
These are local workload measurements, not isolated CPU/GPU comparisons. CUDA/MPS training
performance remains unmeasured.

## Explained recorded episodes

On `regime-43004`, corrected B2 exited the east boundary at 263.613848 s, position
(600.0, 485.416573) m, before completing a tack. At 250 s it was at x=569.88 m and kept
37.96°; its conditional mean rollout scored that action at 575.48 s. At 260 s it was at
x=594.53 m; every candidate then scored 2400 s. Finite turning could no longer avoid the
boundary. Offline truth at 240 s was −29.02° while the causal belief was −22.68°: estimator
lag and a misspecified mean-reverting forecast help explain why the mean rollout was optimistic.
This is a diagnostic explanation, not a causal attribution experiment.

Risk MPC on the same world finished in 689.342746 s with three tacks. At 250 s it requested
22.93°, reducing eastward motion before its 260 s tack request. The route avoided the boundary,
but B1 belief finished in 634.452137 s with two tacks. This success illustrates a failure mode and
tradeoff; it does not make risk MPC the strongest policy. The M3 browser example is another
actual risk run, `ou-42004`, finishing in 517.176448 s; corrected B2 finished that world in
498.957793 s. It is a navigable example, not a cherry-picked performance claim.

## Reproduce and audit

Recorded artefacts (ignored by Git):

- `artifacts/diagnostics/m2-v2/`: estimator/forecast tables, world arrays, configs and provenance.
- `artifacts/studies/m3/`: original plan, 304 validation episode records, frozen policies, all 160 pilot runs.
- `artifacts/studies/m3-mean-instruments/`: amendment plan, 16 additional validation records, combined 192-episode table, 32 new runs, parent hash and report.
- `artifacts/exports/m3/`: verified public results, paired contrasts, selected replay, scientific SVG and export provenance.

Run directories include full Parquet trajectories, available observations, separate diagnostic
truth, candidate decision costs/hashes, source snapshots, dependency locks, config/artefact hashes,
random-stream lineage, errors and CPU timings. SQLite catalogues index individual runs. README
commands reproduce the current baseline or the historical original-plus-amendment sequence.
The current gate has 87 Python and 16 desktop/mobile browser checks, alongside format/lint/types
and production build. Synthetic evidence cannot establish dinghy realism or financial alpha.

## M4: learning and memory, development evidence

The Gym boundary exposes fixed public features and an empty info dictionary. All representations
share eight categorical commands derived from raw instruments and known geometry, rather than
a hidden/filter-based feasibility map. History stacks four masked frames and resets completely.
Belief features add the causal vector mean/covariance. No LSTM was added: these simpler memory
representations provide an informative controlled comparison without recurrent-state complexity.
Features, scales/clipping and model/config hashes are saved. Neither normalisation nor training
sees final-test worlds or diagnostic truth.

Ordinary clipped PPO uses separate 64×64 actor/value networks. Gamma=1 and terminal-zero
potential shaping give total return `−episode_cost/100−initial_potential`. This preserves
undiscounted expected-cost ordering, including early boundary and deadline failures. It is not
risk-sensitive PPO. Gym checks, reward telescoping, all terminal types and memory resets pass.
Saved Adam state is verified. Small CPU replicas have identical parameters within the locked
environment; ordinary checkpoint resume does not restore live physics/buffers/all RNG states.

Actual 32768-decision workloads measured CPU one/four Torch threads at 1849/1774 decisions/s
versus MPS 924/s. CUDA was unavailable on this host; the RTX 2070 Super was not measured.
The CPU one-thread budget was selected. The earlier 2048-step pilot ranked CPU thread counts
differently, so these are finite workload observations, not universal hardware rankings.

Twelve registered development trials each trained for 65536 decisions: three independent seeds
for present, belief and history, followed by three randomised-training belief models. Total
training budget was 786432 decisions. Whole-study wall time was 469.47 s. The trial catalogue
retains every model, seed, completed training episode, checkpoint, fault channel and source.
Training curves average recent completed exploratory episodes; unfinished rollout cutoffs are
not evaluated finish outcomes. They cannot establish out-of-sample efficacy.

Both unrestricted and command-projected B1 searched twelve configurations on the same 32
validation worlds. Strong B1 selected margin 5 s, commitment 20 s, guard 45 s and scored
545.73 s; projected B1 scored 552.97 s. Selection remains development work: no final-test
fitting or best-seed selection occurred.

| PPO group | Finished model episodes | Mean cost (s) | Difference vs B1 / joint 95% interval (s) | Between-training SD of mean cost (s) |
|---|---:|---:|---:|---:|
| Present, fixed | 67/96 | 1129.46 | +583.73 [74.49,1290.52] | 662.56 |
| Belief, fixed | 96/96 | 551.52 | +5.79 [−24.61,37.91] | 11.04 |
| Four-frame history, fixed | 93/96 | 615.24 | +69.51 [−9.76,193.68] | 71.71 |
| Belief, randomised | 90/96 | 712.63 | +166.90 [27.04,370.28] | 140.71 |

Each group has only **32 independent validation worlds**, with the same worlds reused across
three trained models. Joint intervals resample training seeds and paired whole worlds; repeated
model episodes do not multiply independent-weather evidence. The 95% tail has only 1.6
effective independent worlds. Belief is the mean-across-seeds validation selection, but it
does not demonstrate an advantage over B1. The wider randomised distribution may require a
larger budget; this result does not establish that domain randomisation is generally harmful.

Evidence: `artifacts/benchmarks/m4-v1`, `m4-v2`, `artifacts/studies/m4` and
`artifacts/exports/m4`. Exact trial/feature/scenario/dependency hashes accompany raw JSON/Parquet.
The application displays training diagnostics separately from policy outcome intervals and
retains per-model completion/failure counts.

## M5 registration and limits

Five independent fixed and five randomised belief models are frozen for 1024 nominal worlds
and 512 one-factor shift worlds. Roots are disjoint from actual training and validation manifests.
The predeclared classical subset has 192 worlds: 128 nominal and four per shift slice. It
retains corrected B2 mean and B3 mean/risk alongside strong raw/belief/projected B1. This smaller
planner cohort has limited CVaR precision, even if the larger learned-policy cohort meets the
provisional overall sampling target. Matched-compute and new-family comparisons remain open.

Preaccess review recorded A37: the generic source hash also included README text. The numerical
verifier now hashes Python/dependency inputs independently, allowing later documentation.
The original prepared lock/source remain alongside a replacement registration and access-log
event. The same amendment registers paired-tail intervals and nominal-family slices. Policies,
models, world roots and counts are unchanged; no locked outcomes had been opened. Amendment
after evaluation access is rejected. Local locks are procedural, not security against an owner.
See `locked-evaluation.md`; results are interpreted only after the complete cohort is retained.

## Initial locked M5 results

Evaluation completed with **1536 independent worlds, 20544 retained episodes and 1581 sailing
failures**. The 1024 nominal worlds comprise four equally weighted families; 512 shift worlds
comprise sixteen one-factor slices of 32. Ten selected-representation models (five fixed,
five randomised) and raw/belief/projected B1 share every world. The predeclared planner
subset includes 128 nominal and 64 shifted worlds. Four CPU workers took 851.25 s.
All source, model, dependency and scenario hashes verified before/after; no final-data
fitting, source adjustment or best-seed selection followed evaluation access.

### Nominal learned-policy comparison

| Controller/group | Finished | Mean cost (s) | Difference vs B1 / joint 95% interval (s) | CVaR95 (s) |
|---|---:|---:|---:|---:|
| B1 belief | 1024/1024 worlds | 541.06 | reference | 689.44 |
| B1 command-projected | 1024/1024 worlds | 546.05 | +4.99, conditional world interval in raw report | 693.75 |
| B1 raw | 1024/1024 worlds | 554.93 | +13.87, conditional world interval in raw report | 697.08 |
| PPO belief fixed, five seeds | 4987/5120 model episodes | 603.80 | +62.74 [26.71,114.17] | 1605.60 mixture |
| PPO belief randomised, five seeds | 4524/5120 model episodes | 798.62 | +257.56 [128.06,422.54] | 2400.00 mixture |

B1's mean-cost interval is [537.78,544.33] s. Fixed PPO's between-training SD of model
means is 58.55 s; randomised PPO's is 188.01 s. Fixed-model seed means range from 558.96
to 705.99 s. All five seeds are retained. On the shifted aggregate, B1 costs 662.02 s
(510/512 completions), fixed PPO 835.61 s (2255/2560), and randomised PPO 1027.63 s
(2022/2560). Fixed/randomised differences from B1 are +173.59 [105.25,254.75] and
+365.61 [207.27,592.26] s. These support a negative learned-policy result for this
training budget, features, course and simulator; they do not prove RL cannot work.

CVaR of a seed/world mixture is named explicitly: it is not the average of five
per-model CVaRs or the risk of an ensemble. Nominal tail mass is about 51.2 independent
worlds, not 256 because five models reuse them. Fixed PPO's mixture-CVaR interval is
[988.99,2400] s. A 2400 s plateau is the declared failure-score cap, not an observed
eventual finish time. Bootstrap completion intervals can degenerate at zero failures;
1024/1024 is not proof of perfect reliability. With independent worlds, zero failures
give a one-sided 95% upper mean failure bound `1−0.05^(1/1024)`, about 0.29%.
For heterogeneous independent worlds this zero-event bound follows from the product
of survival probabilities and the arithmetic/geometric-mean inequality.

### Classical planning and principal question

| Controller | Finished / 192 | Mean cost (s) | Difference vs B1 / paired 95% interval (s) | CVaR95 (s) |
|---|---:|---:|---:|---:|
| B1 belief | 192 | 572.61 | reference | 1199.41 |
| B2 mean | 189 | 592.20 | +19.59 [−10.45,56.91] | 1513.00 |
| B3 expected cost | 190 | 571.27 | −1.35 [−28.56,33.93] | 1292.55 |
| B3 risk | 191 | 566.37 | −6.25 [−29.18,19.98] | 1119.64 |

The preregistered risk-versus-B2 mean difference is **−25.84 s [−58.47,−2.34]** on
this constructed nominal/shift mixture. This is initial pointwise evidence of a mean
benefit under these conditions. Risk versus expected-cost B3 is −4.90 s
[−42.80,27.40], and versus stronger B1 −6.25 s [−29.18,19.98]; neither establishes
an improvement. A descriptive nominal-only view (128 worlds, unchanged registered
bootstrap) gives risk-versus-B2 −16.68 s [−50.08,3.92]. The pooled result should not
be generalised to all nominal sailing conditions or interpreted as superiority to B1.
The extra descriptive nominal/shift tables preserve their parent episode hash in
`artifacts/analysis/m5-classical-slices`; no model/parameter selection used them.

Risk-versus-B2 CVaR difference is −393.37 s [−896.29,−6.15]; versus B1 it is −79.78 s
[−442.75,379.11]. With just 9.6 effective tail worlds, these unstable initial intervals
do not close the tail-precision gate. B3 risk p95 decision time is about 142 ms in
the four-worker workload, while nominal B1 belief p95 is 0.0605 ms. B2/B3 share
action candidates and dynamics, but not equal compute. Matched-compute and larger
fresh classical replication are needed before attributing benefit specifically to
uncertainty modelling. Interval coverage is pointwise, not simultaneous across all
metrics and robustness slices; slice results are diagnoses, not universal claims.

### Robustness and a real failure trace

Bias exposes a meaningful exception to the aggregate negative PPO result. In the
32-world +15°/+1 m/s instrument-bias slice, B1 mean cost is 1474.38 s versus fixed
PPO 958.53 s and randomised PPO 1000.07 s. The large-no-go and long-course slices
instead strongly penalise learned policies: fixed PPO means 2265.67 and 1939.09 s,
versus B1 1209.01 and 785.73 s. Randomisation did not cover 45° no-go or course length,
so these are extrapolation checks. A13 couples no-go changes to synthetic-polar
interpolation; these are not isolated measured boat effects. Thirty-two worlds per
slice leave only 1.6 tail worlds, and multiple slice inspection limits inference.

The standalone replay is the fixed PPO seed 71001 worst outcome, `ou-490003`, selected
after evaluation solely for diagnosis. Under the 45° no-go shift it reaches the
1800 s deadline at (−3.54,970.58) m, about 29.63 m from the target, outside the 10 m
finish circle. It travels 1578.52 m and averages about 0.88 m/s through water.
Approximately 27% of recorded headings lie inside the actual no-go zone. At 1790 s
the policy requests 3.27°, nearly direct to the mark, with 0.84 m/s water speed.
This trace illustrates ineffective commands/near-mark generalisation; it does not
establish the network's internal causal reasoning. All sixteen controllers have
worst replays, and full recorded simulation exactly matches each locked outcome.

### Artefacts, checks and remaining work

`artifacts/studies/m5` contains both prepared locks, source snapshots, all model IDs,
independent manifests and access log. Its `evaluation/` holds the complete Parquet
episode table, report, partial-record copy and verified manifest. `artifacts/exports/m5`
holds the public result with training curves and source-manifest hash.
`artifacts/figures/m5` has inspected standalone SVG/PDF/PNG paired-mean and mean/tail
figures. `artifacts/replays/m5-worst` has the sixteen source/config/hash-linked actual
recordings. No weights or generated artefacts are committed.

97 Python tests, 32 desktop/mobile browser contract checks plus actual-export checks, Ruff, strict mypy,
TypeScript/ESLint/Prettier and production build pass. M4 is complete and initial locked
M5 execution is complete. M5's broader precision/ablation gate remains open: larger
fresh planner tail replication, matched compute and new-family evaluation. M6's
paired replay, recorded decision inspector and evidence application now pass locally;
local M7 measurement/weather support and bounded AWS definitions now pass;
actual measurement validation and optional MCP investigation remain. Paid cloud execution remains
disabled without an explicitly supplied budget. No post-test policy revision is made.


## M6 portfolio demonstration

The catalogue publishes all sixteen retained controllers on three diagnostic worlds:
`ou-310000` (first nominal OU index, outcome independent), `ou-310026` (reported
planner failure), and `ou-490003` (reported wider-no-go PPO failure). These were
selected after evaluation, explicitly for explanation. Forty-eight reproductions
match locked status/time/distance/tacks, retaining eleven failures. They add no
independent worlds or efficacy evidence. The failed initial probability-recorder
attempt is preserved in `artifacts/replays/m6-paired`, with its tuple-conversion
traceback; corrected complete recordings are in `m6-paired-v2`. No controller or
locked outcome was changed to fix that presentation fault.

The app synchronises boats on elapsed time and holds terminal states. Its inspector
uses the latest recorded decision at or before the clock: instrument age/mask,
vector belief/covariance, available forecast and requested/executed heading. MPC
candidate mean/CVaR/objective values come from the actual sampled conditional paths;
they are planning estimates, not realised counterfactual finish times. PPO's measured
categorical distribution uses the same public feature vector as deterministic
inference. A maximum-probability action is not a calibrated probability of success
or an explanation of learned intent; local sensitivity is not claimed.

Interactive mean/tail points distinguish the 1024-world nominal learned mixture from
the 192-world classical mixture. Confidence intervals, model variability, completion,
failure cost, effective tails and CPU latency are displayed near their definitions.
M2 estimator/forecast coverage remains separately labelled development evidence.
Every one of the 20544 locked outcomes is downloadable, alongside scientific SVG/PDF,
run/config/decision records, report and cards. Static exports have hashes and bounded
replay size; the browser checks selected bytes and rejects stale, corrupt or future-dated
records. Numerical source and lock verify unchanged. No public job-submission endpoint
or paid service is required.

Current exports are `artifacts/exports/m6-release`, published under `web/public/demo`;
actual desktop/mobile QA captures are in `artifacts/qa/m6`. The 97 Python / 32 browser
checks plus lint, strict types, formatting and production build pass. Presentation
scripts require the repository checkout; the installable numerical package retains
its original frozen scope. See `portfolio-demo.md` for reproduction and limits.

## M7: measurement and operational support

An independently locked support package adds GPX/CSV quality processing, WGS84 metric
projection, raw hashes/Parquet and private browser inspection. Rejected rows are retained
and paths break at gaps, clock errors and impossible jumps. GPS-derived ground speed is
separate from instrumented through-water speed. No measured tracks were available.

Six labelled synthetic days recover twelve known heading transitions. The chronological
four-day training split fits a 13 s mean; the last two days have 6 s mean absolute predictive
error. This tests group-held-out calibration mechanics under designed duration drift, not
physical tack minimum, speed loss or a polar. Small day counts, sampling resolution,
instrument synchronisation and wind/current conventions limit interpretation. The simulator
remains uncalibrated and no policy result is revised.

Four weather adapters preserve UTC/source/model/grid metadata. Archived individual runs
must be available at the historical decision; the default eight-hour publication lag is a
qualified assumption, not proof. Reanalysis, stitched and fixed-lead products are context
only. Offline synthetic responses test the boundary; no downloaded weather is efficacy
or racecourse evidence.

Bounded internal SQLite/subprocess jobs preserve failures, faults and attempts. The actual
four-controller fresh OU job completed in 21.18 s on this Mac; separately indexed runs
match all four retained numerical outcomes exactly. The Batch entrypoint also round-trips
a real deadline failure through a local S3 protocol fixture and strict merge. This is
compatibility evidence, not cloud speed. Terraform definitions and six mock tests cover
zero-resource defaults, private storage, bounded CPU arrays, retries and separate static
hosting. A sourced cost plan reserves an estimated $2.86 under declared usage assumptions;
authorised spend remains zero. Docker is unavailable; Linux equivalence and actual AWS
execution remain unverified. No account access or paid deployment occurred.

See measurement support, bounded jobs and
AWS plan/retries/teardown. M7 extends the local engineering workflow;
M5 precision/matched-compute questions and empirical model validation remain open.

## M8: evaluated local evidence investigation

Nine schema-validated official-SDK MCP tools expose registered evidence, failure-aware
paired statistics and allowlisted development jobs with persistent authority. The
separately locked package preserves the original numerical source/dependencies. No
assistant controls the boat, tunes final policies or submits paid work.

An explicitly authorised download pinned Qwen3.5 9B in 4-bit MLX format. The actual M4
investigation workload peaked at 6.58 GB MLX allocation and decoded at 16.32–20.75 tokens/s;
these are runtime measurements, not full-system memory or a GPU-versus-CPU comparison.
The prompt and inference source froze before model access to evaluation tasks. Two probed
tasks from a separate twelve-case development set and retained failed outputs informed
prompt clarification.

A fixed comprehensive script, one model and a sequential same-model critic were compared
on twelve held-out synthetic audit cases each. Known faults included sensor/polar changes,
unequal compute, declared hidden-state access, excluded failures, unsupported claims,
unpaired worlds and inconsistent costs. Three cases were clean, including an untrusted
instruction-injection note. These are deliberately constructed engineering cases, not a
representative or blinded real-world investigation benchmark.

Exact issue sets were 12/12 for the script, 6/12 for the assistant and 5/12 with its critic.
The model conditions made six and seven false accusations respectively, while each missed
one true issue. Unsupported structured claims numbered seven and eight. All proposed
citations were valid, but correct citations did not make the reasoning correct. Numerical
fields were supplied by tools and accepted reports passed an independent mean check;
this does not evaluate the model's own numerical arithmetic. All cases respected their
bounded authority. The script took 0.31 s versus 369.97 s and 644.80 s across task execution;
the model generated 3671 and 6741 output tokens. Download/load and hardware/electricity
cost are separate; paid inference cost was zero. The critic provided no observed benefit
on this suite. Raw outputs, rejected drafts and all 36 task/workflow records survive.

Two actual paired development jobs retained 16 episodes on four roots. The model selected
12 s sensor delay rather than combined delay/tack stress, so the joint question remains
unanswered. Mean B3-risk minus tuned B1-belief cost was +0.54 s [−14.01,+9.77] at baseline
and +4.63 s [−9.21,+13.20] under delay. Each cohort has only 0.20 effective CVaR95 tail
worlds; neither the sample maxima nor their bootstrap intervals establish tail efficacy.
No policy or M5 conclusion changed. The demonstration shows bounded evidence retrieval,
follow-up submission and verified reporting, alongside an important scope failure.

See M8 protocol, full scores and reproduction. M8's local engineering
and evaluation gate is complete. Strong scripts remain the default for known audits;
M5's separately registered tail/matched-compute work and empirical sailing validation
remain the more important research gaps.

## Independent classical planner tail replication

A separately registered follow-up retained 6400 episodes on 1280 fresh synthetic worlds,
including 41 boundary failures and no implementation faults. Validation selected a 120 s
deterministic horizon from 60/120/240/480 s; scenario planners retained eight 60 s paths.
Four M4 CPU workers completed the locked run in 4960.26 s (82.67 minutes). All original
M5 identities remain unchanged. This study is not pooled with its earlier opened mixture.

On 1024 nominal worlds (51.2 effective tail worlds), B3-risk minus selected B2 is
−14.83 s [−28.97,−1.64] in mean failure-aware cost and −286.81 s
[−547.10,−48.52] in CVaR95. Original B2 comparisons are also negative. These are
pointwise stratified paired 95% intervals, not simultaneous inference. The strongest B1
comparison remains unresolved: +0.64 s [−5.62,+8.26] and +86.91 s
[−16.47,+222.26]. The closer risk-versus-expected-cost B3 objective ablation is also
unresolved: −3.54 s [−13.27,+6.00] and −70.30 s [−251.73,+109.89].
The evidence supports a conditional planner comparison, not superiority of the risk term.

Nominal boundary failures are 0/14/11/4/3 for B1/original B2/selected B2/B3-mean/B3-risk.
B3-risk uses about 4.34 times selected B2's actual outer model steps; their decision p95
values are 138.86 and 34.68 ms. A shared allowance is not equal actual work or runtime.
B1 decision p95 is 0.055 ms under the same instrumented concurrent workload, and its
zero observed nominal failures imply only a 0.292% one-sided 95% upper bound.

Sensor-delay stress reverses the comparison: risk minus selected B2 is +36.41 s
[+2.10,+83.81] and +617.32 s [+15.80,+1370.73]; risk is worse than every
reference in both pointwise endpoints. Joint delay/tack stress has an unresolved mean
versus selected B2 and a marginally negative secondary tail interval. Each stress cohort
has only 6.4 effective tail worlds, and distinct roots prevent interpreting the difference
between stress cohorts as a causal tack interaction. The new nominal sample target is
met, but strong-baseline/objective precision, exact matched computation, new families and
empirical calibration remain open. See [full protocol, tables and reproduction](planner-replication.md).

## Public recorded-data component validation

A separately registered empirical study used actual Greer Jarrett sailing GPS from pinned
Zenodo record 17340505 (CC BY-SA 4.0) and NOAA NDBC 44013 observations from early 2025.
Sources, methods, dependencies, exclusions and chronological splits were fixed before
held-out diagnostics. No resulting fit entered the simulator or any frozen policy.
The GPS voyages concern traditional clinker craft, not a calibrated racing dinghy.

All 93,151 source rows are accounted for: 47 unique provenance/clock exclusions,
91,490 accepted and 1,614 quality-rejected. Whole-voyage registration contains 18 training
and 9 held-out identifiers; one reconstructed training alias has no primary GPS, leaving
17/9 actual primary-GPS voyages. Exclusions, gaps, rejected rows and daily/chunk frame
resets break continuity. On 10,173 held-out intervals, projected/geodesic SOG p99 relative
error is 0.000749945%, below the registered 1% numerical gate. Both estimates reuse GPS;
this checks coordinate processing on recorded inputs, not independent instrument accuracy.
A complete +02 versus +01 assumed-clock sensitivity leaves all relative metrics unchanged.
Absolute UTC remains unverified, so no historical forecast or buoy alignment is inferred.

All wind models and covariances fit 60 earlier days; 4,278 common transitions on 30 later
days supply held-out scores. Vector RMSE is 0.842779/0.843854/7.641174 m/s for scalar vector
AR(1), persistence and training climatology. AR(1) minus persistence is −0.00107468 m/s
[−0.00205857, −0.00004670], a pointwise whole-day bootstrap interval. The approximately
0.13% vector improvement is small; speed and direction RMSE worsen slightly. Nominal
95% ellipse coverage is 93.619%/93.782% for AR(1)/persistence. Thirty adjacent days at
one station/season and averaged observations limit the effective sample size and transfer
claim; these are predictions of recorded buoy values, not historical boat forecasts or
instantaneous hidden truth.

These sources support real-input projection checks and a narrow measured-wind predictor
comparison. They do not identify empirical polar, turn/tack/recovery, current or water
speed. Public component validation is complete within that identifiable scope; full
boat-physics calibration remains explicitly unsupported. Sources, original records,
failed preaccess preparation, complete quality/prediction tables, sensitivity artifacts,
licence and figures are retained. See [methods, full tables and reproduction](recorded-validation.md).

## Measured-work comparison and unseen wind processes

The final follow-up froze five controllers on 2048 fresh paired worlds: 1024 nominal,
512 compound-Poisson gust-front and 512 persistent-trend worlds. Neither new family
participated in tuning. Eight separate nominal validation worlds selected the best feasible
130 s B2 horizon from five candidates. Expected/risk B3 use two conditional 60 s paths;
original eight-path risk remains an unmatched strong reference. Public inputs, finite
physics, candidate construction and recourse stay causal and unchanged.

All 10240 episodes and 67 boundary failures remain. Four CPU workers completed evaluation
in 5908.21 s under a 10800 s cap. Both primary cohorts and all six family slices pass
registered ±10% matching on cohort mean actual outer integration requests per decision.
Nominal expected/risk ratios relative to B2 are 1.00789/1.00695; new-family ratios
0.99987/0.99845. This is not exact per-state, elapsed-time or total-episode matching.
Original eight-path risk uses about four times B2's requests per decision. Nominal decision
p95 is 38.24 ms B2, 36.39 ms expected B3, 36.14 ms risk B3, 143.15 ms original risk
and 0.057 ms B1. Timings include instrumentation and concurrent local workloads.

Each primary pool has 51.2 effective CVaR95 worlds. Positive differences mean two-path
risk is worse; pointwise paired stratified-world 95% intervals condition on frozen policies.

| Cohort / reference | Mean difference (s), interval | CVaR95 difference (s), interval |
|---|---:|---:|
| Nominal / B1 | +16.47 [5.48,28.79] | +348.70 [154.49,574.92] |
| Nominal / matched B2 | −2.21 [−19.67,14.60] | −90.43 [−410.57,222.50] |
| Nominal / matched expected-cost B3 | +9.38 [−1.52,21.04] | +168.60 [−33.94,379.16] |
| New processes / B1 | +31.45 [18.24,46.29] | +537.89 [285.43,820.68] |
| New processes / matched B2 | +11.09 [−6.74,29.58] | +88.31 [−254.31,438.04] |
| New processes / matched expected-cost B3 | +21.43 [9.53,34.79] | +383.28 [150.42,634.54] |

H2 has no established benefit at this resource point; risk-versus-B2 remains uncertain.
Against strong B1, both mean and tail are worse in both pools. H6's risk objective does
not earn its place on unseen processes: two-path risk has 15 failures versus expected
B3's 5 and B1's zero. Nominal counts are 10/5/0. All five controllers, six families and
failure-penalty sensitivities remain in the full report. At penalties 1800/3600 s the
risk-minus-B1 point signs stay positive. No tuning or sample expansion followed access.

New evidence for original eight-path risk is separate: nominal mean versus B1 remains
uncertain (+6.77 [−1.10,16.10] s), but tail is worse (+206.48 [64.51,374.98] s).
On unseen processes its mean is slightly worse than B1 (+5.00 [0.43,11.29] s), tail
unresolved, and mean/tail better than B2 while using about four times its work. Earlier
eight-path risk-versus-eight-path-mean uncertainty is retained: the new study has no
eight-path mean controller. Studies are not pooled. Zero-crossing intervals are not
proof of equivalence; pointwise endpoints are not simultaneous guarantees. Secondary
new-family tails have only 25.6 effective worlds each, old families 12.8.

Reducing samples changes the estimator as well as work; internal CVaR95 is a maximum-sample
proxy at two paths. Process misspecification, terminal approximation and estimator
undercoverage are possible explanations, not identified causal mechanisms. This finding
concerns an implemented budget/controller/process mixture, not all uncertainty modelling
or real-sailing performance. Recorded buoy intervals condition on training-fitted models;
fitting uncertainty is not resampled. Public components do not empirically calibrate the
simulator. See [complete tables, registered resource choice, figures and reproduction](compute-followup.md).
