# Assumptions register

All defaults are synthetic engineering assumptions, not measured Mirror performance.
Changes must name affected runs and whether new confirmatory data are needed.

| ID | Assumption / adjustment | Justification and consequence | Review |
|---|---|---|---|
| A01 | Canonical brief copied from supplied `SAILING_PROJECT_BRIEF.md` | Requested filename absent; preserve original and identical canonical content | Resolved |
| A02 | Python 3.12 installed in isolated local uv runtime | Installed 3.13 is not the specified runtime; no global environment changes | Setup |
| A03 | Symmetric synthetic displacement polar, 35° no-go, clamped high-wind table | Illustrative, bounded; cannot establish boat realism | Tom / measurements |
| A04 | 8°/s turn limit; 5 s exponential speed response; minimum 8 s tack | Tack turns are latched for max(minimum duration, required turn time). Only natural polar loss and relaxation; no extra speed multiplier | Tom / tracks |
| A05 | 1 km north target, 10 m finish circle, ±600 m lateral bounds, 1800 s deadline | Initial leg only; no race rules/rounding claim; circle wholly inside bounds | Development sensitivity |
| A06 | Physics 1 s; decisions 10 s; sensors 2 s; forecasts 30 s | Clocks remain fixed during timestep convergence; provisional tolerance 1% time, identical completion and tack count | Convergence gate |
| A07 | M1 instruments provide perfect local wind and known uniform current | Explicit deterministic validation information set; no future access. M2 must supply noisy/delayed observations instead | Replace at M2 |
| A08 | CPU mandatory; M4 / 24 GB / Node 22.14 observed | NVIDIA host unavailable; do not claim CUDA measured. MPS selection requires actual workload benchmark | M4 gate |
| A09 | Cloud disabled and spend budget zero until explicitly supplied | No paid resources needed for local milestones | Owner budget required for paid jobs |
| A10 | B1 fixture tuning is engineering validation only | Tuned manoeuvre/layline margin must be retuned fairly on development worlds at M3 | No final-test tuning |
| A11 | 100 m direct approach on an already sailable side | Corrects unnecessary near-finish tack exposed by recorded integration; not a general optimality claim | M3 sensitivity |
| A12 | Demo uses 5187 and browser tests 5188, strict port selection | Existing unrelated app on 5173 exposed accidental test reuse; leave other app untouched | Verified desktop/mobile |
| A13 | Polar no-go variation shifts the first knot; when ≥40°, remaps all angular knots uniformly | Keeps interpolation continuous/ordered but changes more than no-go angle; robustness must disclose this coupling | Revisit before polar ablations |
| A14 | Frozen 2 s world knots, stationary exact OU latent transitions, linear interpolation between knots | Physics-step changes query the same world; interpolated path is not an exact continuous OU process | M2 moments/convergence |
| A15 | Log-speed latent mean corrected by half stationary variance; physical speed capped at 12 m/s | Uncapped mean is reference speed; cap changes physical moments and is part of the synthetic model | M2 diagnostics |
| A16 | Initial heading is a course setting (50° default), independent of policy and hidden wind | Prevents initial pose leaking realised wind; reflection fixtures explicitly reflect course heading | Regression checked |
| A17 | Tack completion uses wind direction at initiation as its side anchor | Wind shifts alone do not create tacks; changing wind during a turn remains a simplified manoeuvre model | Sensitivity / Tom |
| A18 | Forecast stream appended to the six existing seed children | Existing stream indices unchanged; sensor/forecast/planner errors independent | Named stream tests |
| A19 | Synthetic sensors observe local ground-frame air velocity, exact position/heading/water speed and known uniform current | An instrument model, not GPS-only observability; noise is angle/speed, delivered after fixed latency and keyed by acquisition tick | M2 leakage tests |
| A20 | Vector random-walk Kalman approximation with fixed q=0.015 m²/s³ and declared prior | Joseph update preserves PSD; delayed R adds q×age; no fitted calibration or optimal smoothing claim | Coverage fails under bias/regimes |
| A21 | Station nowcast uses only issue-time wind; conditional angle/speed Gaussian means revert to public references | Arithmetic-speed forecast differs from log-speed truth; station is broad context rather than a spatial wind map | Separate forecast diagnostics |
| A22 | Dropout masks are delivered with sensor latency; stale measurements retain timestamps and validity | Avoids revealing future delivery status; raw held values and priors remain distinguishable | Acquisition/delivery tests |
| A23 | B2/B3 share current-observation candidates (VMG angle ± offsets, sailable direct target and latched target) | B1's VMG/direct choice is feasible in this same set; B1 raw is a declared processing ablation, B1 belief matches planner inputs | M3 candidate tests |
| A24 | Conditional mean blends local estimate with station forecast; local offset decays at 0.02/s | No inferred spatial field; forecast values beyond packet horizon are held at the last prediction, not sourced from realised future | Model misspecification |
| A25 | Conditional angle/speed errors use OU innovations; initial variance projects local vector covariance, diffusion inferred from available packet uncertainty | Planning priors differ from regime/spatial truth; clipped nonnegative speed is not a Gaussian physical distribution | Calibration stress tests |
| A26 | Rollout recourse is B1 on hypothetical delayed noisy raw instruments, with commitment restarted at horizon start | Shared first action, causal recourse; pending pre-horizon measurements and future filter assimilation omitted; current uniform model ignores spatial measurement position | M3 approximation / sensitivity |
| A27 | Terminal cost is two-leg/direct remaining time at public conditional mean plus an anticipated tack and active manoeuvre | Approximate finite horizon value; cannot claim globally optimal routing | Horizon ablation |
| A28 | Failed sailing episodes cost 2400 s; risk objective (1−λ)mean + λCVaR95, λ=0 or 0.25 | Penalty sensitivity at 1800/2400/3600 s; eight planning samples make CVaR95 a worst-sample proxy | Small-tail warning |
| A29 | B1 boundary lookahead 0 or 30 s is validation-tuned; safety switch can override commitment | Uses public course bounds, prevents a deliberately weak boundary-unaware baseline; fixture default stays 0 | Fair validation search |
| A30 | B2 future instruments now use conditional means with latency; its earlier single-noise-draw rollout is retained as B2_sensor_draw | Code review exposed an unnecessary random recourse draw in the deterministic baseline. Added a stronger version; fit horizons on validation only. Pilot was already open, so the amendment is explicitly exploratory | New revision manifest; no original results erased |
| A31 | PPO uses SB3 clipped PPO, fixed 64×64 actor/value networks and gamma=1 | Maintained optimiser; undiscounted finite-horizon reward telescopes to declared total cost, not CVaR | M4 tests / budget pilot |
| A32 | All PPO representations use eight raw-instrument/geometry commands, with no feasibility mask | Prevents a filter-dependent action map leaking belief into present-only PPO; action difference from M3 requires projected and unrestricted B1 comparisons | Matched-action evaluation |
| A33 | Four-frame masked history and fixed clipped scales, no LSTM initially | Directly tests short memory without recurrent-state complexity; no final-data scaler fitting | Validation / memory reset tests |
| A34 | Randomisation ranges in ppo-protocol.md are synthetic; checkpoints resume operationally | No measured plausibility claim; SB3 does not restore live physics, buffers or all RNG state | Explicit limitations |
| A35 | Locked learned-policy cohort: 1024 nominal and 512 one-factor shift worlds; five fixed and five randomised model seeds | Independent roots and model/source/dependency hashes locked before evaluation; deliberately constructed mixture is not an empirical weather distribution | M5 precision / family limits |
| A36 | Initial locked classical cohort is a predeclared 192-world subset; four local CPU workers, 2400 s wall cap | Measured B3 cost makes full planner replication substantially slower; no outcome-based subset selection. Tail precision, matched compute and new-family comparisons remain open | Do not claim final H2 resolution |
| A37 | Preaccess registration amendment separates documentation from numerical-source hashes and adds paired tail intervals / nominal-family slices | Original prepared lock and source retained; no final outcomes opened, policy/model/scenario changes or tuning. README/report updates must not invalidate numerical provenance | Access-log enforced; amendment prohibited after evaluation starts |

## A38 — M6 diagnostic publication and locked numerical source

The three portfolio worlds are selected after evaluation as stated in `portfolio-demo.md`; their replays are diagnostic examples, not new comparisons or tuning inputs. All retained controllers are reproduced and checked against their evaluation outcomes. Presentation contracts and publication tools live outside the frozen numerical package to keep the M5 lock valid. PPO action probabilities describe the recorded categorical distribution, not calibrated outcome probabilities or a textual rationale. No local sensitivity claim is made without an experiment.

## A39 — Independent M7 support environment

Track, weather and cloud support use a separately locked package under `extensions/support`. This preserves the numerical source and dependency identities of the completed M5 study. Root policy code and model parameters are unchanged. Support commands require the repository checkout; later scientific changes require a fresh registration.

## A40 — Track quality and identifiable transition fixtures

Defaults are 1 s nominal sampling, gaps above 10 s, maximum plausible ground speed 15 m/s and a 100 km local frame. These are configurable quality limits, not measured boat performance. Rejected rows and disconnected segments are retained. Heading-transition duration requires heading, wind and current instruments; GPS speed never substitutes for through-water speed. Six synthetic days exercise grouped chronological fitting; they cannot validate a real polar or tack loss.

## A41 — Archived forecast availability

Open-Meteo single-run initialisation is separate from publication. The optional assumed publication lag is eight hours, deliberately conservative relative to documented typical global-model delay but not a guarantee. Decision-time use is conditional on that assumption unless known availability and evidence are supplied. Reanalysis, stitched historical forecasts and fixed-lead series are context only. Regional/hourly values and interpolation do not establish local wind observations.

## A42 — Bounded development compatibility jobs

M7 uses development roots 600000–699999, separate from M4/M5. At most eight episodes, one local worker/numerical thread, 600 s cooperative wall time and two explicitly retained attempts are permitted. The demonstration preserves tuned classical parameters and one paired OU world; it is not efficacy evidence. Cancellation/timeout/worker errors are not sailing outcomes. Complete, hash-verified indices are required before aggregation.

## A43 — CPU Fargate and conditional cost estimates

Initial AWS definitions use On-Demand Fargate with two vCPUs / 8 GB per task and four configured maximum vCPUs, instead of EC2/Spot/GPU. Storage bootstraps separately from compute. Prices are dated US East quotes; image size, pull/stop time, storage/log volume and network allowance are planning assumptions. A durable reservation ledger and explicit owner budget restrict submissions; neither allowances nor billing alerts guarantee total spend. Defaults authorise zero dollars. Public hosting has a separate budget gate. Docker/Linux equivalence and actual AWS cost/performance remain unverified.

## A44 — M8 isolated evidence investigation

Use the official MCP SDK 2.x in an independent research environment. Evidence tools expose
registered, hash-verified summaries rather than arbitrary files or diagnostic future wind.
Only M8 development roots 624000–624999 and four allowlisted treatments can launch local
CPU jobs. Durable reservations retain failed launches and cannot spend on cloud resources.
Synthetic labelled audit cases score investigation quality separately from sailing results.

## A45 — Local model and bounded sequential critic

The user explicitly authorised a local pretrained model download. A 4-bit Qwen3.5 9B MLX
candidate is chosen for the M4's 24 GB memory; actual task throughput and peak allocation
must be measured. No universal best-model claim is made. The critic reuses this model in
one additional bounded step, so its errors are correlated and its time cost is charged.
Fixture construction is known to the implementing researcher; held-out task results are
not opened for prompt tuning. Paid external API inference remains disabled without a
supplied budget; the optional loopback Ollama adapter is tested with a protocol fixture.

## A46 — Independent planner replication and work ceilings

The already opened M5 planner subset motivates a new, separately locked replication, not
an amendment to old outcomes. A fresh eight-world validation set selects a longer-horizon
deterministic baseline within a 480 simulated-second allowance per immediate heading.
Scenario planners retain eight 60 s rollouts. Candidate counts, early termination and
manoeuvre split events differ; equal ceilings are not equal actual work or elapsed time.
The meter counts outer integration requests separately from recursive step calls at
manoeuvre splits. Measured latency includes instrumentation and local worker contention.
Defaults provide 1024 nominal worlds and two smaller stress cohorts on fresh root ranges;
stratified paired resampling preserves family counts. Sample size is locked before outcomes.
See [replication design](planner-replication.md); new-family/empirical validation stays open.

## A47 — Replication stress contrasts and sample-target language

The registered three cohorts use distinct root ranges. Policies are paired within each
world, but differences between sensor-only and joint delay/tack cohorts do not estimate
a within-world causal tack interaction. The report's `precision_target_met` field denotes
the nominal count threshold (1024 worlds), not scientific precision or gate closure.
Stress/family tails and pointwise multiplicity remain explicit limitations after execution.
These interpretation clarifications change no frozen policies, numerical methods or analysis.

## A48 — Actual-work matching and unseen wind processes

A new locked follow-up selects deterministic lookahead on eight fresh nominal validation
worlds to match two-path B3's measured integration requests per tactical decision within
5%; final cohort matching tolerance is 10%, fixed before access. This matches one measured
numerical-work resource point, not exact statewise work or wall latency. Eight-path B3
and strong B1 remain reported references. Two new synthetic processes use shot noise and
persistent angular rates; no new-family outcomes select parameters. Prior strong-reference
and eight-path objective comparisons remain unresolved, rather than forced to significance.
See [protocol](compute-followup.md); old numerical locks remain unchanged.

## A49 — Public recorded components and identifiable scope

User-authorised public data replace synthetic-only input validation. Jarrett's licensed
GPS archive supports loss-aware import and held-out projection/velocity consistency;
source SOG remains GPS-derived, and corrected local CET clocks have an explicit +01:00
assumption with constant-offset sensitivity. Source-provenance exclusions break paths.
NOAA buoy 44013 supports training-only ten-minute vector prediction and held-out-day
coverage diagnostics. Neither source identifies dinghy polars or finite tack loss; those
remain uncalibrated. Historical buoy observations never become available forecasts in
policy replay. See [recorded validation protocol](recorded-validation.md).
