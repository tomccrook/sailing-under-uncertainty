# M3 classical planner design and development protocol

B1 raw and B1 belief use identical polar/layline/direct/boundary logic. Raw holds the last
instrument value (including its age/mask) and uses the declared prior before first delivery;
belief uses the vector estimate. All policies have access to the same immutable observation;
the ablation changes processing. No realised evaluation world is passed to a planner.

B2 mean and B3 mean/risk search the same immediate headings: on each side, VMG−5° (at
least no-go+1°), VMG and VMG+10° (capped at 90°), the latched executed target, and the
sailable direct target. Duplicate headings are removed. B1 belief's VMG/direct output
is feasible in this set. Candidate actions are headings; physical finite-turn/tack rules
are shared. B1 does not score all candidates. Rollouts call the single simulator at the
same 1 s resolution as the episode; no separate instantaneous-turn planner model exists.

B2 uses the conditional mean path. B3 samples eight conditional angle/speed OU error paths,
with posterior local covariance and diffusion derived from the available forecast packet.
Samples never come from the realised world's future. All candidates use common samples and
hypothetical instrument innovations. Each candidate has one immediate action across samples.
Future B1 recourse responds to deliverable noisy measurements within that hypothetical path.
The simplified recourse holds raw values and resets commitment at horizon start; it omits
future filter assimilation and pending pre-horizon acquisitions. No scenario-tree optimality
is claimed. The station blend does not estimate a spatial field. See A23–A29.

The terminal approximation estimates two-leg/direct remaining time using the public
forecast mean, with active-turn time and an anticipated tack. Boundary/deadline failures
receive 2400 s; finish cost is actual finish time. B3 risk uses 25% upper-tail CVaR95 and
75% mean. With eight rollout samples this tail score is the maximum sample, which can
cause conservative or unstable choices. Full candidate costs, chosen index, compute time,
conditional-path hash, seed/tick and recourse version are saved in decision records.

`uv run sail-lab study-classical` writes the plan before execution; current runs use expected
B2 instruments. `--legacy-b2` reproduces the original single-draw version below. Validation uses two
independent roots per oscillating/OU/regime/spatial family (31000+family×1000+i).
Each B1 processing variant gets the same 16-setting search: layline margins 5/9/15/25 s,
commitment 20/40 s, boundary lookahead 0/30 s. Both planners inherit tuned B1 belief recourse;
horizons 40/60 s receive identical validation worlds. Expected cost selects settings for
all variants. The risk weight and eight-sample budget are declared defaults, not fitted.

Selected policies are timestamped and frozen before opening any pilot outcomes. Pilot roots
41000+family×1000+i are disjoint from validation and M2 diagnostics; default eight per family,
32 independent paired worlds and 160 episodes. This is held-out development evaluation;
M5 locked final evaluation remains untouched and unimplemented. Every sailing failure stays
in the table. Implementation faults abort the study and preserve error/config evidence.
Paired world bootstrap intervals apply to fixed policies and the pooled empirical family mix.
At 95% CVaR, 32 worlds provide only 1.6 effective tail outcomes; tail conclusions are exploratory.
Per-family counts, penalty sensitivity and actual CPU planner latency are disclosed.

Smoke evidence (root 20261003 OU, 60 s horizon, 8 samples, 30 s recourse boundary lookahead):
B2 finished at 528.940317 s with four tacks and 0.620 s aggregate policy compute;
B3 risk finished at 506.514865 s with two tacks and 4.755 s compute. These are single-world
functional checks made before validation tuning. They are not evidence of superiority.

## Baseline modelling amendment

Code review after the first pilot identified that B2's mean-wind rollout used a single
future instrument-noise draw. That is a deterministic weather model with noisy recourse,
not a strictly deterministic mean-instrument model. It can add avoidable decision noise.
`deterministic_sensor_model=expected` now supplies conditional-mean future instruments,
retains latency, and treats future delivery as present rather than sampling dropout.
B3 continues sampling hypothetical noisy/dropped instruments. This is a disclosed
certainty-equivalent instrument approximation for B2.

`uv run sail-lab revise-classical` preserves the original study and its B2 outcomes as
`B2_sensor_draw`. It tunes only 40/60 s horizons on the same eight validation roots,
freezes the selection, then adds corrected B2 episodes on the same 32 development worlds.
The new complete six-policy table retains every original outcome and verifies parent
artefact hashes. The pilot was already open; this is an exploratory modelling amendment,
not a preregistered confirmatory result. Final M5 evaluation must lock the corrected
policy and new untouched scenarios. A30 records the adjustment.

## Acceptance evidence

The original study retained 304 validation records and 160 pilot episodes; the amendment
added 16 validation records and 32 corrected B2 episodes. The combined 192-episode table
contains five boundary failures on 32 independent paired worlds. All six variants are
reported, including the legacy B2. No failed episode was removed and no final-test fitting
occurred. Source revisions, snapshots, locks, streams and original/amended plan hashes
identify the two executions. Full results and episode explanations: `research-report.md`.

The browser's exported risk replay for `ou-42004` is a genuine retained run, finishing at
517.176448 s. `export-classical` verifies study/run hashes, recomputes paired contrasts against
B1 belief, corrected B2 and B3 mean, then exports static JSON and a scientific SVG. Original
records stay intact; display policy labels/parameters are explicitly derived fields.
The data export has no experiment-submission API. A completed mean-instrument study is
rejected by the amendment command because it already contains the correction.

M3 passes the implementation and exploratory comparison gate. It does not establish
policy effectiveness. M4 is next; final policy freezing and untouched evaluation remain M5.
