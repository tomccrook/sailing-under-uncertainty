# Model card — classical controls and M4 PPO

B1 raw/belief uses the original synthetic polar's VMG,
layline/direct geometry, commitment and validation-tuned boundary lookahead. Its inputs are
synthetic noisy/delayed wind instruments or the vector filter, with known uniform current.
B2 is finite-model receding-horizon planning at conditional forecast means. B3 samples
conditional angle/speed paths and uses expected-cost or 25% CVaR95-weighted objectives.
The immediate candidate set and finite physics are shared. Neither receives the realised world.

The vector filter is a local random-walk approximation; delayed regime shifts and biased
instruments expose undercoverage. Station forecasts do not identify a whole spatial field.
Rollout recourse is an approximate causal heuristic, not an optimal scenario tree. Horizon 60 s,
eight scenario samples, finite manoeuvres and terminal-cost assumptions are disclosed in
`classical-planning.md`. Code/source hashes and full resolved parameters identify each run.

The 32-world development pilot does not demonstrate benefit over tuned filtered B1. The B2
mean-instrument amendment followed opening the pilot; it remains exploratory. No final-test
fitting, measured calibration or cloud run has occurred. This model is intended
for research demonstration, with no live navigation or racing-rule enforcement validation.

M4 trained twelve ordinary expected-cost PPO models in SB3/PyTorch: present, four-frame
history and explicit vector belief at three independent optimiser seeds each, followed
by three domain-randomised models of the validation-selected belief representation.
Actor and value networks each have two 64-unit tanh layers. Fixed scales, clipping,
feature provenance and raw-instrument eight-command action mapping are in ppo-protocol.md.
There is no hidden-world feature or feasibility mask. Gamma=1 and telescoping shaping
preserve unconditional episode cost. This is not a CVaR-trained model.

Each model has a content-hashed SB3 checkpoint, Adam state, feature transform, config,
whole-world training manifest, dependency/source snapshot and completed-episode log.
CPU support is verified; MPS is benchmarked and slower on this workload; CUDA is unavailable
on the current host. Checkpoint continuation is operational, not exact RNG/env restoration.
Two small CPU replicas match parameters in the current locked environment only.

Development belief PPO was close to B1 (551.52 versus 545.73 s), without evidence of
improvement; present-only and history showed larger between-seed instability. Randomised
training did not improve nominal validation at the same budget. These are development
observations on 32 worlds, not a learned-policy deployment claim. M5 extends both selected
training groups to five models and evaluates 1024 independent nominal and 512 shifted worlds.
Fixed belief PPO scored 603.80 s versus B1 541.06 s on nominal worlds; joint paired
difference interval [26.71,114.17] s. Nominal completions were 4987/5120 learned-model
episodes versus 1024/1024 B1 worlds. Randomised belief training scored 798.62 s.
These support a negative learned-policy result for this budget and simulator, not a
general conclusion against reinforcement learning. Every failure is retained. The
192-world classical cohort suggests a risk-planner mean benefit versus B2, without
establishing superiority to strong B1 or reliable CVaR benefit. Sensors, course,
polar and randomisation are synthetic. Navigation, racing rules, measured polars and
PPO generalisation to new process families remain outside validated use.

M6 exposes the measured categorical distributions of all ten frozen belief models on
three post-evaluation diagnostic worlds. They use the same encoded observation as the
requested deterministic heading; the argmax is checked. Requested and executed targets remain distinct during finite manoeuvres. These values are policy action
probabilities, not calibrated success probabilities, feature attribution or causal intent.
No sensitivity or empirical deployment claim follows from the replay. All model weights
remain local, outside the public application and Git.

## Separate M8 investigation model

Qwen3.5 9B 4-bit MLX, revision `8b2b98c00a6b4d291155e4890773ca8f769aee53`, is used only
for evidence investigation. Model cards declare Apache-2.0; no fine-tuning was performed.
Thinking is disabled and decoding uses temperature zero, at most 700 output / 8192 context
tokens per call, four model turns plus one optional sequential critic. All weights/tokenizer
files are hashed; remote Python code is disabled. API services are unnecessary.

Exact issue detection on twelve held-out synthetic audit cases was 6/12 for the model
and 5/12 with the same-model critic, compared with 12/12 for the comprehensive script.
Unsupported structured claims were rejected and retained. This model does not steer sailing,
compute research metrics, validate a physical polar or establish efficacy. Local MLX peak
allocation was 6.58 GB; this is not total system RAM. See `mcp-investigation.md` for full
scores, raw-output provenance, measured time/tokens, scope failures and sample limitations.


## Recorded-component evidence boundary

Public Jarrett GPS and NOAA buoy observations now validate real-input projection consistency
and a separately fitted, ten-minute observed-wind predictor. They do not fit or update any
sailing controller, feature transform, synthetic polar, manoeuvre parameter or one-second
wind model. The buoy prediction comparison provides no measured sailing-policy benefit.
Full boat-physics calibration remains unsupported without suitably instrumented sessions.
See [recorded methods and results](recorded-validation.md).

## Registered classical work/new-process evaluation

Two-path expected/risk B3 retain 60 s horizons; B2's 130 s horizon is selected by
preaccess validation under a declared work constraint. Original tuned B1 and original
eight-path risk remain visible. Cohort-average integration requests per decision match
within ±10% in both primary pools and all six families; statewise/runtime equality is
not claimed. Two-path risk is worse than B1 in both pools, and worse than matched
expected-cost B3 on unseen processes. Matched B2 contrasts remain uncertain. Earlier
eight-path objective uncertainty is retained without pooling. Classical controllers alone
were tested on gust-front/persistent-trend worlds; PPO was neither retrained nor evaluated
there. This is synthetic research evidence, not calibrated boat deployment. No numerical
model or learned checkpoint changed after locked access. See compute-followup.md.
