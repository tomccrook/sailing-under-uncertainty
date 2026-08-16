# Architecture and information contracts

`config.py` validates resolved SI parameters. `geometry.py` fixes conventions and event geometry.
`dynamics/polar.py` owns the original synthetic table; `dynamics/simulator.py` owns mutable truth
and a single finite-motion implementation. `contracts.py` separates immutable policy observations
from offline replay diagnostics. `policies.py` receives observations through reset/act only.

M1 assumes perfect local instruments and explicitly known uniform current. It is an engineering
fixture, not the practical partial-observation experiment. The policy does not receive Simulator,
BoatState, diagnostic samples, future wind or a world RNG. Offline recording happens outside act().
M2 implements fixed acquisition clocks, delayed delivery masks, immutable forecast packets and
a vector filter. `world.py` freezes exogenous numerical arrays; `streams.py` keys sensor and
forecast innovations independently of route/call order. `observations.py` owns instrument
delivery and inference. `diagnostics.py` scores truth offline; the publisher queries only
the fixed station at issue time. Forecasts describe conditional predictions, not realised futures.

Heading is clockwise from north; port/starboard symmetry is represented by reflecting east and
heading sign in north-wind/no-current fixtures. Polar uses water-relative wind, ground drift is added
once, and apparent wind is diagnostic only. A no-go angle can stop acceleration but does not erase
existing momentum instantaneously.

Midpoint heading approximates curved motion; target speed is held at that heading during each step.
Speed and travelled water distance are integrated analytically for that constant target. Terminal
events use the earliest intersection of the approximated segment, then interpolate the terminal
time and state. This is an integrator approximation, covered by timestep convergence, not exact
continuous curved-path collision solving. No boundary clipping/rescue occurs.

A tack is a commanded upwind side change. It latches the executed target, turns at a finite rate
over max(minimum duration, required turn time), and is counted only on completed opposite-side
arrival. Requested and executed targets can differ during a tack. Gybes are not tacks. Turns
through the no-go region reduce the natural polar target; speed response causes recovery loss.
No separate artificial tack multiplier is applied. Modelled manoeuvre time is tracked explicitly.

The ideal B0 45°/3 m/s point-target fixture is a separate lossless numerical model. It cannot be
used as the expected realistic polar time. B1 chooses the polar's VMG optimum and estimates
two-leg layline times from observed wind/current, with fixture-only switch margin/commitment.
M3 tunes this baseline and adds `planning.py`: conditional paths, a shared immediate
candidate set, finite-simulator hypothetical rollouts and approximate causal recourse.
`classical_study.py` owns validation searches, freezing, paired scoring and retained faults.

`rl_env.py` adapts the same simulator to Gymnasium. `FeatureEncoder` accepts only
`Observation`; present/history cannot receive belief via features or action semantics.
Episode/session metadata stay in recorder objects and `info` is empty. `training.py`
owns strict typed budgets/device checks, SB3 PPO, checkpoints and immutable trial evidence.
`learning_study.py` owns registered memory/randomisation comparisons, separately tuned
unrestricted/projected B1, compact evaluation and seed/world bootstrap. Compact evaluation
matches full recorded simulation outcomes in integration tests.

`locked_evaluation.py` freezes actual model/dependency/scenario/source identities before
independent cohorts are accessed. Four worker processes evaluate complete paired worlds;
work submission is bounded and implementation faults preserve partial evidence and abort.
Source hashes distinguish numerical inputs from later README/report updates. Access-log and
hash controls enforce the declared workflow locally, not security against an owner.
Worst-case replays re-run the frozen controller and must match the locked outcome exactly.

`presentation.py` and the learning exporter publish verified static JSON separately from
training/evaluation. React validates artefact contracts and distinguishes development results,
locked outcomes and exploratory training curves. It offers no job-submission endpoint.
Neither the simulator nor CLI requires the browser, cloud or an assistant to execute.

M6 publication lives in repository-local `scripts/`: typed catalogue/decision contracts,
exact frozen-policy reproduction and read-only verified exports. This keeps the numerical
M5 identity intact. Schema-derived TypeScript plus browser SHA-256 validation bind selected
replay/detail files to their catalogue. The same canvas draws both recorded boats; physics
remains exclusively Python. Abortable loading and a shared elapsed clock prevent stale
policy/world responses from replacing the current selection. No execution endpoint is
exposed. All outcome rows and existing scientific figures are exported with provenance;
MPC conditional scores, PPO probabilities and empirical uncertainty retain distinct labels.
