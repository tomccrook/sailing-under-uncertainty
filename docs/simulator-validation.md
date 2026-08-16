# Deterministic simulator validation — M1

Evidence level: **synthetic**. This validates numerical assumptions; it does not calibrate a
Mirror or answer H2. All measurements below come from commands in this repository.

## Numerical models

The independent ideal fixture uses north wind, no current, headings ±45°, constant 3 m/s,
zero turn/acceleration loss and a point target at (0,1000). Analytical arrival is
471.404520791 s, distance 1414.213562373 m and maximum lateral displacement 500 m.
Event-split integration at 0.1, 1, 7 and 60 s matches to 1e-8 absolute tolerance.

The finite model uses the original illustrative polar in `dynamics/polar.py`. At 6 m/s wind,
its optimal upwind angle is 50° with 3.2 m/s through water. This is separate from the 45°
fixture. The lossless two-heading/circle reference derives distance d/cos(a)−r, time distance/v,
and first-leg length (d/cos(a)−2r)/2. Tests check its endpoint on the appropriate finish circle.
The realistic model includes acceleration, finite heading change and recovery and should take
longer than that corresponding lossless reference; the two models are not forced to match.

## Recorded convergence

Command: `uv run sail-lab validate-simulator`. Artefact: `artifacts/validation/m1.json`.
Decisions remain every 10 s and instruments every 2 s when reducing the physics step.
Provisional gate declared before the run: finish-time differences below 1%, same completion
and tack count, north-wind reflection symmetry within 1e-6 s.

| Physics step (s) | Finish time (s) | Distance over ground (m) | Completed tacks | Manoeuvre time (s) |
|---|---:|---:|---:|---:|
| 1.00 | 497.541222 | 1528.204008 | 1 | 12.5 |
| 0.50 | 497.523286 | 1528.232992 | 1 | 12.5 |
| 0.25 | 497.601427 | 1528.219422 | 1 | 12.5 |

All finished. Maximum relative difference from the finest run is 0.015704%; reflected routes
have identical finish time. This fixture's convergence is not universal convergence evidence.
New wind families, control rates or high currents require corresponding checks.

## Fixture-only B1 tuning

Command: `uv run sail-lab tune-fixture`; all trials have manifests and complete trajectories.
The direct-approach threshold is 100 m, commitment 20 s, decisions every 10 s.
Cost is finish time on success; deadline 1800 + penalty 600 s for sailing failures.

| Layline margin (s) | Status | Elapsed (s) | Cost (s) | Completed tacks |
|---|---|---:|---:|---:|
| 0 | boundary violation | 249.763867 | 2400 | 0 |
| 5 | boundary violation | 249.763867 | 2400 | 0 |
| 9 | finished | 497.541222 | 497.541222 | 1 |
| 12 | finished | 497.541222 | 497.541222 | 1 |
| 20 | finished | 510.679752 | 510.679752 | 3 |

The smaller tied successful margin, 9 s, is retained. Narrow lateral bounds punish switching
late; an early margin can create repeated manoeuvres. This is engineering selection on one
fixture, not proof that B1 is competitive across stochastic worlds. M3 must tune it seriously
on validation worlds and record compute/action-set matching.

## Faults and checks

Numerical tests cover cardinal conventions, angle wrap, knots, current-relative polar speed,
zero wind, polar boundaries/no-go continuity, acceleration integral, finite turn rate, tack latch,
minimum duration, recovery, circle crossing, boundary exit and deadline handling.
Integration tests check deterministic reproduction, manifest hashes, Parquet/replay/summary
agreement, preserved sailing failures and injected implementation faults. Future measurement
timestamps and inconsistent replay terminal records are rejected.

The first complete route revealed an unnecessary near-finish tack; B1 now uses a sailable direct
approach on its current side within 100 m. The regression test requires exactly one completed
tack, 12.5 s manoeuvre time, and no active terminal manoeuvre for the default fixture.

Browser tests run on strict dedicated ports, cover desktop/mobile, play/pause/scrub/reset,
keyboard input, truth labelling, downloads and missing/malformed records. The initial browser
test used an unrelated server on 5173; fixed ports 5187 (demo) and 5188 (tests) prevent reuse.

## Limits

Motion uses midpoint heading and a constant target speed inside each physics step. Circle
events are exact on the approximated segment; curved-path events and terminal acceleration
time are approximate. No hydrodynamics, waves, racing rules, calibration or real-policy trial
are claimed. Tack losses combine the no-go polar with speed relaxation; minimum duration
limits turn rate rather than imposing another speed penalty. The model is useful for a controlled
decision study, subject to future sensitivity and component validation.
