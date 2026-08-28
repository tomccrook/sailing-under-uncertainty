# Locked evaluation registration

The next evaluation separates three questions: can the selected PPO representation
generalise to independent nominal worlds, does domain randomisation help under
declared shifts, and does the classical risk planner improve on its mean-planning
counterparts? Selection uses M4 validation only, averaged over three seeds. Two
additional seeds extend the selected fixed and randomised representations to five
models each without selecting a favourable individual seed.

Nominal roots 300000–339999 occupy four family blocks of 10000 roots. The planned
nominal cohort is 256 worlds per family (1024 total, about 51 independent 95%-tail
worlds overall and 12.8 per family). Shift roots start at 400000, with independent
10000-root blocks per named slice, 32 worlds per slice. All learned models, B1 raw,
the tuned unrestricted belief B1, and its separately tuned command-projected
version share every world and instrument stream. Model IDs, transforms, baseline
parameters, action semantics, failure score, scenario configs, source, dependencies
and analysis procedure are hashed before evaluation.

The synthetic shift matrix varies one interpretable feature at a time: slower
direction persistence, stronger speed gusts, faster/larger regime shifts, shorter
spatial correlation, eastward current, longer sensor delay, more dropout, biased
instruments, slower polar, larger no-go angle, longer tacks, slower recovery,
overconfident forecasts, longer course, perfect instruments, and an idealised
fast-turn/fast-recovery boat. These are extrapolation or observation-failure
checks; the last two are instrument/dynamics ablations. The coupled no-go-table
adjustment is disclosed in A13. No scenario is claimed to represent a measured
distribution of dinghy performance. New-family testing is not included in this
registration.

M3 measured B3 inference at about 0.10 s per decision, versus 0.014 s for B2 and
0.00005 s for B1. A 1000-world, two-B3 comparison is therefore a substantially
larger workload than the PPO/B1 evaluation. The initial locked classical cohort
is 32 nominal worlds per family and four worlds per shift slice (192 total).
This subset is declared before results are read; it is not selected by outcome.
B2 mean, B3 expected cost and B3 risk use their existing validation-frozen M3
parameters and the same finite dynamics/actions. They retain the strongest B1
reference. This cohort has only 9.6 overall effective tail worlds and 6.4 nominal
tail worlds: **it cannot support a stable-tail claim or close the principal H2
question by itself**. Expansion requires a new predeclared independent cohort;
no revision may tune on this cohort. M5's final precision gate remains open if
uncertainty is unresolved. The nominal PPO cohort reaches the provisional overall
sample target, but seed/world resampling and per-family uncertainty still apply.

Local lock files and an append-only access log are process controls, not security
against a repository owner. Evaluation verifies hashes and disjoint roots against
the actual model training manifests and M4 validation manifest. It rejects model,
transform, parameter, source or manifest changes, repeats in the same output, and
partial/missing outcomes. Numeric faults abort and preserve completed rows, errors
and the lock. Sailing failures are scored, never dropped.

Report paired world-bootstrap intervals conditional on each trained model, plus
joint training-seed/world bootstrap intervals for learned groups. Between-training
SD is separate. Nominal, shifted and family/slice outcomes are separately reported;
an aggregate across this deliberately constructed mixture has no empirical-weather
frequency interpretation. Costs, completion/failure categories, deadline misses,
tacks, distance, phase duration, command switches, measured inference time,
estimator diagnostics and penalty sensitivity accompany intervals. Worst cases are
re-run only for diagnostic traces, with outcome equality checked. They never feed
model selection. Training curves remain a separately labelled diagnostic.

Preaccess review found that the generic source hash included README text. The
numerical verifier now hashes Python source and dependency inputs separately,
allowing later report documentation updates. The original prepared registration
is preserved as `original-lock.json` with its source snapshot; the replacement
records the ancestor hash, reason and access-log event before evaluation starts.
This same amendment registers paired CVaR-difference intervals and nominal-family
slices. Policies, model IDs, scenarios and sample sizes do not change. The runner
prohibits this amendment after any evaluation-start event or evaluation directory.
