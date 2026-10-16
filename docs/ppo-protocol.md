# Expected-cost PPO protocol

M4 uses Stable-Baselines3 clipped PPO and PyTorch, rather than maintaining a second
optimiser. The locked dependency versions are in `uv.lock`. Networks have separate
64 × 64 tanh actor and value heads. Ordinary PPO optimises expected reward; measured
CVaR does not make this a risk-sensitive training algorithm.

`SailingEnv` advances the existing finite simulator for one 10 s decision interval.
Its Gym observation is a float32 vector of public instruments and geometry; `info`
is empty, including at termination. Deadline and boundary failures are terminal
task outcomes, not time-limit truncations with a bootstrapped continuation value.
Implementation exceptions abort a trial and preserve its manifest/error.

Eight categorical actions mean raw-water-wind source ±45°, ±55°, ±70°, the executed
target, and direct bearing to the finish. All are available: no hidden wind mask.
An attempted no-go heading may be ineffective. Commands have the same meaning for
present, belief and history policies; a belief-dependent command set would leak
filter information into the present-only ablation. This differs from M3's VMG
candidates, so evaluation must include a B1 projected onto these commands alongside
the stronger unrestricted, tuned B1. Parameters randomised in the boat are not
used to construct commands. The direct target is based on known geometry.

Feature names/order/scales are saved with each trial. Distances use 1000 m, air and
water speed 6 m/s, deadline 1800 s, measurement age 30 s, phase duration 30 s,
forecast age 120 s, covariance 4 m²/s². All features clip to ±10. No fitted scaler
exists. Angles use sine/cosine. The forecast summary is the closest available
packet point to a 60 s lead, held at the last prediction beyond its horizon.
Belief adds the two mean components, three unique covariance entries and a mask.
History stacks four present frames, oldest first, with presence masks and zero
padding. Reset clears every frame. No LSTM is added before these simpler
representations have been assessed; this is a consequential scope decision.

With gamma = 1, reward is `−delta_time/100`, plus on sailing failure
`−(2400−elapsed)/100`. Optional shaping adds `Phi(next)−Phi(current)`, where
`Phi = −distance_to_target/200` and terminal Phi is zero. The complete return is
therefore `−cost/100−Phi(initial)`: the policy ordering equals undiscounted cost,
and loops/turns cannot earn shaping profit. GAE lambda = 0.95 trades estimator
bias/variance without changing the declared return. Finite task horizon is at
most 180 decisions, rather than a discounted effective horizon. Tests cover
success, early boundary exit, deadline, telescoping and reset semantics.

Training manifests reserve roots 100000–139999; validation roots 200000–239999,
with separate 10000-root blocks for oscillating, OU, regime and spatial families.
Environment sampling uses its Gym RNG, seeded from the independent optimiser
seed by SB3, to select complete worlds. Sensor/world/forecast streams remain
counter-keyed by each selected root. Trial episode logs name every completed
world/config and training step. Training returns are diagnostics, not final
out-of-sample efficacy measurements.

Fixed-model training uses nominal M3 dynamics/instruments. Randomised training
uses a fixed generated manifest with direction sigma [0.8,1.8] deg/√s, kappa
[0.01,0.04]/s, latency {2,4,6} s, dropout [0,0.15], tack minimum [6,12] s,
polar scale [0.9,1.1], no-go [32,38]° and forecast uncertainty scale [0.8,1.2].
These are synthetic ranges, not measured boat/weather uncertainty. A13's coupled
no-go interpolation adjustment remains applicable. Selection uses validation only.

CPU is mandatory. Requests for unavailable CUDA/MPS fail; an MPS workload rejects
`PYTORCH_ENABLE_MPS_FALLBACK=1`. GPU benchmarks include the same Python simulator,
features, rollouts and PPO updates, with device synchronisation. The first 2048-step
pilot found CPU 1383/1929 decisions/s for one/four Torch threads and MPS 468/s;
CUDA was unavailable. These short, single runs include setup/checkpoint overhead,
so a longer pilot must determine the development budget.

The 32768-step pilot (`artifacts/benchmarks/m4-v2`) measured CPU one/four threads at
1849/1774 decisions/s and MPS at 924/s, including setup and checkpoints. CPU one
thread was selected. CUDA was unavailable. The registered M4 study used twelve
65536-step trials, 786432 training decisions total, and about 469 s wall time for
the whole study including baseline searches and validation. No trials were removed.
Both unrestricted and command-projected B1 searched twelve settings on the same
32 validation worlds. Selected unrestricted B1: margin 5 s, commitment 20 s,
boundary guard 45 s, mean 545.73 s. Belief PPO mean across three seeds: 551.52 s,
paired joint interval for the difference [−24.61,+37.91] s. Present/history means
were 1129.46/615.24 s. These development results justify the selected representation,
not a superiority claim or a decision to cherry-pick seed 71002. Randomised belief
training scored 712.63 s and had larger between-training variation (140.71 s SD of
model mean costs versus 11.04 s for fixed belief). A limited fixed budget may
undertrain a wider distribution; this does not establish that randomisation is
intrinsically harmful. Both training types continue into independent locked tests.

SB3 checkpoints contain policy and Adam optimiser state. Its ordinary save/load
does not preserve live simulator state, rollout buffer or all RNG states. Resuming
is operationally continuous, not an exact interrupted-run replay. CPU replicas
within the current locked environment are tested for identical parameters after
a small trial; this is not a cross-device/version bitwise guarantee. Models and
generated evidence remain outside Git.

Sources checked 2026-10-03: [Gymnasium API](https://gymnasium.farama.org/api/env/),
[SB3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html),
[SB3 save format](https://stable-baselines3.readthedocs.io/en/master/guide/save_format.html),
[PPO paper](https://arxiv.org/abs/1707.06347).
