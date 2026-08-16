# Sources and implementation references

The full source list is brief §25. No external sailing measurements have been ingested.
Implementation documentation checked on 2026-10-03:

- [uv project environments/lockfiles](https://docs.astral.sh/uv/guides/projects/)
- [Pydantic model validation](https://docs.pydantic.dev/latest/concepts/models/)
- [Gymnasium termination and reset contract](https://gymnasium.farama.org/api/env/)
- [Stable-Baselines3 PPO API](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), [checkpoint format](https://stable-baselines3.readthedocs.io/en/master/guide/save_format.html)
- [PPO original paper](https://arxiv.org/abs/1707.06347), [PyTorch MPS capabilities](https://docs.pytorch.org/docs/main/notes/mps.html)
- [Vite runtime requirements](https://vite.dev/guide/)
- [GitHub Python setup](https://github.com/actions/setup-python), [Node setup](https://github.com/actions/setup-node), [checkout](https://github.com/actions/checkout)

Exact installed dependency versions are resolver-produced in `uv.lock` and `web/package-lock.json`.
Resolve additional libraries only when their milestone needs them. Synthetic polar data are original
illustrative inputs, not a sourced boat-class table.
