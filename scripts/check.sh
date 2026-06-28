#!/bin/sh
set -eu
uv run ruff check .
uv run ruff format --check .
uv run mypy src scripts
uv run python -m pytest
uv run sail-lab validate-config configs/experiments/smoke.yaml
uv run sail-lab validate-simulator
uv run sail-lab demo
cd web
npm run lint
npm run format:check
npm run build
npm test
