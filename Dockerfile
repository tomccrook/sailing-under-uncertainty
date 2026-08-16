FROM python:3.12-slim
WORKDIR /app
RUN python -m pip install --no-cache-dir uv==0.12.22
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs
RUN uv sync --locked --no-dev --no-editable --cache-dir /tmp/uv-cache
ENV PATH="/app/.venv/bin:$PATH" MPLCONFIGDIR="/tmp/matplotlib"
CMD ["sail-lab", "run", "--output", "/app/artifacts/runs"]
