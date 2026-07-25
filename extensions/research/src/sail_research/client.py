"""Negotiated official MCP client with bounded calls and structured-result validation."""

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from sail_research.desk import ToolReply


@asynccontextmanager
async def connect(root: Path, project: Path, budget_path: Path) -> AsyncIterator[Client]:
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "sail_research.cli",
            "serve",
            "--root",
            str(root.resolve()),
            "--project",
            str(project.resolve()),
            "--budget",
            str(budget_path.resolve()),
        ],
    )
    async with Client(params, mode="auto", read_timeout_seconds=25) as client:
        await client.list_tools()
        yield client


async def call(client: Client, tool: str, arguments: dict[str, Any]) -> ToolReply:
    if tool in {"submit_bounded_experiment", "estimate_job_cost"}:
        arguments = {"request": arguments}
    elif tool == "export_evidence_report":
        arguments = {"draft": arguments}
    response = await client.call_tool(tool, arguments)
    if response.is_error:
        return ToolReply(ok=False, error=str(response.content))
    return ToolReply.model_validate(response.structured_content)
