import asyncio
from pathlib import Path

from sail_research.client import call, connect
from sail_research.contracts import Budget
from sail_research.desk import INPUTS
from sail_research.fixtures import create_suite
from sail_research.workflow import fixed_workflow

PROJECT = Path(__file__).resolve().parents[3]


def test_real_stdio_negotiation_schemas_and_evidence(tmp_path: Path) -> None:
    manifest = create_suite(tmp_path / "suite")
    budget = Budget(max_tool_calls=1000, max_jobs=0, max_reserved_wall_s=0, max_episode_tasks=0)
    (tmp_path / "budget.json").write_text(budget.model_dump_json())

    async def check() -> None:
        async with connect(tmp_path / "suite/desk", PROJECT, tmp_path / "budget.json") as client:
            tools = await client.list_tools()
            assert {tool.name for tool in tools.tools} == set(INPUTS)
            assert all(tool.output_schema is not None for tool in tools.tools)
            assert client.protocol_version in {"2026-07-28", "2025-11-25"}
            task = manifest["tasks"][0]  # type: ignore[index]
            result = await fixed_workflow(client, task)
            assert result["status"] == "succeeded"
            assert result["report"]["payload"]["server_verified_metrics"]["worlds"] == 16
            denied = await call(client, "submit_bounded_experiment", {"variation": "baseline"})
            assert not denied.ok and "budget" in str(denied.error)
            unsafe = await call(client, "get_run_manifest", {"experiment_id": "../../labels"})
            assert not unsafe.ok

    asyncio.run(check())
