"""Reproducible private stdio server, fixtures, model setup and bounded evaluation."""

import argparse
import asyncio
import json
from pathlib import Path

from sail_research.client import call, connect
from sail_research.contracts import Budget
from sail_research.demo import demonstrate
from sail_research.evaluation import evaluate
from sail_research.fixtures import create_suite
from sail_research.models import download_model
from sail_research.publication import publish_evaluation
from sail_research.server import serve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "serve",
            "fixtures",
            "call",
            "download-model",
            "evaluate",
            "demonstrate",
            "publish",
        ),
    )
    parser.add_argument("--root", type=Path, default=Path("artifacts/investigations/local"))
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--budget", type=Path, default=Path("configs/research/local.json"))
    parser.add_argument("--tool")
    parser.add_argument("--arguments", default="{}")
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--model-root", type=Path, default=Path("artifacts/models/m8"))
    parser.add_argument("--split", choices=("development", "evaluation"), default="evaluation")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "fixtures":
        print(json.dumps(create_suite(args.root)))
    elif args.command == "download-model":
        print(json.dumps(download_model(args.root)))
    elif args.command == "serve":
        serve(args.root, args.project, Budget.model_validate_json(args.budget.read_text()))
    elif args.command == "evaluate":
        if args.suite is None:
            parser.error("Registered suite required")
        result = asyncio.run(
            evaluate(args.suite, args.root, args.project, args.model_root, args.split)
        )
        print(json.dumps(result["strategies"]))
    elif args.command == "demonstrate":
        result = asyncio.run(demonstrate(args.root, args.project, args.model_root))
        print(json.dumps({"status": result["status"], "model_choice": result.get("model_choice")}))
    elif args.command == "publish":
        if args.run is None or args.output is None:
            parser.error("Complete run and new output file required")
        publish_evaluation(args.run, args.output)
    else:
        if args.tool is None:
            parser.error("Tool name required")

        async def invoke() -> None:
            async with connect(args.root, args.project, args.budget) as client:
                reply = await call(client, args.tool, json.loads(args.arguments))
                print(
                    json.dumps(
                        {
                            "protocol_version": client.protocol_version,
                            "result": reply.model_dump(mode="json"),
                        }
                    )
                )

        asyncio.run(invoke())


if __name__ == "__main__":
    main()
