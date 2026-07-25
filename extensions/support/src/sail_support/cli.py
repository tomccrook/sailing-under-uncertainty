"""Local extension commands; the frozen scientific stack remains independent."""

import argparse
from pathlib import Path
from typing import Literal

from sail_support.calibration import fit_timing
from sail_support.contracts import ImportConfig, Track
from sail_support.fixtures import make_measurement_fixture
from sail_support.tracks import import_track, save_track
from sail_support.weather import WeatherRequest, fetch_weather


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("measurement-fixture")
    fixture.add_argument("--output", type=Path, required=True)
    track = commands.add_parser("import-track")
    track.add_argument("source", type=Path)
    track.add_argument("--session", required=True)
    track.add_argument("--config", type=Path)
    track.add_argument("--device-model")
    track.add_argument("--output", type=Path, required=True)
    timing = commands.add_parser("fit-timing")
    timing.add_argument("tracks", type=Path, nargs="+")
    timing.add_argument("--output", type=Path, required=True)
    weather = commands.add_parser("fetch-weather")
    weather.add_argument("request", type=Path)
    weather.add_argument("--output", type=Path, required=True)
    server = commands.add_parser("serve")
    server.add_argument(
        "--output", type=Path, default=Path("artifacts/measurements/private-imports")
    )
    server.add_argument("--port", type=int, default=8787)
    schemas = commands.add_parser("export-schemas")
    schemas.add_argument("--root", type=Path, default=Path.cwd())
    export = commands.add_parser("export-fixture")
    export.add_argument("--track", type=Path, required=True)
    export.add_argument("--timing", type=Path, required=True)
    export.add_argument("--weather", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    cost = commands.add_parser("cost-plan")
    cost.add_argument("--plan", type=Path, required=True)
    cost.add_argument("--quote", type=Path, required=True)
    cost.add_argument("--output", type=Path, required=True)
    submit = commands.add_parser("submit-cloud")
    submit.add_argument("--config", type=Path, required=True)
    submit.add_argument("--plan", type=Path, required=True)
    submit.add_argument("--quote", type=Path, required=True)
    submit.add_argument("--spec", type=Path, required=True)
    submit.add_argument("--ledger", type=Path, required=True)
    submit.add_argument("--numerical-python", type=Path, default=Path.cwd() / ".venv/bin/python")
    arguments = parser.parse_args()
    if arguments.command == "measurement-fixture":
        print(make_measurement_fixture(arguments.output).resolve())
    elif arguments.command == "import-track":
        source: Path = arguments.source
        if source.suffix.lower() not in {".csv", ".gpx"}:
            parser.error("Source extension must be .csv or .gpx")
        source_format: Literal["csv", "gpx"] = "csv" if source.suffix.lower() == ".csv" else "gpx"
        config = (
            ImportConfig.model_validate_json(arguments.config.read_text())
            if arguments.config
            else ImportConfig()
        )
        raw = source.read_bytes()
        result = import_track(
            raw,
            source_format,
            session_id=arguments.session,
            config=config,
            device_metadata={"model": arguments.device_model} if arguments.device_model else {},
        )
        print(save_track(result, raw, arguments.output).resolve())
    elif arguments.command == "fit-timing":
        report = fit_timing(
            tuple(Track.model_validate_json(path.read_text()) for path in arguments.tracks)
        )
        with arguments.output.open("x") as target:
            target.write(report.model_dump_json(indent=2) + "\n")
        print(arguments.output.resolve())
    elif arguments.command == "fetch-weather":
        request = WeatherRequest.model_validate_json(arguments.request.read_text())
        fetch_weather(request, arguments.output)
        print(arguments.output.resolve())
    elif arguments.command == "serve":
        import uvicorn

        from sail_support.server import create_app

        uvicorn.run(create_app(arguments.output), host="127.0.0.1", port=arguments.port)
    elif arguments.command == "export-schemas":
        from sail_support.web_export import export_schemas

        export_schemas(arguments.root)
    elif arguments.command == "export-fixture":
        from sail_support.web_export import export_fixture

        export_fixture(arguments.track, arguments.timing, arguments.weather, arguments.output)
    elif arguments.command in {"cost-plan", "submit-cloud"}:
        import json
        import subprocess

        from sail_support.cloud import (
            CloudConfig,
            PriceQuote,
            S3Store,
            WorkPlan,
            approved_cost,
            estimate_cost,
            json_hash,
            submit_batch,
        )

        plan = WorkPlan.model_validate_json(arguments.plan.read_text())
        quote = PriceQuote.model_validate_json(arguments.quote.read_text())
        if arguments.command == "cost-plan":
            cost_result = estimate_cost(plan, quote)
            with arguments.output.open("x") as file:
                json.dump(cost_result, file, indent=2, allow_nan=False)
            print(
                f"Planning only: {cost_result['reservation_usd']:.2f} USD reserved estimate; "
                f"authorised spend 0 USD; cost_sha256={json_hash(cost_result)}"
            )
        else:
            cloud_config = CloudConfig.model_validate_json(arguments.config.read_text())
            if not cloud_config.enabled:
                parser.error(
                    "Cloud submission disabled; explicit budget and account access required"
                )
            approved_cost(cloud_config, plan, quote)
            info = json.loads(
                subprocess.check_output(
                    [
                        str(arguments.numerical_python),
                        "-m",
                        "scripts.bounded_jobs",
                        "check-spec",
                        "--spec",
                        str(arguments.spec),
                    ],
                    text=True,
                    timeout=30,
                )
            )
            import boto3
            from botocore.config import Config

            sdk_config = Config(
                retries={"max_attempts": 1, "mode": "standard"}, connect_timeout=5, read_timeout=15
            )
            store = S3Store(
                boto3.client("s3", region_name=cloud_config.region, config=sdk_config),
                cloud_config.artifact_bucket,
            )
            # SubmitJob has no idempotency token: disable SDK retries for ambiguous submissions.
            client = boto3.client(
                "batch",
                region_name=cloud_config.region,
                config=Config(
                    retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=15
                ),
            )
            print(
                json.dumps(
                    submit_batch(
                        cloud_config,
                        plan,
                        quote,
                        job_sha256=info["job_sha256"],
                        spec_raw=arguments.spec.read_bytes(),
                        ledger=arguments.ledger,
                        store=store,
                        client=client,
                    ),
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
