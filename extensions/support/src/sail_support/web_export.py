"""Explicit fixture publication and shared schema generation, never private track publication."""

import json
from pathlib import Path
from shutil import copyfile
from typing import Any

from sail_support.calibration import TimingReport
from sail_support.contracts import Track
from sail_support.weather import WeatherPacket


def _draft7(value: Any) -> Any:
    if isinstance(value, dict):
        converted = {key: _draft7(item) for key, item in value.items() if key != "prefixItems"}
        if "prefixItems" in value:
            converted["items"] = _draft7(value["prefixItems"])
            converted["additionalItems"] = False
        return converted
    if isinstance(value, list):
        return [_draft7(item) for item in value]
    return value


def export_schemas(root: Path) -> None:
    for name, model in (("track", Track), ("weather", WeatherPacket), ("timing", TimingReport)):
        schema = model.model_json_schema()
        (root / "schemas" / f"{name}.schema.json").write_text(json.dumps(schema, indent=2) + "\n")
        (root / "web" / f"{name}.schema.json").write_text(
            json.dumps(_draft7(schema), indent=2) + "\n"
        )


def export_fixture(track_path: Path, timing_path: Path, weather_path: Path, output: Path) -> None:
    track = Track.model_validate_json(track_path.read_text())
    timing = TimingReport.model_validate_json(timing_path.read_text())
    weather = WeatherPacket.model_validate_json(weather_path.read_text())
    if (
        track.evidence != "synthetic_fixture"
        or timing.evidence != "synthetic_fixture"
        or weather.source_evidence != "synthetic_fixture"
    ):
        raise ValueError("Only explicitly synthetic support fixtures may be published by this tool")
    if track.source_sha256 not in timing.source_hashes:
        raise ValueError("Displayed track is not a source of this timing fixture report")
    output.mkdir(parents=True, exist_ok=False)
    for name, path in (("track", track_path), ("timing", timing_path), ("weather", weather_path)):
        copyfile(path, output / f"{name}.json")
