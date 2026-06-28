"""Generate Python/TypeScript contract inputs; run before npm generate-types."""

from pathlib import Path
from typing import Any

from sail_lab.config import ExperimentConfig
from sail_lab.contracts import ReplayBundle
from sail_lab.experiments import RunManifest, write_json

root = Path(__file__).resolve().parents[1]
(root / "schemas").mkdir(exist_ok=True)
for name, model in (
    ("experiment", ExperimentConfig),
    ("replay", ReplayBundle),
    ("run-manifest", RunManifest),
):
    schema = model.model_json_schema()
    write_json(root / "schemas" / f"{name}.schema.json", schema)
    if name == "replay":
        # json-schema-to-typescript uses draft-7 tuples, not 2020-12 prefixItems.
        def draft7(value: Any) -> Any:
            if isinstance(value, dict):
                converted = {
                    key: draft7(item) for key, item in value.items() if key != "prefixItems"
                }
                if "prefixItems" in value:
                    converted["items"] = draft7(value["prefixItems"])
                    converted["additionalItems"] = False
                return converted
            if isinstance(value, list):
                return [draft7(item) for item in value]
            return value

        write_json(root / "web" / "replay.schema.json", draft7(schema))
