"""Generate the shared Python/TypeScript presentation contracts."""

from pathlib import Path
from typing import Any

from sail_lab.experiments import write_json
from scripts.demo_contracts import Catalogue, Details


def draft7(value: Any) -> Any:
    if isinstance(value, dict):
        converted = {k: draft7(v) for k, v in value.items() if k != "prefixItems"}
        if "prefixItems" in value:
            converted["items"] = draft7(value["prefixItems"])
            converted["additionalItems"] = False
        return converted
    if isinstance(value, list):
        return [draft7(v) for v in value]
    return value


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for name, model in (("catalogue", Catalogue), ("details", Details)):
        schema = model.model_json_schema()
        write_json(root / f"schemas/{name}.schema.json", schema)
        write_json(root / f"web/{name}.schema.json", draft7(schema))


if __name__ == "__main__":
    main()
