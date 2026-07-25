from datetime import UTC, datetime
from pathlib import Path

import pytest

from sail_support.fixtures import make_measurement_fixture, synthetic_csv
from sail_support.tracks import import_track
from sail_support.web_export import export_fixture


def test_fixture_publication_cannot_publish_a_private_measurement(tmp_path: Path) -> None:
    source = make_measurement_fixture(tmp_path / "fixture")
    export_fixture(
        source / "synthetic-day-1/track.json",
        source / "timing.json",
        source / "weather.json",
        tmp_path / "public",
    )
    assert len(list((tmp_path / "public").iterdir())) == 3
    private = import_track(
        (source / "synthetic-day-1/source.csv").read_bytes(),
        "csv",
        session_id="private",
        imported_utc=datetime.now(UTC),
    )
    (tmp_path / "private.json").write_text(private.model_dump_json())
    with pytest.raises(ValueError, match="synthetic"):
        export_fixture(
            tmp_path / "private.json",
            source / "timing.json",
            source / "weather.json",
            tmp_path / "unsafe-public",
        )
    assert not (tmp_path / "unsafe-public").exists()
    unrelated = import_track(
        synthetic_csv(1, 11), "csv", session_id="other-fixture", evidence="synthetic_fixture"
    )
    (tmp_path / "other.json").write_text(unrelated.model_dump_json())
    with pytest.raises(ValueError, match="not a source"):
        export_fixture(
            tmp_path / "other.json",
            source / "timing.json",
            source / "weather.json",
            tmp_path / "unrelated-public",
        )
    assert not (tmp_path / "unrelated-public").exists()
