import json
from datetime import UTC, datetime, timedelta

import pytest

from sail_support.weather import WeatherRequest, available_forecast, parse_weather, request_url

RUN = datetime(2026, 6, 1, tzinfo=UTC)


def request(kind: str = "single_run") -> WeatherRequest:
    return WeatherRequest.model_validate(
        {
            "kind": kind,
            "latitude_deg": 50.7,
            "longitude_deg": -1.3,
            "model": "ecmwf_ifs",
            "declared_spatial_resolution_m": 9000,
            "run_initialisation_utc": RUN if kind == "single_run" else None,
        }
    )


def payload(suffix: str = "") -> bytes:
    return json.dumps(
        {
            "latitude": 50.75,
            "longitude": -1.25,
            "timezone": "GMT",
            "utc_offset_seconds": 0,
            "hourly_units": {f"wind_speed_10m{suffix}": "km/h", f"wind_direction_10m{suffix}": "°"},
            "hourly": {
                "time": ["2026-06-01T08:00", "2026-06-01T09:00", "2026-06-01T10:00"],
                f"wind_speed_10m{suffix}": [21.6, 18, None],
                f"wind_direction_10m{suffix}": [360, 90, None],
            },
        }
    ).encode()


def test_initialisation_is_not_availability_and_units_are_explicit() -> None:
    packet = parse_weather(
        payload(),
        request(),
        source_evidence="synthetic_fixture",
        retrieved_utc=RUN + timedelta(days=100),
    )
    assert packet.available_utc == RUN + timedelta(hours=8)
    assert packet.availability == "assumed_lag"
    with pytest.raises(ValueError, match="not yet available"):
        available_forecast(packet, RUN + timedelta(hours=7, minutes=59))
    points = available_forecast(packet, RUN + timedelta(hours=8))
    assert len(points) == 2
    assert points[0].wind_speed_m_s == pytest.approx(6)
    assert points[0].wind_from_deg == 0
    assert len(available_forecast(packet, RUN + timedelta(hours=8, minutes=1))) == 1
    assert "run=2026-06-01T00%3A00" in request_url(request())
    assert packet.returned_latitude_deg != packet.request.latitude_deg


@pytest.mark.parametrize("kind", ["reanalysis", "historical_forecast", "previous_runs"])
def test_retrospective_and_stitched_products_are_never_historical_information(kind: str) -> None:
    suffix = "_previous_day1" if kind == "previous_runs" else ""
    packet = parse_weather(
        payload(suffix), request(kind), source_evidence="synthetic_fixture", retrieved_utc=RUN
    )
    assert packet.availability == "context_only" and packet.available_utc is None
    with pytest.raises(ValueError, match="Context"):
        available_forecast(packet, RUN + timedelta(days=1))


def test_declared_known_availability_needs_evidence_and_cannot_predate_run() -> None:
    data = request().model_dump()
    data["known_available_utc"] = RUN + timedelta(hours=5)
    with pytest.raises(ValueError, match="provenance"):
        WeatherRequest.model_validate(data)
    data["availability_evidence"] = "test fixture publisher log"
    config = WeatherRequest.model_validate(data)
    packet = parse_weather(
        payload(), config, source_evidence="synthetic_fixture", retrieved_utc=RUN
    )
    assert packet.availability == "known"
    assert packet.available_utc == data["known_available_utc"]
    forged = packet.model_dump()
    forged["available_utc"] = RUN
    with pytest.raises(ValueError, match="does not match"):
        type(packet).model_validate(forged)
    data["known_available_utc"] = RUN - timedelta(hours=1)
    with pytest.raises(ValueError, match="before"):
        WeatherRequest.model_validate(data)


@pytest.mark.parametrize("change", ["timezone", "length", "time_order", "nan", "unit"])
def test_malformed_weather_is_not_silently_repaired(change: str) -> None:
    data = json.loads(payload())
    if change == "timezone":
        data["timezone"] = "Europe/London"
    elif change == "length":
        data["hourly"]["wind_speed_10m"].pop()
    elif change == "time_order":
        data["hourly"]["time"][1] = data["hourly"]["time"][0]
    elif change == "nan":
        data["hourly"]["wind_speed_10m"][0] = float("nan")
    else:
        data["hourly_units"]["wind_speed_10m"] = "unknown"
    with pytest.raises(ValueError):
        parse_weather(
            json.dumps(data).encode(),
            request(),
            source_evidence="synthetic_fixture",
            retrieved_utc=RUN,
        )
