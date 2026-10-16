"""Instrument-supported heading-transition timing; never infer a polar from GPS alone."""

import hashlib
from datetime import date, datetime
from typing import Literal

import numpy as np
from pydantic import Field

from sail_support.contracts import Contract, Track, TrackPoint


class TimingConfig(Contract):
    angular_rate_threshold_deg_s: float = Field(default=2, gt=0)
    stable_samples: int = Field(default=3, ge=2)
    min_upwind_angle_deg: float = Field(default=35, gt=0, lt=90)
    max_upwind_angle_deg: float = Field(default=85, gt=0, lt=90)
    max_wind_change_deg: float = Field(default=10, gt=0)
    bootstrap_seed: int = Field(default=27001, ge=0)
    bootstrap_replicates: int = Field(default=2000, ge=100, le=100_000)


DEFAULT_TIMING_CONFIG = TimingConfig()


class Transition(Contract):
    session_id: str
    source_sha256: str
    day: date
    start_utc: datetime
    end_utc: datetime
    duration_s: float = Field(gt=0)
    sampling_uncertainty_s: float = Field(gt=0)


class TimingReport(Contract):
    schema_version: Literal["timing-v1"] = "timing-v1"
    evidence: Literal["synthetic_fixture", "unvalidated_measurement"]
    quantity: Literal["observed instrumented upwind heading-transition duration"] = (
        "observed instrumented upwind heading-transition duration"
    )
    config: TimingConfig
    protocol: str = "Chronological UTC-day groups; latest two days held out before fitting"
    source_hashes: tuple[str, ...]
    split_sha256: str
    training_days: tuple[date, ...]
    holdout_days: tuple[date, ...]
    transitions: tuple[Transition, ...]
    fitted_mean_s: float
    fitted_mean_day_bootstrap_95_s: tuple[float, float]
    holdout_mae_s: float
    holdout_mae_day_bootstrap_95_s: tuple[float, float]
    unsupported: tuple[str, ...] = (
        "polar calibration",
        "minimum tack time",
        "speed-loss calibration",
        "policy efficacy",
    )
    limitations: str = (
        "Small independent-day counts; sampling bounds are timing resolution, not a confidence "
        "interval. Instrument synchronisation and current/wind conventions require domain review."
    )


def _wrap(value: float) -> float:
    return (value + 180) % 360 - 180


def _relative(point: TrackPoint) -> float | None:
    import math

    values = (
        point.heading_deg,
        point.wind_from_deg,
        point.wind_speed_m_s,
        point.current_east_m_s,
        point.current_north_m_s,
        point.speed_water_m_s,
    )
    if any(value is None for value in values):
        return None
    heading, wind_from, wind_speed, east, north, _ = values
    assert heading is not None and wind_from is not None and wind_speed is not None
    assert east is not None and north is not None
    wx = -wind_speed * math.sin(math.radians(wind_from)) - east
    wy = -wind_speed * math.cos(math.radians(wind_from)) - north
    if math.hypot(wx, wy) < 0.1:
        return None
    relative_from = math.degrees(math.atan2(-wx, -wy)) % 360
    return _wrap(heading - relative_from)


def transitions(
    track: Track, config: TimingConfig = DEFAULT_TIMING_CONFIG
) -> tuple[Transition, ...]:
    found: list[Transition] = []
    previous: TrackPoint | None = None
    start: TrackPoint | None = None
    end: TrackPoint | None = None
    stable = 0
    moving = False
    max_dt = 0.0
    wind_change = 0.0
    for point in track.points:
        angle = _relative(point)
        if not point.accepted or angle is None or point.timestamp_utc is None:
            previous = start = end = None
            moving = False
            stable = 0
            continue
        if previous is None or previous.segment != point.segment:
            previous = point
            start = end = None
            moving = False
            stable = 0
            continue
        assert previous.timestamp_utc is not None
        assert previous.heading_deg is not None and point.heading_deg is not None
        dt = (point.timestamp_utc - previous.timestamp_utc).total_seconds()
        rate = abs(_wrap(point.heading_deg - previous.heading_deg)) / dt
        if rate > config.angular_rate_threshold_deg_s:
            if not moving and stable >= config.stable_samples:
                start = previous
                moving = True
                max_dt = dt
                wind_change = 0
            if moving:
                end = point
                max_dt = max(max_dt, dt)
                assert start is not None and start.wind_from_deg is not None
                assert point.wind_from_deg is not None
                wind_change = max(
                    wind_change, abs(_wrap(point.wind_from_deg - start.wind_from_deg))
                )
            stable = 0
        else:
            stable += 1
            if moving and stable >= config.stable_samples:
                assert start is not None and end is not None
                a, b = _relative(start), _relative(end)
                assert a is not None and b is not None
                assert start.timestamp_utc is not None and end.timestamp_utc is not None
                if (
                    a * b < 0
                    and config.min_upwind_angle_deg <= abs(a) <= config.max_upwind_angle_deg
                    and config.min_upwind_angle_deg <= abs(b) <= config.max_upwind_angle_deg
                    and wind_change <= config.max_wind_change_deg
                    and start.timestamp_utc.date() == end.timestamp_utc.date()
                ):
                    found.append(
                        Transition(
                            session_id=track.session_id,
                            source_sha256=track.source_sha256,
                            day=start.timestamp_utc.date(),
                            start_utc=start.timestamp_utc,
                            end_utc=end.timestamp_utc,
                            duration_s=(end.timestamp_utc - start.timestamp_utc).total_seconds(),
                            sampling_uncertainty_s=2 * max_dt,
                        )
                    )
                moving = False
                start = end = None
        previous = point
    return tuple(found)


def _interval(groups: list[list[float]], seed: int, replicates: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.array(
        [
            np.mean(
                [
                    value
                    for index in rng.integers(0, len(groups), len(groups))
                    for value in groups[int(index)]
                ]
            )
            for _ in range(replicates)
        ]
    )
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def fit_timing(
    tracks: tuple[Track, ...], config: TimingConfig = DEFAULT_TIMING_CONFIG
) -> TimingReport:
    if len({track.source_sha256 for track in tracks}) != len(tracks):
        raise ValueError("Duplicate raw sources cannot be independent calibration sessions")
    if len({track.session_id for track in tracks}) != len(tracks):
        raise ValueError("Session identifiers must be unique")
    evidence = {track.evidence for track in tracks}
    if len(evidence) != 1:
        raise ValueError("Synthetic fixtures and measurements must be fitted separately")
    # Define groups from all accepted input times, before looking for transition outcomes.
    days: dict[date, list[Track]] = {}
    for track in tracks:
        track_days = {
            p.timestamp_utc.date()
            for p in track.points
            if p.accepted and p.timestamp_utc is not None
        }
        if len(track_days) != 1:
            raise ValueError(
                "Each session must have one UTC day; split multi-day tracks explicitly"
            )
        days.setdefault(next(iter(track_days)), []).append(track)
    ordered = sorted(days)
    if len(ordered) < 6:
        raise ValueError("Timing fixture protocol requires at least six independent UTC days")
    training_days, holdout_days = ordered[:-2], ordered[-2:]
    events = tuple(event for track in tracks for event in transitions(track, config))
    groups = {day: [e.duration_s for e in events if e.day == day] for day in ordered}
    if any(not values for values in groups.values()):
        raise ValueError(
            "Every declared day needs instrument-supported transitions; no day dropped"
        )
    train_values = [value for day in training_days for value in groups[day]]
    mean = float(np.mean(train_values))
    errors = [[abs(value - mean) for value in groups[day]] for day in holdout_days]
    split = "\n".join(
        f"{track.source_sha256}:{day}:{'holdout' if day in holdout_days else 'training'}"
        for day in ordered
        for track in sorted(days[day], key=lambda t: t.source_sha256)
    )
    return TimingReport(
        evidence=tracks[0].evidence,
        config=config,
        source_hashes=tuple(track.source_sha256 for track in tracks),
        split_sha256=hashlib.sha256(split.encode()).hexdigest(),
        training_days=tuple(training_days),
        holdout_days=tuple(holdout_days),
        transitions=events,
        fitted_mean_s=mean,
        fitted_mean_day_bootstrap_95_s=_interval(
            [groups[day] for day in training_days],
            config.bootstrap_seed,
            config.bootstrap_replicates,
        ),
        holdout_mae_s=float(np.mean([value for group in errors for value in group])),
        holdout_mae_day_bootstrap_95_s=_interval(
            errors,
            config.bootstrap_seed + 1,
            config.bootstrap_replicates,
        ),
    )
