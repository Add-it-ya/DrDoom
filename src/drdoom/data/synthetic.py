"""Synthetic service metrics with controlled root causes.

This exists alongside the real dataset for the things real data cannot give: known root
cause per event, and a severity dial. It is a controlled experiment harness, not the
headline benchmark, and results from it are reported separately.

Two properties are deliberate:

* Root causes are assigned round-robin rather than sampled, so no split can end up
  missing a class and no tuning objective is blind to part of the label space.
* Severity is drawn across a range that includes near-threshold events. Anomalies large
  enough for any detector to catch make every detector look equally good and tell you
  nothing about which one to ship.
"""

from __future__ import annotations

import numpy as np

from drdoom.data.schema import AnomalyEvent, MetricSeries, event_id_track

SOURCE = "synthetic"
FEATURE_NAMES: list[str] = ["latency_ms", "error_rate_pct", "cpu_pct", "queue_depth"]
ROOT_CAUSES: tuple[str, ...] = (
    "memory_leak",
    "deploy_regression",
    "traffic_spike",
    "downstream_outage",
)

MINUTES_PER_DAY = 24 * 60


def _baseline(n_points: int, rng: np.random.Generator) -> np.ndarray:
    """A daily traffic cycle with per-service baseline, amplitude and noise."""
    steps = np.arange(n_points)
    phase = rng.uniform(0, 2 * np.pi)
    daily = 0.5 + 0.5 * np.sin(2 * np.pi * steps / MINUTES_PER_DAY - np.pi / 2 + phase)

    columns = []
    for base, swing, noise in (
        (rng.uniform(60, 100), 40.0, 4.0),
        (rng.uniform(0.3, 0.7), 0.3, 0.05),
        (rng.uniform(20, 40), 30.0, 3.0),
        (rng.uniform(3, 8), 10.0, 1.5),
    ):
        columns.append(base + swing * daily + rng.normal(0, noise, n_points))
    return np.column_stack(columns).astype(np.float32)


def _apply_cause(
    values: np.ndarray,
    cause: str,
    start: int,
    end: int,
    severity: float,
    rng: np.random.Generator,
) -> None:
    """Add the signature of one root cause in place over ``[start, end)``."""
    span = end - start
    ramp = np.linspace(0.0, 1.0, span, dtype=np.float32)
    bump = np.hanning(span).astype(np.float32) if span > 2 else np.ones(span, dtype=np.float32)

    if cause == "memory_leak":
        values[start:end, 0] += ramp * severity * rng.uniform(60, 120)
        values[start:end, 2] += ramp * severity * rng.uniform(20, 40)
    elif cause == "deploy_regression":
        values[start:end, 1] += severity * rng.uniform(3, 8)
        values[start:end, 0] += severity * rng.uniform(15, 30)
    elif cause == "traffic_spike":
        values[start:end, 3] += bump * severity * rng.uniform(40, 90)
        values[start:end, 0] += bump * severity * rng.uniform(30, 70)
        values[start:end, 2] += bump * severity * rng.uniform(15, 35)
    elif cause == "downstream_outage":
        values[start:end, 1] += bump * severity * rng.uniform(10, 25)
        values[start:end, 0] += bump * severity * rng.uniform(20, 50)
    else:
        raise ValueError(f"unknown root cause {cause!r}")


def _event_starts(
    n_points: int, n_events: int, max_duration: int, rng: np.random.Generator
) -> list[int]:
    """Pick non-overlapping event starts, leaving room for the longest duration."""
    slot = n_points // max(n_events, 1)
    if slot <= max_duration:
        return []
    starts = []
    for index in range(n_events):
        low = index * slot
        high = low + slot - max_duration
        if high <= low:
            continue
        starts.append(int(rng.integers(low, high)))
    return starts


def generate_scenario(
    scenario_index: int,
    days: int = 4,
    events_per_day: float = 1.5,
    seed: int = 42,
    next_event_id: int = 0,
    cause_offset: int = 0,
    min_duration: int = 30,
    max_duration: int = 180,
) -> MetricSeries:
    """Generate one virtual service with labelled, non-overlapping anomaly events."""
    rng = np.random.default_rng(seed + scenario_index * 7919)
    n_points = days * MINUTES_PER_DAY
    values = _baseline(n_points, rng)

    n_events = max(1, round(days * events_per_day))
    starts = _event_starts(n_points, n_events, max_duration, rng)

    events: list[AnomalyEvent] = []
    point_labels = np.zeros(n_points, dtype=np.int8)

    for offset, start in enumerate(starts):
        cause = ROOT_CAUSES[(cause_offset + offset) % len(ROOT_CAUSES)]
        duration = int(rng.integers(min_duration, max_duration))
        end = min(start + duration, n_points)
        if end - start < min_duration:
            continue
        severity = float(rng.uniform(0.35, 1.6))
        _apply_cause(values, cause, start, end, severity, rng)
        point_labels[start:end] = 1
        events.append(
            AnomalyEvent(
                event_id=next_event_id + len(events),
                source=SOURCE,
                series_id=f"service-{scenario_index:04d}",
                start=start,
                end=end,
                label=cause,
            )
        )

    np.clip(values[:, 0], 5, None, out=values[:, 0])
    np.clip(values[:, 1], 0, 100, out=values[:, 1])
    np.clip(values[:, 2], 0, 100, out=values[:, 2])
    np.clip(values[:, 3], 0, None, out=values[:, 3])

    return MetricSeries(
        source=SOURCE,
        series_id=f"service-{scenario_index:04d}",
        values=values,
        point_labels=point_labels,
        event_ids=event_id_track(n_points, events),
        events=events,
        feature_names=list(FEATURE_NAMES),
    )


def generate(
    n_scenarios: int = 200,
    days: int = 4,
    events_per_day: float = 1.5,
    seed: int = 42,
) -> list[MetricSeries]:
    """Generate a full synthetic corpus with globally unique event ids."""
    series: list[MetricSeries] = []
    next_event_id = 0
    cause_offset = 0
    for index in range(n_scenarios):
        scenario = generate_scenario(
            index,
            days=days,
            events_per_day=events_per_day,
            seed=seed,
            next_event_id=next_event_id,
            cause_offset=cause_offset,
        )
        next_event_id += len(scenario.events)
        cause_offset += len(scenario.events)
        series.append(scenario)
    return series
