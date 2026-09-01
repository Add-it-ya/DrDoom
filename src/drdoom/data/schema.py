"""Shared structures for metric series and the anomaly events inside them.

An *event* is one contiguous run of anomalous timesteps in one series. Events are the
unit every downstream metric is reported against: a single incident lasting hours would
otherwise be counted as hundreds of overlapping windows and inflate the apparent sample
size far beyond the number of independent incidents actually observed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NORMAL_EVENT_ID = -1
UNKNOWN_LABEL = "unknown"


@dataclass(frozen=True)
class AnomalyEvent:
    """A contiguous anomalous interval, half-open as ``[start, end)``."""

    event_id: int
    source: str
    series_id: str
    start: int
    end: int
    dims: tuple[int, ...] = ()
    label: str = UNKNOWN_LABEL

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"event {self.event_id} has empty span [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class MetricSeries:
    """One machine or scenario: a metric matrix plus its per-timestep annotations."""

    source: str
    series_id: str
    values: np.ndarray
    point_labels: np.ndarray
    event_ids: np.ndarray
    events: list[AnomalyEvent] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(f"values must be 2d, got shape {self.values.shape}")
        n_steps = len(self.values)
        for name, array in (("point_labels", self.point_labels), ("event_ids", self.event_ids)):
            if len(array) != n_steps:
                raise ValueError(f"{name} has {len(array)} entries, expected {n_steps}")
        if self.feature_names and len(self.feature_names) != self.values.shape[1]:
            raise ValueError(
                f"{len(self.feature_names)} feature names for {self.values.shape[1]} columns"
            )

    @property
    def n_timesteps(self) -> int:
        return len(self.values)

    @property
    def n_features(self) -> int:
        return self.values.shape[1]


def find_label_runs(point_labels: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous ``[start, end)`` spans where ``point_labels`` is non-zero."""
    flags = (np.asarray(point_labels) != 0).astype(np.int8)
    edges = np.diff(np.concatenate([[0], flags, [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def event_id_track(n_timesteps: int, events: list[AnomalyEvent]) -> np.ndarray:
    """Expand events into a per-timestep event id track, ``NORMAL_EVENT_ID`` elsewhere."""
    track = np.full(n_timesteps, NORMAL_EVENT_ID, dtype=np.int32)
    for event in events:
        track[event.start : event.end] = event.event_id
    return track
