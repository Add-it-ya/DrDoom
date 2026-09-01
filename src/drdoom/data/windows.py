"""Sliding windows over metric series, and the scaler applied to them.

Windows are held as an *index* -- a series reference plus a start offset -- rather than
as a materialised tensor. Materialising every window of the real dataset at a useful
stride would cost a few gigabytes, and every consumer either iterates in batches or
wants a subset, so the index is built once and slices are cut on demand.

Each window carries the id of the event it overlaps most, which is what lets downstream
evaluation group windows by incident instead of counting overlapping slices of one long
outage as independent observations.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drdoom.data.schema import NORMAL_EVENT_ID, MetricSeries

DEFAULT_WINDOW = 60
DEFAULT_STRIDE = 10


@dataclass(frozen=True)
class WindowIndex:
    """Positions of every window, with its label and owning event."""

    series_index: np.ndarray
    start: np.ndarray
    label: np.ndarray
    event_id: np.ndarray
    window_size: int
    stride: int

    def __len__(self) -> int:
        return len(self.start)

    @property
    def anomaly_rate(self) -> float:
        return float(self.label.mean()) if len(self) else 0.0

    def subset(self, rows: np.ndarray) -> WindowIndex:
        return WindowIndex(
            series_index=self.series_index[rows],
            start=self.start[rows],
            label=self.label[rows],
            event_id=self.event_id[rows],
            window_size=self.window_size,
            stride=self.stride,
        )

    def normal_only(self) -> WindowIndex:
        return self.subset(np.flatnonzero(self.label == 0))


def build_index(
    series: list[MetricSeries],
    window_size: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
) -> WindowIndex:
    """Index every window that fits inside a single series.

    A window never spans two series, because each series is walked independently.
    """
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")

    series_index: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    event_ids: list[np.ndarray] = []

    for position, item in enumerate(series):
        if item.n_timesteps < window_size:
            continue
        offsets = np.arange(0, item.n_timesteps - window_size + 1, stride, dtype=np.int32)
        window_labels = np.empty(len(offsets), dtype=np.int8)
        window_events = np.empty(len(offsets), dtype=np.int32)

        for row, offset in enumerate(offsets):
            track = item.event_ids[offset : offset + window_size]
            anomalous = track[track != NORMAL_EVENT_ID]
            if anomalous.size:
                identifiers, counts = np.unique(anomalous, return_counts=True)
                window_events[row] = identifiers[counts.argmax()]
                window_labels[row] = 1
            else:
                window_events[row] = NORMAL_EVENT_ID
                window_labels[row] = 0

        series_index.append(np.full(len(offsets), position, dtype=np.int32))
        starts.append(offsets)
        labels.append(window_labels)
        event_ids.append(window_events)

    if not starts:
        empty_int = np.empty(0, dtype=np.int32)
        return WindowIndex(
            series_index=empty_int,
            start=empty_int,
            label=np.empty(0, dtype=np.int8),
            event_id=empty_int,
            window_size=window_size,
            stride=stride,
        )

    return WindowIndex(
        series_index=np.concatenate(series_index),
        start=np.concatenate(starts),
        label=np.concatenate(labels),
        event_id=np.concatenate(event_ids),
        window_size=window_size,
        stride=stride,
    )


def materialise(series: list[MetricSeries], index: WindowIndex) -> np.ndarray:
    """Cut the indexed windows into a ``(n_windows, window_size, n_features)`` tensor."""
    if not len(index):
        n_features = series[0].n_features if series else 0
        return np.empty((0, index.window_size, n_features), dtype=np.float32)

    n_features = series[int(index.series_index[0])].n_features
    output = np.empty((len(index), index.window_size, n_features), dtype=np.float32)
    for row, (position, offset) in enumerate(zip(index.series_index, index.start, strict=True)):
        output[row] = series[int(position)].values[offset : offset + index.window_size]
    return output


def iter_batches(
    series: list[MetricSeries], index: WindowIndex, batch_size: int = 256
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(windows, labels)`` batches without materialising the whole tensor."""
    for begin in range(0, len(index), batch_size):
        rows = np.arange(begin, min(begin + batch_size, len(index)))
        batch = index.subset(rows)
        yield materialise(series, batch), batch.label


@dataclass(frozen=True)
class Scaler:
    """Per-feature standardisation, fitted on training data only.

    ``feature_names`` travels with the statistics so that loading a scaler against a
    differently ordered feature set fails loudly instead of silently mis-scaling.
    """

    means: np.ndarray
    stds: np.ndarray
    feature_names: tuple[str, ...]

    @classmethod
    def fit(cls, series: list[MetricSeries]) -> Scaler:
        if not series:
            raise ValueError("cannot fit a scaler on no series")
        names = tuple(series[0].feature_names)
        for item in series[1:]:
            if tuple(item.feature_names) != names:
                raise ValueError(f"series {item.series_id} has a different feature order")

        stacked = np.concatenate([item.values for item in series], axis=0)
        means = stacked.mean(axis=0, dtype=np.float64)
        stds = stacked.std(axis=0, dtype=np.float64)
        # A constant column carries no information; scaling it by zero would produce
        # infinities, so leave it centred at zero with unit scale.
        stds[stds < 1e-8] = 1.0
        return cls(
            means=means.astype(np.float32),
            stds=stds.astype(np.float32),
            feature_names=names,
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        if values.shape[-1] != len(self.means):
            raise ValueError(f"expected {len(self.means)} features, got {values.shape[-1]}")
        return ((values - self.means) / self.stds).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return (values * self.stds + self.means).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            means=self.means,
            stds=self.stds,
            feature_names=np.array(self.feature_names, dtype=object),
        )

    @classmethod
    def load(cls, path: Path, expected_features: list[str] | None = None) -> Scaler:
        payload = np.load(path, allow_pickle=True)
        names = tuple(str(name) for name in payload["feature_names"])
        if expected_features is not None and tuple(expected_features) != names:
            raise ValueError(
                "scaler feature order does not match the caller: "
                f"saved {names[:3]}... vs expected {tuple(expected_features)[:3]}..."
            )
        return cls(means=payload["means"], stds=payload["stds"], feature_names=names)
