"""Train/validation/test partitioning.

Two strategies, because they answer different questions:

* ``time_based`` cuts each series chronologically. It measures whether the model can
  detect a future incident on a machine it has already seen, and it never lets future
  data inform training.
* ``held_out_series`` assigns whole machines to one split. It measures whether the model
  transfers to a service it has never observed, which is the harder and more honest
  question for a system meant to be pointed at new infrastructure.

Cuts are nudged to the nearest normal timestep so that no split boundary bisects an
incident. A half-event on either side of a boundary would be counted as a whole event by
downstream metrics and quietly inflate the event totals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drdoom.data.schema import AnomalyEvent, MetricSeries, event_id_track

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class SplitResult:
    strategy: str
    train: list[MetricSeries]
    val: list[MetricSeries]
    test: list[MetricSeries]

    def as_dict(self) -> dict[str, list[MetricSeries]]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def event_counts(self) -> dict[str, int]:
        return {name: sum(len(s.events) for s in items) for name, items in self.as_dict().items()}


def nearest_normal_cut(point_labels: np.ndarray, target: int) -> int:
    """Move ``target`` to the closest index whose label is normal."""
    n = len(point_labels)
    target = int(np.clip(target, 0, n))
    if target in (0, n) or point_labels[target] == 0:
        return target
    normal = np.flatnonzero(point_labels == 0)
    if normal.size == 0:
        return target
    return int(normal[np.abs(normal - target).argmin()])


def slice_series(series: MetricSeries, start: int, end: int) -> MetricSeries:
    """Cut ``[start, end)`` out of a series, keeping only events fully inside it."""
    values = series.values[start:end]
    point_labels = series.point_labels[start:end]
    events = [
        AnomalyEvent(
            event_id=event.event_id,
            source=event.source,
            series_id=series.series_id,
            start=event.start - start,
            end=event.end - start,
            dims=event.dims,
            label=event.label,
        )
        for event in series.events
        if event.start >= start and event.end <= end
    ]
    return MetricSeries(
        source=series.source,
        series_id=series.series_id,
        values=values,
        point_labels=point_labels,
        event_ids=event_id_track(len(values), events),
        events=events,
        feature_names=list(series.feature_names),
    )


def time_based_split(
    series: list[MetricSeries], train_frac: float = 0.7, val_frac: float = 0.15
) -> SplitResult:
    """Cut every series chronologically into train, validation and test portions."""
    _check_fractions(train_frac, val_frac)
    train: list[MetricSeries] = []
    val: list[MetricSeries] = []
    test: list[MetricSeries] = []

    for item in series:
        n = item.n_timesteps
        first = nearest_normal_cut(item.point_labels, int(n * train_frac))
        second = nearest_normal_cut(item.point_labels, int(n * (train_frac + val_frac)))
        second = max(second, first)
        train.append(slice_series(item, 0, first))
        val.append(slice_series(item, first, second))
        test.append(slice_series(item, second, n))

    return SplitResult(strategy="time_based", train=train, val=val, test=test)


def held_out_series_split(
    series: list[MetricSeries],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 0,
) -> SplitResult:
    """Assign whole series to a split, so test series are never seen during training."""
    _check_fractions(train_frac, val_frac)
    ordered = sorted(series, key=lambda item: item.series_id)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ordered))

    n = len(ordered)
    n_train = int(n * train_frac)
    n_val = max(1, int(n * val_frac)) if n - n_train > 1 else 0
    groups: dict[str, list[MetricSeries]] = {name: [] for name in SPLIT_NAMES}
    for rank, position in enumerate(order):
        if rank < n_train:
            groups["train"].append(ordered[position])
        elif rank < n_train + n_val:
            groups["val"].append(ordered[position])
        else:
            groups["test"].append(ordered[position])

    return SplitResult(strategy="held_out_series", **groups)


def _check_fractions(train_frac: float, val_frac: float) -> None:
    if not 0 < train_frac < 1 or not 0 <= val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError(
            f"train_frac={train_frac} and val_frac={val_frac} must leave a non-empty test split"
        )
