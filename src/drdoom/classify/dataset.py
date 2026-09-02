"""Assemble labelled windows for the classifier.

Classification is conditional on detection: the question is only asked once a window has
been flagged as anomalous, so only anomalous windows are used here.

Every row carries the id of the event it came from. Those ids become the grouping key for
cross-validation, so windows cut from the same incident can never land on both sides of a
fold boundary. Without that, overlapping slices of one outage appear in train and
validation at once and the score reported is largely a measure of memorisation.

Features are taken from raw, unscaled windows. Tree models are invariant to monotone
rescaling, and leaving the units alone keeps a feature importance readable as
"latency slope in milliseconds per minute" rather than in standard deviations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drdoom.classify.archetypes import Archetypes
from drdoom.classify.features import FeatureSpec, extract_features
from drdoom.data.schema import MetricSeries
from drdoom.data.windows import WindowIndex, materialise

MIN_ROWS_PER_CLASS = 1


@dataclass(frozen=True)
class ClassificationData:
    """Feature rows, integer labels, and the event each row came from."""

    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    label_names: tuple[str, ...]
    spec: FeatureSpec

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def n_events(self) -> int:
        return len(np.unique(self.groups))

    def events_per_class(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for position, name in enumerate(self.label_names):
            rows = self.labels == position
            counts[name] = len(np.unique(self.groups[rows])) if rows.any() else 0
        return counts

    def rows_per_class(self) -> dict[str, int]:
        return {
            name: int((self.labels == position).sum())
            for position, name in enumerate(self.label_names)
        }


def event_labels(
    series: list[MetricSeries], archetypes: Archetypes | None = None
) -> dict[int, str]:
    """Map every event id to its class name.

    The synthetic corpus carries a true causal label on each event. The real dataset does
    not, so its events are labelled by the signature archetype derived from the
    interpretation file.
    """
    lookup: dict[int, str] = {}
    for item in series:
        for event in item.events:
            if archetypes is not None:
                lookup[event.event_id] = archetypes.label_for(event)
            else:
                lookup[event.event_id] = event.label
    return lookup


def build(
    series: list[MetricSeries],
    index: WindowIndex,
    label_names: tuple[str, ...],
    archetypes: Archetypes | None = None,
    batch_size: int = 2048,
) -> ClassificationData:
    """Turn the anomalous windows of a split into a labelled feature matrix."""
    spec = FeatureSpec(metric_names=tuple(series[0].feature_names)) if series else FeatureSpec(())
    anomalous = index.subset(np.flatnonzero(index.label == 1))
    lookup = event_labels(series, archetypes)
    position_of = {name: position for position, name in enumerate(label_names)}

    if not len(anomalous):
        return ClassificationData(
            features=np.empty((0, spec.n_columns), dtype=np.float32),
            labels=np.empty(0, dtype=np.int64),
            groups=np.empty(0, dtype=np.int64),
            label_names=label_names,
            spec=spec,
        )

    blocks = []
    for begin in range(0, len(anomalous), batch_size):
        rows = np.arange(begin, min(begin + batch_size, len(anomalous)))
        blocks.append(extract_features(materialise(series, anomalous.subset(rows))))
    features = np.concatenate(blocks)

    names = [lookup.get(int(event_id), "other") for event_id in anomalous.event_id]
    keep = np.array([name in position_of for name in names], dtype=bool)
    labels = np.array(
        [position_of[name] for name, ok in zip(names, keep, strict=True) if ok], dtype=np.int64
    )

    return ClassificationData(
        features=features[keep],
        labels=labels,
        groups=anomalous.event_id[keep].astype(np.int64),
        label_names=label_names,
        spec=spec,
    )
