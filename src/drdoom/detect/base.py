"""The interface every detector implements.

A detector turns a window into a single number where larger means more anomalous. It
never decides what counts as an anomaly -- thresholding is a separate, operational
choice made in ``evaluate`` against a false alarm budget, so that every detector is
compared on the same footing.

Scoring runs in batches because materialising every window of the real dataset at once
costs several gigabytes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Self

import numpy as np

from drdoom.data.schema import MetricSeries
from drdoom.data.windows import Scaler, WindowIndex, iter_batches

BATCH_SIZE = 2048


class Detector(ABC):
    """Scores windows. Higher is more anomalous."""

    name: str = "detector"

    def __init__(self) -> None:
        self.scaler: Scaler | None = None

    def fit(self, series: list[MetricSeries], index: WindowIndex, scaler: Scaler) -> Self:
        """Learn whatever the detector needs from anomaly-free training windows."""
        self.scaler = scaler
        return self

    @abstractmethod
    def score(self, series: list[MetricSeries], index: WindowIndex) -> np.ndarray:
        """Return one score per window in ``index``."""

    def _scaled_batches(
        self, series: list[MetricSeries], index: WindowIndex
    ) -> Iterator[np.ndarray]:
        if self.scaler is None:
            raise RuntimeError(f"{self.name} must be fitted before scoring")
        for batch, _ in iter_batches(series, index, batch_size=BATCH_SIZE):
            yield self.scaler.transform(batch)


class WindowStatisticDetector(Detector):
    """Base for detectors that reduce a scaled window to a scalar with no fitted state."""

    def score(self, series: list[MetricSeries], index: WindowIndex) -> np.ndarray:
        if not len(index):
            return np.empty(0, dtype=np.float32)
        return np.concatenate(
            [self._statistic(batch) for batch in self._scaled_batches(series, index)]
        )

    @abstractmethod
    def _statistic(self, windows: np.ndarray) -> np.ndarray:
        """Reduce ``(batch, window, features)`` to ``(batch,)``."""
