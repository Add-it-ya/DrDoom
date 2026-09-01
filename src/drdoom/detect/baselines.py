"""Simple detectors, built before the neural network and measured against it.

None of these learn a temporal model. They exist to establish what a few lines of numpy
already buy, so that any gain claimed for the autoencoder is a gain over something rather
than a number reported in isolation.

Every score is computed on windows standardised by the training scaler, so a value is
already expressed in training standard deviations.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from drdoom.data.schema import MetricSeries
from drdoom.data.windows import Scaler, WindowIndex
from drdoom.detect.base import Detector, WindowStatisticDetector


class MaxDeviation(WindowStatisticDetector):
    """Largest absolute departure from the training mean, in training deviations."""

    name = "max_deviation"

    def _statistic(self, windows: np.ndarray) -> np.ndarray:
        return np.abs(windows).max(axis=(1, 2))


class WindowSpread(WindowStatisticDetector):
    """Largest within-window standard deviation across features.

    Catches a metric that moves a lot inside the window regardless of its level, which
    covers ramps and spikes that never leave the normal absolute range.
    """

    name = "window_spread"

    def _statistic(self, windows: np.ndarray) -> np.ndarray:
        return windows.std(axis=1).max(axis=1)


class EwmaResidual(WindowStatisticDetector):
    """Largest departure from an exponentially weighted moving average of the window."""

    name = "ewma_residual"

    def __init__(self, alpha: float = 0.3) -> None:
        super().__init__()
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha

    def _statistic(self, windows: np.ndarray) -> np.ndarray:
        smoothed = np.empty_like(windows)
        smoothed[:, 0] = windows[:, 0]
        for step in range(1, windows.shape[1]):
            smoothed[:, step] = (
                self.alpha * windows[:, step] + (1 - self.alpha) * smoothed[:, step - 1]
            )
        return np.abs(windows - smoothed).max(axis=(1, 2))


class NaiveResidual(WindowStatisticDetector):
    """Largest step between consecutive timesteps.

    The seasonal-naive residual this stands in for needs a full period to compare
    against, and a sixty minute window does not contain one. A lag-one difference is the
    honest degenerate case: it catches abrupt change without pretending to model season.
    """

    name = "naive_residual"

    def _statistic(self, windows: np.ndarray) -> np.ndarray:
        if windows.shape[1] < 2:
            return np.zeros(len(windows), dtype=np.float32)
        return np.abs(np.diff(windows, axis=1)).max(axis=(1, 2))


def summarise_windows(windows: np.ndarray) -> np.ndarray:
    """Reduce each window to per-feature mean, deviation, minimum and maximum."""
    return np.concatenate(
        [
            windows.mean(axis=1),
            windows.std(axis=1),
            windows.min(axis=1),
            windows.max(axis=1),
        ],
        axis=1,
    )


class IsolationForestDetector(Detector):
    """Isolation forest over per-window summary statistics.

    The only baseline here that fits anything. It sees the same anomaly-free training
    windows as the autoencoder, which makes it the fairer comparison of the two.
    """

    name = "isolation_forest"

    def __init__(self, n_estimators: int = 200, seed: int = 42) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.seed = seed
        self.model: IsolationForest | None = None

    def fit(
        self, series: list[MetricSeries], index: WindowIndex, scaler: Scaler
    ) -> IsolationForestDetector:
        super().fit(series, index, scaler)
        features = np.concatenate(
            [summarise_windows(batch) for batch in self._scaled_batches(series, index)]
        )
        self.model = IsolationForest(
            n_estimators=self.n_estimators, random_state=self.seed, n_jobs=-1
        ).fit(features)
        return self

    def score(self, series: list[MetricSeries], index: WindowIndex) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{self.name} must be fitted before scoring")
        if not len(index):
            return np.empty(0, dtype=np.float32)
        features = np.concatenate(
            [summarise_windows(batch) for batch in self._scaled_batches(series, index)]
        )
        # decision_function is higher for inliers; invert so higher means more anomalous.
        return -self.model.decision_function(features).astype(np.float32)


def all_baselines() -> list[Detector]:
    return [
        MaxDeviation(),
        WindowSpread(),
        EwmaResidual(),
        NaiveResidual(),
        IsolationForestDetector(),
    ]
