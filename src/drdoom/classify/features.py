"""Window summary features, with a column order that is stated rather than implied.

Root cause is a question about the *shape* of a window -- how fast a metric climbed,
whether it stepped and held, how far it swung -- so the classifier reads summary
statistics rather than the raw sequence. Tree models handle those well on modest data
and stay interpretable, which matters when the output is shown to an on-call engineer.

The column order is the part worth being careful about. Building a feature vector by
iterating a dictionary happens to work while insertion order matches training, and fails
silently the moment it does not: no exception, no shape error, just a confidently wrong
prediction feeding a confidently worded diagnosis. ``FeatureSpec`` names every column
explicitly, travels with the model, and is checked on load.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STATISTICS: tuple[str, ...] = ("mean", "std", "min", "max", "slope", "half_diff")


def feature_order(metric_names: Sequence[str]) -> list[str]:
    """The one definition of column order, metric-major and statistic-minor."""
    return [f"{metric}_{statistic}" for metric in metric_names for statistic in STATISTICS]


@dataclass(frozen=True)
class FeatureSpec:
    """The metrics a model was trained on and the columns they expand into."""

    metric_names: tuple[str, ...]

    @property
    def columns(self) -> list[str]:
        return feature_order(self.metric_names)

    @property
    def n_columns(self) -> int:
        return len(self.metric_names) * len(STATISTICS)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"metric_names": list(self.metric_names), "columns": self.columns}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> FeatureSpec:
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec = cls(metric_names=tuple(payload["metric_names"]))
        saved = list(payload["columns"])
        if saved != spec.columns:
            raise ValueError(
                "saved feature columns do not match the current feature definition: "
                f"{saved[:3]}... vs {spec.columns[:3]}..."
            )
        return spec

    def check(self, metric_names: Sequence[str]) -> None:
        """Fail loudly when the caller's metrics differ from the trained ones."""
        if tuple(metric_names) != self.metric_names:
            raise ValueError(
                "metric order does not match the trained model: "
                f"{tuple(metric_names)[:3]}... vs {self.metric_names[:3]}..."
            )


def extract_features(windows: np.ndarray) -> np.ndarray:
    """Summarise ``(n, window, metrics)`` into ``(n, metrics * statistics)``.

    Columns come out metric-major and statistic-minor, matching ``feature_order`` exactly.
    The two are kept in step by a test rather than by convention.
    """
    if windows.ndim != 3:
        raise ValueError(f"expected (n, window, metrics), got shape {windows.shape}")
    n_windows, length, _ = windows.shape
    if length < 2:
        raise ValueError("a window needs at least two timesteps to have a slope")

    steps = np.arange(length, dtype=np.float32)
    centred_steps = steps - steps.mean()
    denominator = float((centred_steps**2).sum())

    mean = windows.mean(axis=1)
    slope = np.tensordot(windows - mean[:, None, :], centred_steps, axes=([1], [0])) / denominator
    half = length // 2
    half_diff = windows[:, half:, :].mean(axis=1) - windows[:, :half, :].mean(axis=1)

    stacked = np.stack(
        [mean, windows.std(axis=1), windows.min(axis=1), windows.max(axis=1), slope, half_diff],
        axis=2,
    )
    return stacked.reshape(n_windows, -1).astype(np.float32)
