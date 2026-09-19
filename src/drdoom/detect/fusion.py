"""Combine detectors by taking the most alarmed of them, on a common scale.

Scores from different detectors are in different units: a reconstruction error, a jump
between consecutive minutes. Each member is put on a robust scale fitted to its own scores
over the training windows, the median and interquartile range, and the fused score is the
largest of the rescaled members. A window fires if any member finds it unusual by its own
standard.

The scale uses training windows only. No label and no validation window shapes it, so the
fusion adds nothing that could leak the evaluation into the model.
"""

from __future__ import annotations

from typing import Self

import numpy as np

from drdoom.data.schema import MetricSeries
from drdoom.data.windows import Scaler, WindowIndex
from drdoom.detect.base import Detector


class MaxFusion(Detector):
    """The largest robust z-score among fitted member detectors."""

    def __init__(self, members: list[Detector]) -> None:
        super().__init__()
        if len(members) < 2:
            raise ValueError("a fusion needs at least two members")
        self.members = members
        self.name = "+".join(member.name for member in members)
        self.centres: list[float] = []
        self.spreads: list[float] = []

    def fit(self, series: list[MetricSeries], index: WindowIndex, scaler: Scaler) -> Self:
        """Fit the scale of each already-fitted member on the training windows.

        Members are fitted by whoever built them, so a trained network is not trained
        again here. Only its score distribution over training windows is measured.
        """
        super().fit(series, index, scaler)
        self.centres, self.spreads = [], []
        for member in self.members:
            scores = member.score(series, index).astype(np.float64)
            low, centre, high = np.percentile(scores, [25, 50, 75])
            self.centres.append(float(centre))
            self.spreads.append(float(max(high - low, 1e-9)))
        return self

    def score(self, series: list[MetricSeries], index: WindowIndex) -> np.ndarray:
        if not self.centres:
            raise RuntimeError(f"{self.name} must be fitted before scoring")
        if not len(index):
            return np.empty(0, dtype=np.float32)
        rescaled = [
            (member.score(series, index).astype(np.float64) - centre) / spread
            for member, centre, spread in zip(self.members, self.centres, self.spreads, strict=True)
        ]
        return np.max(rescaled, axis=0).astype(np.float32)
