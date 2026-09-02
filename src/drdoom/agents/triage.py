"""Triage: is anything wrong, and what kind of wrong.

No model is called here. Detection and classification are the measured, deterministic
part of the system, and the language layer downstream depends on them being decided
before it is asked anything.

Both components are injected rather than constructed, so the detector that
measurement actually favoured can be swapped in without editing this file. On the real
dataset that was a window statistic, not the autoencoder.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drdoom.classify.features import FeatureSpec, extract_features
from drdoom.data.schema import NORMAL_EVENT_ID, MetricSeries
from drdoom.data.windows import WindowIndex, build_index
from drdoom.detect.base import Detector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TriageResult:
    """What triage concluded about one window."""

    is_anomaly: bool
    score: float
    threshold: float
    root_cause: str | None = None
    confidence: float | None = None

    def as_dict(self) -> dict:
        return {
            "is_anomaly": self.is_anomaly,
            "score": round(self.score, 6),
            "threshold": round(self.threshold, 6),
            "root_cause": self.root_cause,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
        }


def window_to_series(window: np.ndarray, feature_names: list[str]) -> tuple[list, WindowIndex]:
    """Wrap a single raw window so the detectors can score it unchanged."""
    if window.ndim != 2:
        raise ValueError(f"expected a (timesteps, metrics) window, got shape {window.shape}")
    series = MetricSeries(
        source="live",
        series_id="live",
        values=np.ascontiguousarray(window, dtype=np.float32),
        point_labels=np.zeros(len(window), dtype=np.int8),
        event_ids=np.full(len(window), NORMAL_EVENT_ID, dtype=np.int32),
        feature_names=list(feature_names),
    )
    return [series], build_index([series], window_size=len(window), stride=len(window))


class Classifier:
    """The trained root-cause model, loaded with the feature order it was fitted on."""

    def __init__(self, model, spec: FeatureSpec, labels: list[str]) -> None:
        self.model = model
        self.spec = spec
        self.labels = labels

    @classmethod
    def load(cls, directory: Path) -> Classifier:
        import xgboost as xgb

        model = xgb.XGBClassifier()
        model.load_model(directory / "model.json")
        spec = FeatureSpec.load(directory / "features.json")
        labels = json.loads((directory / "labels.json").read_text(encoding="utf-8"))
        if model.n_features_in_ != spec.n_columns:
            raise ValueError(
                f"model expects {model.n_features_in_} features, "
                f"the saved specification describes {spec.n_columns}"
            )
        return cls(model, spec, labels)

    def predict(self, window: np.ndarray, feature_names: list[str]) -> tuple[str, float]:
        self.spec.check(feature_names)
        features = extract_features(window[None, :, :])
        probabilities = self.model.predict_proba(features)[0]
        best = int(np.argmax(probabilities))
        return self.labels[best], float(probabilities[best])


class TriageAgent:
    """Scores a window, then names the cause only if something is wrong."""

    def __init__(
        self,
        detector: Detector,
        threshold: float,
        feature_names: list[str],
        classifier: Classifier | None = None,
    ) -> None:
        self.detector = detector
        self.threshold = threshold
        self.feature_names = list(feature_names)
        self.classifier = classifier

    def run(self, window: np.ndarray) -> TriageResult:
        series, index = window_to_series(window, self.feature_names)
        score = float(self.detector.score(series, index)[0])

        if score < self.threshold:
            return TriageResult(is_anomaly=False, score=score, threshold=self.threshold)

        root_cause, confidence = (None, None)
        if self.classifier is not None:
            try:
                root_cause, confidence = self.classifier.predict(window, self.feature_names)
            except ValueError:
                logger.exception("classification failed; reporting the anomaly without a cause")

        return TriageResult(
            is_anomaly=True,
            score=score,
            threshold=self.threshold,
            root_cause=root_cause,
            confidence=confidence,
        )
