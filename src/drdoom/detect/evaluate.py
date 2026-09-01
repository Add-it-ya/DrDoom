"""Event-level scoring for detectors.

Window-level precision and recall are the wrong unit. A single outage lasting hours
produces hundreds of overlapping windows, so a detector that catches one long incident
and misses ten short ones can still post an excellent window F1. Worse, the apparent
sample size is inflated by an order of magnitude and any confidence interval computed
over windows is far too narrow.

What an on-call rotation actually asks is: did we catch the incident, how long after it
started, and how often were we paged for nothing. Those are the primary metrics here.
Window precision, recall and F1 are kept as a secondary table for comparability with
published results.

Timesteps are one minute apart in both sources, so a timestep is a minute throughout.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from drdoom.data.schema import MetricSeries
from drdoom.data.windows import WindowIndex

MINUTES_PER_DAY = 1440
DEFAULT_FALSE_ALARM_BUDGET = 1.0
BOOTSTRAP_SAMPLES = 2000


@dataclass(frozen=True)
class EventOutcome:
    event_id: int
    series_id: str
    detected: bool
    minutes_to_detect: float | None


@dataclass(frozen=True)
class DetectionReport:
    """Everything measured for one detector on one split."""

    detector: str
    threshold: float
    n_events: int
    detection_rate: float
    detection_rate_ci: tuple[float, float]
    median_minutes_to_detect: float | None
    minutes_to_detect_ci: tuple[float, float] | None
    false_alarms_per_day: float
    false_alarms_per_day_ci: tuple[float, float]
    window_precision: float
    window_recall: float
    window_f1: float
    roc_auc: float
    pr_auc: float
    outcomes: list[EventOutcome] = field(default_factory=list, repr=False)

    def as_row(self) -> dict:
        return {
            "detector": self.detector,
            "events": self.n_events,
            "detection_rate": round(self.detection_rate, 4),
            "detection_rate_ci": [round(v, 4) for v in self.detection_rate_ci],
            "median_minutes_to_detect": self.median_minutes_to_detect,
            "minutes_to_detect_ci": (
                [round(v, 1) for v in self.minutes_to_detect_ci]
                if self.minutes_to_detect_ci
                else None
            ),
            "false_alarms_per_day": round(self.false_alarms_per_day, 3),
            "false_alarms_per_day_ci": [round(v, 3) for v in self.false_alarms_per_day_ci],
            "window_precision": round(self.window_precision, 4),
            "window_recall": round(self.window_recall, 4),
            "window_f1": round(self.window_f1, 4),
            "roc_auc": round(self.roc_auc, 4),
            "pr_auc": round(self.pr_auc, 4),
            "threshold": round(float(self.threshold), 6),
        }


def _rows_by_series(index: WindowIndex, n_series: int) -> list[np.ndarray]:
    return [np.flatnonzero(index.series_index == position) for position in range(n_series)]


def event_outcomes(
    scores: np.ndarray,
    index: WindowIndex,
    series: list[MetricSeries],
    threshold: float,
) -> list[EventOutcome]:
    """Decide, for every event, whether any overlapping window fired and how late.

    A window is treated as raising its alert at its final timestep, because the whole
    window is needed before it can be scored. Time to detect is measured from the start
    of the event to that alert.
    """
    flagged = scores >= threshold
    rows_for = _rows_by_series(index, len(series))
    window = index.window_size

    outcomes: list[EventOutcome] = []
    for position, item in enumerate(series):
        rows = rows_for[position]
        if not len(rows):
            outcomes.extend(
                EventOutcome(event.event_id, item.series_id, False, None) for event in item.events
            )
            continue

        starts = index.start[rows]
        ends = starts + window
        for event in item.events:
            overlaps = (starts < event.end) & (ends > event.start)
            hit_rows = rows[overlaps & flagged[rows]]
            if not len(hit_rows):
                outcomes.append(EventOutcome(event.event_id, item.series_id, False, None))
                continue
            alert_times = index.start[hit_rows] + window
            delay = float(max(0, int(alert_times.min()) - event.start))
            outcomes.append(EventOutcome(event.event_id, item.series_id, True, delay))
    return outcomes


def false_alarm_episodes(scores: np.ndarray, index: WindowIndex, threshold: float) -> int:
    """Count runs of consecutive flagged windows that overlap no event.

    Consecutive flagged windows are one page, not many, so they are collapsed into a
    single episode. Counting each window separately would overstate alert fatigue by
    roughly the window-to-stride ratio.
    """
    rows = np.flatnonzero((index.label == 0) & (scores >= threshold))
    if not len(rows):
        return 0
    breaks = (np.diff(rows) != 1) | (np.diff(index.series_index[rows]) != 0)
    return int(1 + breaks.sum())


def normal_minutes(series: list[MetricSeries]) -> int:
    return int(sum(int((item.point_labels == 0).sum()) for item in series))


def _bootstrap_ci(
    sample: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap over the given sample."""
    if not len(sample):
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sample), size=(n_samples, len(sample)))
    values = np.array([statistic(sample[row]) for row in draws])
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


def _window_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> tuple[float, float, float]:
    predicted = scores >= threshold
    truth = labels == 1
    true_positive = int((predicted & truth).sum())
    false_positive = int((predicted & ~truth).sum())
    false_negative = int((~predicted & truth).sum())
    precision = true_positive / (true_positive + false_positive) if predicted.any() else 0.0
    recall = true_positive / (true_positive + false_negative) if truth.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate(
    detector_name: str,
    scores: np.ndarray,
    index: WindowIndex,
    series: list[MetricSeries],
    threshold: float,
    seed: int = 0,
) -> DetectionReport:
    """Score one detector on one split at a fixed threshold."""
    outcomes = event_outcomes(scores, index, series, threshold)
    detected = np.array([outcome.detected for outcome in outcomes], dtype=bool)
    delays = np.array(
        [o.minutes_to_detect for o in outcomes if o.minutes_to_detect is not None], dtype=float
    )

    days = max(normal_minutes(series) / MINUTES_PER_DAY, 1e-9)
    episodes = false_alarm_episodes(scores, index, threshold)

    # Series are the independent unit for a false alarm rate, so the interval is
    # resampled over series rather than over windows.
    per_series_rate = []
    for position, item in enumerate(series):
        rows = np.flatnonzero(index.series_index == position)
        if not len(rows):
            continue
        subset = index.subset(rows)
        item_days = max(int((item.point_labels == 0).sum()) / MINUTES_PER_DAY, 1e-9)
        per_series_rate.append(false_alarm_episodes(scores[rows], subset, threshold) / item_days)

    precision, recall, f1 = _window_metrics(scores, index.label, threshold)
    has_both_classes = 0 < index.label.mean() < 1

    return DetectionReport(
        detector=detector_name,
        threshold=float(threshold),
        n_events=len(outcomes),
        detection_rate=float(detected.mean()) if len(detected) else 0.0,
        detection_rate_ci=_bootstrap_ci(detected, np.mean, seed=seed),
        median_minutes_to_detect=float(np.median(delays)) if len(delays) else None,
        minutes_to_detect_ci=(_bootstrap_ci(delays, np.median, seed=seed) if len(delays) else None),
        false_alarms_per_day=episodes / days,
        false_alarms_per_day_ci=_bootstrap_ci(
            np.array(per_series_rate, dtype=float), np.mean, seed=seed
        ),
        window_precision=precision,
        window_recall=recall,
        window_f1=f1,
        roc_auc=float(roc_auc_score(index.label, scores)) if has_both_classes else float("nan"),
        pr_auc=(
            float(average_precision_score(index.label, scores))
            if has_both_classes
            else float("nan")
        ),
        outcomes=outcomes,
    )


def select_threshold(
    scores: np.ndarray,
    index: WindowIndex,
    series: list[MetricSeries],
    budget_per_day: float = DEFAULT_FALSE_ALARM_BUDGET,
    n_candidates: int = 200,
) -> float:
    """Pick the most sensitive threshold that stays inside a false alarm budget.

    This is how an alerting threshold is chosen in practice: the tolerable page rate is
    fixed first, and sensitivity is whatever that budget allows. Tuning instead for best
    window F1 optimises a quantity nobody is on call for.
    """
    if not len(scores):
        return float("inf")

    candidates = np.unique(np.percentile(scores, np.linspace(50.0, 100.0, n_candidates)))
    days = max(normal_minutes(series) / MINUTES_PER_DAY, 1e-9)

    affordable = [
        threshold
        for threshold in candidates
        if false_alarm_episodes(scores, index, threshold) / days <= budget_per_day
    ]
    if not affordable:
        return float(candidates[-1])

    # Lower threshold means higher sensitivity, so take the smallest affordable one.
    return float(min(affordable))


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray, n_candidates: int = 200) -> float:
    """Threshold maximising window F1, kept for comparison with published numbers."""
    if not len(scores) or not 0 < labels.mean() < 1:
        return float("inf")
    best_threshold, best_score = float("inf"), -1.0
    for threshold in np.unique(np.percentile(scores, np.linspace(50.0, 99.9, n_candidates))):
        _, _, f1 = _window_metrics(scores, labels, threshold)
        if f1 > best_score:
            best_threshold, best_score = float(threshold), f1
    return best_threshold
