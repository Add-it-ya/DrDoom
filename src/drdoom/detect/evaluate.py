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

**A standing alarm pages again.** Pages used to be counted as runs of consecutive firing
windows, one page per run however long. That rewards a detector that stays on: a machine
whose alarm never clears costs one page a day and "detects" every incident on it. Real
alerting re-notifies an unresolved alert, so a run now costs one page per hour it lasts
(``paged_alarms``). A run shorter than an hour costs one page, as before. The old count is
kept as ``false_alarm_episodes`` so the two can be read side by side.

**A detection needs a fresh alarm.** An incident whose window was already firing just
before it began is still counted as detected, since an alarm was up, but it is flagged as
pre-alarmed and reported separately: the detector was not responding to the incident.

**Detectors are compared across budgets, not at one.** A single threshold hides how a
detector trades pages for detections. ``curve_auc`` averages, over budgets from half a
page to four pages per series-day, the best detection rate each budget allows.

Timesteps are one minute apart in both sources, so a timestep is a minute throughout.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from drdoom.data.schema import MetricSeries
from drdoom.data.windows import WindowIndex

MINUTES_PER_DAY = 1440
DEFAULT_FALSE_ALARM_BUDGET = 1.0
DEFAULT_REPAGE_MINUTES = 60
CURVE_BUDGETS = (0.5, 4.0)
BOOTSTRAP_SAMPLES = 2000

Accounting = Literal["repage", "episodes"]


@dataclass(frozen=True)
class EventOutcome:
    event_id: int
    series_id: str
    detected: bool
    minutes_to_detect: float | None
    pre_alarmed: bool = False


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
    pages_per_day: float
    pages_per_day_ci: tuple[float, float]
    alarm_duty: float
    worst_alarm_duty: float
    pre_alarmed_share: float
    fresh_detection_rate: float
    curve_auc: float
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
            "pages_per_day": round(self.pages_per_day, 3),
            "pages_per_day_ci": [round(v, 3) for v in self.pages_per_day_ci],
            "alarm_duty": round(self.alarm_duty, 4),
            "worst_alarm_duty": round(self.worst_alarm_duty, 4),
            "pre_alarmed_share": round(self.pre_alarmed_share, 4),
            "fresh_detection_rate": round(self.fresh_detection_rate, 4),
            "curve_auc": round(self.curve_auc, 4),
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

    A detected event is marked pre-alarmed when the last normal window that ends before
    it began was already firing: the alarm was up before the incident, so it says nothing
    about the detector noticing it.
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
        normal = index.label[rows] == 0
        for event in item.events:
            overlaps = (starts < event.end) & (ends > event.start)
            hit_rows = rows[overlaps & flagged[rows]]
            if not len(hit_rows):
                outcomes.append(EventOutcome(event.event_id, item.series_id, False, None))
                continue
            alert_times = index.start[hit_rows] + window
            delay = float(max(0, int(alert_times.min()) - event.start))
            before = rows[(ends <= event.start) & normal]
            pre_alarmed = bool(len(before)) and bool(
                flagged[before[np.argmax(index.start[before])]]
            )
            outcomes.append(EventOutcome(event.event_id, item.series_id, True, delay, pre_alarmed))
    return outcomes


def event_peaks(scores: np.ndarray, index: WindowIndex, series: list[MetricSeries]) -> np.ndarray:
    """The highest score of any window overlapping each event, in ``event_outcomes`` order.

    An event is detected at a threshold exactly when its peak reaches it, so a whole
    detection curve costs one pass over the events instead of one per threshold.
    """
    rows_for = _rows_by_series(index, len(series))
    peaks: list[float] = []
    for position, item in enumerate(series):
        rows = rows_for[position]
        starts = index.start[rows]
        ends = starts + index.window_size
        for event in item.events:
            overlapping = rows[(starts < event.end) & (ends > event.start)]
            peaks.append(float(scores[overlapping].max()) if len(overlapping) else -np.inf)
    return np.array(peaks, dtype=float)


def _alarm_runs(scores: np.ndarray, index: WindowIndex, threshold: float) -> list[np.ndarray]:
    """Runs of consecutive firing windows that overlap no event, as arrays of rows."""
    rows = np.flatnonzero((index.label == 0) & (scores >= threshold))
    if not len(rows):
        return []
    breaks = (np.diff(rows) != 1) | (np.diff(index.series_index[rows]) != 0)
    return np.split(rows, np.flatnonzero(breaks) + 1)


def false_alarm_episodes(scores: np.ndarray, index: WindowIndex, threshold: float) -> int:
    """Count runs of consecutive flagged windows that overlap no event.

    Consecutive flagged windows are one page, not many, so they are collapsed into a
    single episode. Counting each window separately would overstate alert fatigue by
    roughly the window-to-stride ratio.
    """
    return len(_alarm_runs(scores, index, threshold))


def paged_alarms(
    scores: np.ndarray,
    index: WindowIndex,
    threshold: float,
    repage_minutes: int = DEFAULT_REPAGE_MINUTES,
) -> int:
    """Pages a human receives for nothing, when an alarm that stays up pages again.

    A run of firing windows lasts from its first window's start to its last window's
    start plus one stride. It costs one page per ``repage_minutes`` of that, rounded up,
    so a run under an hour is one page, as in ``false_alarm_episodes``, and a three-hour
    standing alarm is three.
    """
    total = 0
    for run in _alarm_runs(scores, index, threshold):
        duration = int(index.start[run[-1]] - index.start[run[0]]) + index.stride
        total += max(1, math.ceil(duration / repage_minutes))
    return total


def false_alarms(
    scores: np.ndarray, index: WindowIndex, threshold: float, accounting: Accounting = "repage"
) -> int:
    if accounting == "episodes":
        return false_alarm_episodes(scores, index, threshold)
    return paged_alarms(scores, index, threshold)


def alarm_duty(scores: np.ndarray, index: WindowIndex, threshold: float) -> tuple[float, float]:
    """The share of normal windows in alarm, overall and on the worst series.

    A detector in alarm for a quarter of all normal time can post a fine page rate under
    episode counting while being useless as a pager. This makes that visible.
    """
    normal = index.label == 0
    flagged = scores >= threshold
    if not normal.any():
        return 0.0, 0.0
    worst = 0.0
    for position in np.unique(index.series_index[normal]):
        mine = normal & (index.series_index == position)
        worst = max(worst, float(flagged[mine].mean()))
    return float(flagged[normal].mean()), worst


def threshold_candidates(scores: np.ndarray, n_candidates: int = 200) -> np.ndarray:
    return np.unique(np.percentile(scores, np.linspace(50.0, 100.0, n_candidates)))


def detection_curve(
    scores: np.ndarray,
    index: WindowIndex,
    series: list[MetricSeries],
    n_candidates: int = 100,
    accounting: Accounting = "repage",
) -> list[tuple[float, float]]:
    """(pages per series-day, detection rate) at each candidate threshold."""
    if not len(scores):
        return []
    peaks = event_peaks(scores, index, series)
    days = max(normal_minutes(series) / MINUTES_PER_DAY, 1e-9)
    return [
        (
            false_alarms(scores, index, threshold, accounting) / days,
            float((peaks >= threshold).mean()) if len(peaks) else 0.0,
        )
        for threshold in threshold_candidates(scores, n_candidates)
    ]


def curve_auc(
    curve: list[tuple[float, float]],
    budgets: tuple[float, float] = CURVE_BUDGETS,
    steps: int = 36,
) -> float:
    """Mean, over page budgets in the range, of the best detection each one allows.

    A threshold-free summary: a detector that only detects well by paging constantly
    scores badly across the range, and one that is good at a single budget but falls
    apart either side of it no longer looks as good as its best point.
    """
    if not curve:
        return 0.0
    pages = np.array([point[0] for point in curve])
    detection = np.array([point[1] for point in curve])
    grid = np.linspace(budgets[0], budgets[1], steps)
    best = [
        float(detection[pages <= budget].max()) if (pages <= budget).any() else 0.0
        for budget in grid
    ]
    return float(np.mean(best))


def paired_delta(
    candidate: list[EventOutcome],
    incumbent: list[EventOutcome],
    n_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> tuple[float, tuple[float, float]]:
    """Detection rate difference over the same events, with a paired bootstrap interval.

    Resampling events jointly for both detectors keeps the comparison on equal terms:
    an easy draw of incidents is easy for both.
    """
    if [o.event_id for o in candidate] != [o.event_id for o in incumbent]:
        raise ValueError("paired comparison needs the same events in the same order")
    difference = np.array(
        [float(a.detected) - float(b.detected) for a, b in zip(candidate, incumbent, strict=True)]
    )
    if not len(difference):
        return 0.0, (float("nan"), float("nan"))
    return float(difference.mean()), _bootstrap_ci(difference, np.mean, n_samples, seed)


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
    pages = paged_alarms(scores, index, threshold)
    duty, worst_duty = alarm_duty(scores, index, threshold)
    pre_alarmed = np.array([o.pre_alarmed for o in outcomes], dtype=bool)

    # Series are the independent unit for a false alarm rate, so the interval is
    # resampled over series rather than over windows.
    per_series_rate = []
    per_series_pages = []
    for position, item in enumerate(series):
        rows = np.flatnonzero(index.series_index == position)
        if not len(rows):
            continue
        subset = index.subset(rows)
        item_days = max(int((item.point_labels == 0).sum()) / MINUTES_PER_DAY, 1e-9)
        per_series_rate.append(false_alarm_episodes(scores[rows], subset, threshold) / item_days)
        per_series_pages.append(paged_alarms(scores[rows], subset, threshold) / item_days)

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
        pages_per_day=pages / days,
        pages_per_day_ci=_bootstrap_ci(np.array(per_series_pages, dtype=float), np.mean, seed=seed),
        alarm_duty=duty,
        worst_alarm_duty=worst_duty,
        pre_alarmed_share=float(pre_alarmed[detected].mean()) if detected.any() else 0.0,
        fresh_detection_rate=float((detected & ~pre_alarmed).mean()) if len(detected) else 0.0,
        curve_auc=curve_auc(detection_curve(scores, index, series)),
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
    accounting: Accounting = "repage",
) -> float:
    """Pick the most sensitive threshold that stays inside a false alarm budget.

    This is how an alerting threshold is chosen in practice: the tolerable page rate is
    fixed first, and sensitivity is whatever that budget allows. Tuning instead for best
    window F1 optimises a quantity nobody is on call for.

    The budget is in pages as a person receives them, re-paging included, unless
    ``accounting="episodes"`` asks for the old count.
    """
    if not len(scores):
        return float("inf")

    candidates = threshold_candidates(scores, n_candidates)
    days = max(normal_minutes(series) / MINUTES_PER_DAY, 1e-9)

    affordable = [
        threshold
        for threshold in candidates
        if false_alarms(scores, index, threshold, accounting) / days <= budget_per_day
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
