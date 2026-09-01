"""Event-level metrics: detection, delay, and how often we page for nothing."""

import numpy as np
import pytest

from drdoom.data.schema import AnomalyEvent, MetricSeries, event_id_track
from drdoom.data.windows import build_index
from drdoom.detect import evaluate as ev


def make_series(series_id: str, n_steps: int, events: list[AnomalyEvent]) -> MetricSeries:
    labels = np.zeros(n_steps, dtype=np.int8)
    for event in events:
        labels[event.start : event.end] = 1
    return MetricSeries(
        source="test",
        series_id=series_id,
        values=np.zeros((n_steps, 2), dtype=np.float32),
        point_labels=labels,
        event_ids=event_id_track(n_steps, events),
        events=events,
        feature_names=["a", "b"],
    )


def test_event_is_detected_when_an_overlapping_window_fires() -> None:
    events = [AnomalyEvent(event_id=1, source="test", series_id="a", start=100, end=160)]
    series = [make_series("a", 400, events)]
    index = build_index(series, window_size=60, stride=20)
    scores = np.zeros(len(index))
    overlapping = np.flatnonzero((index.start < 160) & (index.start + 60 > 100))
    scores[overlapping[0]] = 10.0

    outcomes = ev.event_outcomes(scores, index, series, threshold=1.0)

    assert outcomes[0].detected


def test_event_is_missed_when_only_unrelated_windows_fire() -> None:
    events = [AnomalyEvent(event_id=1, source="test", series_id="a", start=100, end=160)]
    series = [make_series("a", 400, events)]
    index = build_index(series, window_size=60, stride=20)
    scores = np.zeros(len(index))
    scores[index.start > 300] = 10.0

    outcomes = ev.event_outcomes(scores, index, series, threshold=1.0)

    assert not outcomes[0].detected
    assert outcomes[0].minutes_to_detect is None


def test_delay_is_measured_from_event_start_to_the_window_end() -> None:
    events = [AnomalyEvent(event_id=1, source="test", series_id="a", start=100, end=200)]
    series = [make_series("a", 400, events)]
    index = build_index(series, window_size=60, stride=20)
    scores = np.zeros(len(index))
    # The window starting at 80 ends at 140, so the alert lands 40 minutes in.
    scores[np.flatnonzero(index.start == 80)] = 10.0

    outcomes = ev.event_outcomes(scores, index, series, threshold=1.0)

    assert outcomes[0].minutes_to_detect == 40.0


def test_delay_uses_the_earliest_firing_window() -> None:
    events = [AnomalyEvent(event_id=1, source="test", series_id="a", start=100, end=300)]
    series = [make_series("a", 500, events)]
    index = build_index(series, window_size=60, stride=20)
    scores = np.zeros(len(index))
    scores[np.flatnonzero((index.start == 80) | (index.start == 200))] = 10.0

    outcomes = ev.event_outcomes(scores, index, series, threshold=1.0)

    assert outcomes[0].minutes_to_detect == 40.0


def test_delay_is_never_negative_for_a_window_that_starts_early() -> None:
    events = [AnomalyEvent(event_id=1, source="test", series_id="a", start=100, end=300)]
    series = [make_series("a", 500, events)]
    index = build_index(series, window_size=60, stride=20)
    scores = np.full(len(index), 10.0)

    outcomes = ev.event_outcomes(scores, index, series, threshold=1.0)

    assert outcomes[0].minutes_to_detect >= 0


def test_events_are_tracked_per_series() -> None:
    first = make_series("a", 300, [AnomalyEvent(1, "test", "a", 100, 160)])
    second = make_series("b", 300, [AnomalyEvent(2, "test", "b", 100, 160)])
    series = [first, second]
    index = build_index(series, window_size=60, stride=20)
    scores = np.where(index.series_index == 0, 10.0, 0.0)

    outcomes = ev.event_outcomes(scores, index, series, threshold=1.0)

    assert [o.detected for o in outcomes] == [True, False]


def test_consecutive_false_windows_count_as_one_episode() -> None:
    series = [make_series("a", 400, [])]
    index = build_index(series, window_size=60, stride=20)
    scores = np.zeros(len(index))
    scores[2:6] = 10.0

    assert ev.false_alarm_episodes(scores, index, threshold=1.0) == 1


def test_separated_false_windows_count_as_separate_episodes() -> None:
    series = [make_series("a", 800, [])]
    index = build_index(series, window_size=60, stride=20)
    scores = np.zeros(len(index))
    scores[[1, 2, 10, 20]] = 10.0

    assert ev.false_alarm_episodes(scores, index, threshold=1.0) == 3


def test_episodes_do_not_run_across_two_series() -> None:
    series = [make_series("a", 200, []), make_series("b", 200, [])]
    index = build_index(series, window_size=60, stride=20)
    scores = np.zeros(len(index))
    last_of_first = np.flatnonzero(index.series_index == 0)[-1]
    scores[[last_of_first, last_of_first + 1]] = 10.0

    assert ev.false_alarm_episodes(scores, index, threshold=1.0) == 2


def test_windows_overlapping_an_event_are_not_false_alarms() -> None:
    events = [AnomalyEvent(event_id=1, source="test", series_id="a", start=100, end=160)]
    series = [make_series("a", 400, events)]
    index = build_index(series, window_size=60, stride=20)
    scores = np.full(len(index), 10.0)

    episodes = ev.false_alarm_episodes(scores, index, threshold=1.0)

    assert episodes < len(index)


def test_threshold_selection_respects_the_budget() -> None:
    rng = np.random.default_rng(0)
    series = [make_series("a", 20000, [])]
    index = build_index(series, window_size=60, stride=20)
    scores = rng.normal(size=len(index))

    threshold = ev.select_threshold(scores, index, series, budget_per_day=1.0)
    days = ev.normal_minutes(series) / ev.MINUTES_PER_DAY

    assert ev.false_alarm_episodes(scores, index, threshold) / days <= 1.0


def test_a_looser_budget_gives_a_lower_threshold() -> None:
    rng = np.random.default_rng(1)
    series = [make_series("a", 20000, [])]
    index = build_index(series, window_size=60, stride=20)
    scores = rng.normal(size=len(index))

    strict = ev.select_threshold(scores, index, series, budget_per_day=0.5)
    loose = ev.select_threshold(scores, index, series, budget_per_day=5.0)

    assert loose <= strict


def test_report_carries_intervals_and_serialises() -> None:
    events = [AnomalyEvent(event_id=1, source="test", series_id="a", start=100, end=200)]
    series = [make_series("a", 2000, events)]
    index = build_index(series, window_size=60, stride=20)
    rng = np.random.default_rng(2)
    scores = rng.normal(size=len(index))
    scores[index.label == 1] += 5.0

    report = ev.evaluate("demo", scores, index, series, threshold=2.0)
    row = report.as_row()

    assert report.n_events == 1
    assert len(report.detection_rate_ci) == 2
    assert row["detector"] == "demo"
    assert set(row) >= {"detection_rate", "false_alarms_per_day", "window_f1", "pr_auc"}


def test_perfect_scores_give_full_detection_and_no_false_alarms() -> None:
    events = [AnomalyEvent(event_id=1, source="test", series_id="a", start=500, end=700)]
    series = [make_series("a", 2000, events)]
    index = build_index(series, window_size=60, stride=20)
    scores = index.label.astype(float)

    report = ev.evaluate("oracle", scores, index, series, threshold=0.5)

    assert report.detection_rate == 1.0
    assert report.false_alarms_per_day == 0.0
    assert report.window_precision == 1.0


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    sample = np.array([True] * 70 + [False] * 30)

    low, high = ev._bootstrap_ci(sample, np.mean, n_samples=500)

    assert low < sample.mean() < high


def test_metrics_are_defined_when_a_split_has_no_anomalies() -> None:
    series = [make_series("a", 1000, [])]
    index = build_index(series, window_size=60, stride=20)
    scores = np.zeros(len(index))

    report = ev.evaluate("demo", scores, index, series, threshold=1.0)

    assert report.n_events == 0
    assert report.detection_rate == 0.0
    assert np.isnan(report.roc_auc)


@pytest.mark.parametrize("budget", [0.1, 1.0, 10.0])
def test_threshold_selection_is_finite_for_any_budget(budget: float) -> None:
    rng = np.random.default_rng(3)
    series = [make_series("a", 5000, [])]
    index = build_index(series, window_size=60, stride=20)
    scores = rng.normal(size=len(index))

    assert np.isfinite(ev.select_threshold(scores, index, series, budget_per_day=budget))
