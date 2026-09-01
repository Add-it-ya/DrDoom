"""Splits must not bisect an incident, and must not correlate a class with a split."""

from collections import Counter

import numpy as np
import pytest

from drdoom.data import splits, synthetic
from drdoom.data.schema import AnomalyEvent, MetricSeries, event_id_track


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


def test_cut_moves_off_an_anomalous_timestep() -> None:
    labels = np.zeros(100, dtype=np.int8)
    labels[40:60] = 1

    assert labels[splits.nearest_normal_cut(labels, 50)] == 0


def test_cut_on_a_normal_timestep_is_unchanged() -> None:
    labels = np.zeros(100, dtype=np.int8)

    assert splits.nearest_normal_cut(labels, 50) == 50


def test_slicing_keeps_only_events_fully_inside_the_window() -> None:
    events = [
        AnomalyEvent(event_id=1, source="test", series_id="a", start=10, end=20),
        AnomalyEvent(event_id=2, source="test", series_id="a", start=90, end=110),
    ]
    series = make_series("a", 200, events)

    sliced = splits.slice_series(series, 0, 100)

    assert [event.event_id for event in sliced.events] == [1]
    assert (sliced.events[0].start, sliced.events[0].end) == (10, 20)


def test_sliced_event_indices_are_rebased_onto_the_slice() -> None:
    events = [AnomalyEvent(event_id=1, source="test", series_id="a", start=120, end=140)]
    sliced = splits.slice_series(make_series("a", 200, events), 100, 200)

    assert (sliced.events[0].start, sliced.events[0].end) == (20, 40)
    assert int(sliced.point_labels[20:40].sum()) == 20


def test_time_based_split_loses_no_events_to_boundaries() -> None:
    series = synthetic.generate(n_scenarios=30, days=4)
    total = sum(len(item.events) for item in series)

    counts = splits.time_based_split(series).event_counts()

    assert sum(counts.values()) == total


def test_held_out_split_assigns_each_series_to_exactly_one_split() -> None:
    series = synthetic.generate(n_scenarios=30, days=4)

    result = splits.held_out_series_split(series)

    groups = [{item.series_id for item in items} for items in result.as_dict().values()]
    assert sum(len(group) for group in groups) == len(series)
    assert set.intersection(*groups) == set()


@pytest.mark.parametrize("strategy", [splits.time_based_split, splits.held_out_series_split])
def test_every_split_contains_every_root_cause(strategy) -> None:
    """A class missing from a split makes any tuning objective blind to part of it."""
    series = synthetic.generate(n_scenarios=60, days=4)

    result = strategy(series)

    for items in result.as_dict().values():
        causes = Counter(event.label for item in items for event in item.events)
        assert set(causes) == set(synthetic.ROOT_CAUSES)


def test_held_out_split_is_deterministic_for_a_seed() -> None:
    series = synthetic.generate(n_scenarios=20, days=4)

    first = splits.held_out_series_split(series, seed=3)
    second = splits.held_out_series_split(series, seed=3)

    assert [item.series_id for item in first.test] == [item.series_id for item in second.test]


def test_impossible_fractions_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty test split"):
        splits.time_based_split([make_series("a", 100, [])], train_frac=0.9, val_frac=0.2)
