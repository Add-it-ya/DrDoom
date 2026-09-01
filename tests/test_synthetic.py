"""The generator's job is balanced, reproducible, non-overlapping labelled events."""

from collections import Counter

import numpy as np

from drdoom.data import synthetic
from drdoom.data.schema import find_label_runs


def test_scenario_is_reproducible_for_a_given_seed() -> None:
    first = synthetic.generate_scenario(3, seed=7)
    second = synthetic.generate_scenario(3, seed=7)

    assert np.array_equal(first.values, second.values)
    assert [event.start for event in first.events] == [event.start for event in second.events]


def test_different_scenarios_differ() -> None:
    assert not np.array_equal(
        synthetic.generate_scenario(1).values, synthetic.generate_scenario(2).values
    )


def test_root_causes_are_balanced_across_the_corpus() -> None:
    series = synthetic.generate(n_scenarios=40, days=4)
    counts = Counter(event.label for scenario in series for event in scenario.events)

    assert set(counts) == set(synthetic.ROOT_CAUSES)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_events_do_not_overlap_and_match_the_point_labels() -> None:
    scenario = synthetic.generate_scenario(11)

    runs = find_label_runs(scenario.point_labels)
    assert runs == [(event.start, event.end) for event in scenario.events]


def test_event_ids_are_unique_across_the_corpus() -> None:
    events = [event for scenario in synthetic.generate(n_scenarios=25) for event in scenario.events]

    assert len({event.event_id for event in events}) == len(events)


def test_metrics_stay_within_physical_bounds() -> None:
    scenario = synthetic.generate_scenario(5)

    assert scenario.values[:, 0].min() >= 5.0
    assert scenario.values[:, 1].min() >= 0.0 and scenario.values[:, 1].max() <= 100.0
    assert scenario.values[:, 2].min() >= 0.0 and scenario.values[:, 2].max() <= 100.0
    assert scenario.values[:, 3].min() >= 0.0


def test_severity_varies_so_some_events_are_subtle() -> None:
    series = synthetic.generate(n_scenarios=30)
    lifts = []
    for scenario in series:
        normal = scenario.values[scenario.point_labels == 0].mean(axis=0)
        for event in scenario.events:
            window = scenario.values[event.start : event.end].mean(axis=0)
            lifts.append(float(np.abs(window - normal).max()))

    assert min(lifts) < np.median(lifts) / 2
