"""Saved series must come back identical, events included."""

import numpy as np

from drdoom.data import store, synthetic
from drdoom.data.schema import AnomalyEvent, MetricSeries, event_id_track


def test_round_trip_preserves_values_labels_and_events(tmp_path) -> None:
    original = synthetic.generate(n_scenarios=3, days=1)
    path = tmp_path / "split"

    store.save_series(path, original)
    restored = store.load_series(path)

    assert len(restored) == len(original)
    for before, after in zip(original, restored, strict=True):
        assert after.series_id == before.series_id
        assert np.array_equal(after.values, before.values)
        assert np.array_equal(after.point_labels, before.point_labels)
        assert np.array_equal(after.event_ids, before.event_ids)
        assert [e.event_id for e in after.events] == [e.event_id for e in before.events]
        assert [e.label for e in after.events] == [e.label for e in before.events]


def test_round_trip_preserves_event_dimensions(tmp_path) -> None:
    events = [AnomalyEvent(event_id=1, source="smd", series_id="m", start=2, end=6, dims=(3, 9))]
    series = [
        MetricSeries(
            source="smd",
            series_id="m",
            values=np.zeros((10, 2), dtype=np.float32),
            point_labels=np.zeros(10, dtype=np.int8),
            event_ids=event_id_track(10, events),
            events=events,
            feature_names=["a", "b"],
        )
    ]
    path = tmp_path / "split"

    store.save_series(path, series)

    assert store.load_series(path)[0].events[0].dims == (3, 9)


def test_series_of_differing_lengths_are_separated_correctly(tmp_path) -> None:
    original = synthetic.generate(n_scenarios=2, days=1)
    original[0] = MetricSeries(
        source=original[0].source,
        series_id=original[0].series_id,
        values=original[0].values[:500],
        point_labels=original[0].point_labels[:500],
        event_ids=original[0].event_ids[:500],
        events=[],
        feature_names=original[0].feature_names,
    )
    path = tmp_path / "split"

    store.save_series(path, original)
    restored = store.load_series(path)

    assert [item.n_timesteps for item in restored] == [item.n_timesteps for item in original]
