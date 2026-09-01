"""Window indexing must respect series boundaries; the scaler must round-trip."""

import numpy as np
import pytest

from drdoom.data import windows
from drdoom.data.schema import NORMAL_EVENT_ID, AnomalyEvent, MetricSeries, event_id_track


def make_series(series_id: str, n_steps: int, events: list[AnomalyEvent]) -> MetricSeries:
    labels = np.zeros(n_steps, dtype=np.int8)
    for event in events:
        labels[event.start : event.end] = 1
    return MetricSeries(
        source="test",
        series_id=series_id,
        values=np.arange(n_steps * 2, dtype=np.float32).reshape(n_steps, 2),
        point_labels=labels,
        event_ids=event_id_track(n_steps, events),
        events=events,
        feature_names=["a", "b"],
    )


def test_windows_never_span_two_series() -> None:
    first = make_series("a", 100, [])
    second = make_series("b", 100, [])

    index = windows.build_index([first, second], window_size=60, stride=10)

    for position, start in zip(index.series_index, index.start, strict=True):
        assert start + index.window_size <= [first, second][position].n_timesteps


def test_series_shorter_than_the_window_are_skipped() -> None:
    index = windows.build_index([make_series("a", 30, [])], window_size=60, stride=10)

    assert len(index) == 0


def test_window_takes_the_event_it_overlaps_most() -> None:
    events = [
        AnomalyEvent(event_id=1, source="test", series_id="a", start=0, end=10),
        AnomalyEvent(event_id=2, source="test", series_id="a", start=10, end=60),
    ]
    series = make_series("a", 120, events)

    index = windows.build_index([series], window_size=60, stride=60)

    assert index.event_id[0] == 2
    assert index.label[0] == 1


def test_windows_clear_of_any_event_are_labelled_normal() -> None:
    events = [AnomalyEvent(event_id=1, source="test", series_id="a", start=0, end=30)]
    index = windows.build_index([make_series("a", 180, events)], window_size=60, stride=60)

    assert index.label.tolist() == [1, 0, 0]
    assert index.event_id.tolist() == [1, NORMAL_EVENT_ID, NORMAL_EVENT_ID]


def test_normal_only_drops_every_anomalous_window() -> None:
    events = [AnomalyEvent(event_id=1, source="test", series_id="a", start=0, end=30)]
    index = windows.build_index([make_series("a", 180, events)], window_size=60, stride=60)

    assert index.normal_only().label.sum() == 0


def test_materialised_windows_match_the_source_rows() -> None:
    series = make_series("a", 200, [])
    index = windows.build_index([series], window_size=60, stride=25)

    tensor = windows.materialise([series], index)

    assert tensor.shape == (len(index), 60, 2)
    for row, start in enumerate(index.start):
        assert np.array_equal(tensor[row], series.values[start : start + 60])


def test_batched_iteration_covers_every_window_once() -> None:
    series = make_series("a", 500, [])
    index = windows.build_index([series], window_size=60, stride=10)

    seen = sum(len(labels) for _, labels in windows.iter_batches([series], index, batch_size=7))

    assert seen == len(index)


def test_invalid_window_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        windows.build_index([make_series("a", 100, [])], window_size=0, stride=1)


def test_scaler_standardises_the_data_it_was_fitted_on() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(loc=[100.0, 0.5], scale=[20.0, 0.1], size=(2000, 2)).astype(np.float32)
    series = MetricSeries(
        source="test",
        series_id="a",
        values=values,
        point_labels=np.zeros(2000, dtype=np.int8),
        event_ids=np.full(2000, NORMAL_EVENT_ID, dtype=np.int32),
        feature_names=["a", "b"],
    )

    scaled = windows.Scaler.fit([series]).transform(values)

    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-3)


def test_scaler_round_trips_through_disk(tmp_path) -> None:
    series = make_series("a", 300, [])
    scaler = windows.Scaler.fit([series])
    path = tmp_path / "scaler.npz"
    scaler.save(path)

    restored = windows.Scaler.load(path, expected_features=["a", "b"])

    assert np.allclose(restored.means, scaler.means)
    assert np.allclose(restored.stds, scaler.stds)
    assert restored.feature_names == ("a", "b")


def test_loading_a_scaler_with_a_different_feature_order_fails(tmp_path) -> None:
    path = tmp_path / "scaler.npz"
    windows.Scaler.fit([make_series("a", 300, [])]).save(path)

    with pytest.raises(ValueError, match="feature order"):
        windows.Scaler.load(path, expected_features=["b", "a"])


def test_inverse_transform_recovers_the_original_values() -> None:
    series = make_series("a", 300, [])
    scaler = windows.Scaler.fit([series])

    recovered = scaler.inverse_transform(scaler.transform(series.values))

    assert np.allclose(recovered, series.values, atol=1e-2)


def test_constant_columns_do_not_produce_infinities() -> None:
    series = MetricSeries(
        source="test",
        series_id="a",
        values=np.column_stack(
            [np.full(100, 3.0, dtype=np.float32), np.arange(100, dtype=np.float32)]
        ),
        point_labels=np.zeros(100, dtype=np.int8),
        event_ids=np.full(100, NORMAL_EVENT_ID, dtype=np.int32),
        feature_names=["constant", "ramp"],
    )

    scaled = windows.Scaler.fit([series]).transform(series.values)

    assert np.isfinite(scaled).all()
    assert np.allclose(scaled[:, 0], 0.0)


def test_scaler_rejects_a_mismatched_feature_count() -> None:
    scaler = windows.Scaler.fit([make_series("a", 100, [])])

    with pytest.raises(ValueError, match="expected 2 features"):
        scaler.transform(np.zeros((5, 3), dtype=np.float32))
