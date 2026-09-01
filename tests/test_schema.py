"""Event extraction has to agree with the point labels it was derived from."""

import numpy as np
import pytest

from drdoom.data.schema import (
    NORMAL_EVENT_ID,
    AnomalyEvent,
    MetricSeries,
    event_id_track,
    find_label_runs,
)


def test_find_label_runs_returns_half_open_spans() -> None:
    labels = np.array([0, 1, 1, 0, 0, 1, 0])

    assert find_label_runs(labels) == [(1, 3), (5, 6)]


def test_find_label_runs_handles_runs_at_both_edges() -> None:
    assert find_label_runs(np.array([1, 1, 0, 1])) == [(0, 2), (3, 4)]


def test_find_label_runs_on_all_normal_series() -> None:
    assert find_label_runs(np.zeros(10, dtype=np.int8)) == []


def test_event_track_marks_exactly_the_event_spans() -> None:
    events = [
        AnomalyEvent(event_id=7, source="s", series_id="a", start=2, end=4),
        AnomalyEvent(event_id=9, source="s", series_id="a", start=6, end=7),
    ]

    track = event_id_track(8, events)

    assert track.tolist() == [-1, -1, 7, 7, -1, -1, 9, -1]
    assert int((track != NORMAL_EVENT_ID).sum()) == 3


def test_empty_event_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty span"):
        AnomalyEvent(event_id=0, source="s", series_id="a", start=5, end=5)


def test_series_rejects_mismatched_annotation_lengths() -> None:
    with pytest.raises(ValueError, match="point_labels"):
        MetricSeries(
            source="s",
            series_id="a",
            values=np.zeros((10, 3), dtype=np.float32),
            point_labels=np.zeros(9, dtype=np.int8),
            event_ids=np.zeros(10, dtype=np.int32),
        )


def test_series_rejects_wrong_number_of_feature_names() -> None:
    with pytest.raises(ValueError, match="feature names"):
        MetricSeries(
            source="s",
            series_id="a",
            values=np.zeros((4, 3), dtype=np.float32),
            point_labels=np.zeros(4, dtype=np.int8),
            event_ids=np.zeros(4, dtype=np.int32),
            feature_names=["a", "b"],
        )
