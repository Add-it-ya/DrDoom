"""Archetype derivation, labelled dataset assembly, and per-incident scoring."""

import json

import numpy as np
import pytest

from drdoom.classify import archetypes as ar
from drdoom.classify import dataset as ds
from drdoom.classify.card import render
from drdoom.classify.features import FeatureSpec
from drdoom.classify.train import (
    ClassifierConfig,
    _class_coverage,
    choose_split,
    event_predictions,
    prepare,
    train,
)
from drdoom.data import synthetic
from drdoom.data.schema import AnomalyEvent, MetricSeries, event_id_track
from drdoom.data.windows import build_index


def event(event_id: int, dims: tuple[int, ...], start: int = 0, end: int = 10) -> AnomalyEvent:
    return AnomalyEvent(
        event_id=event_id, source="smd", series_id="m", start=start, end=end, dims=dims
    )


def test_signature_matrix_treats_dimensions_as_one_based() -> None:
    matrix = ar.signature_matrix([event(0, (1, 3))], n_features=4)

    assert matrix[0].tolist() == [1.0, 0.0, 1.0, 0.0]


def test_dimensions_outside_the_metric_range_are_ignored() -> None:
    matrix = ar.signature_matrix([event(0, (0, 5, 99))], n_features=4)

    assert matrix.sum() == 0.0


def test_clusters_below_the_support_threshold_become_other() -> None:
    narrow = [event(i, (1, 2)) for i in range(30)]
    rare = [event(100 + i, (9,)) for i in range(3)]

    derived = ar.derive(narrow + rare, n_features=10, n_clusters=2, min_support=10)

    assert derived.support[ar.OTHER_LABEL] == 3
    assert all(derived.label_for(e) == ar.OTHER_LABEL for e in rare)


def test_events_with_the_same_signature_share_an_archetype() -> None:
    events = [event(i, (1, 2, 3)) for i in range(20)] + [
        event(50 + i, (7, 8, 9, 10, 11, 12, 13, 14)) for i in range(20)
    ]

    derived = ar.derive(events, n_features=20, n_clusters=2, min_support=5)

    assert derived.label_for(events[0]) == derived.label_for(events[5])
    assert derived.label_for(events[0]) != derived.label_for(events[-1])


def test_breadth_naming_separates_narrow_from_broad() -> None:
    events = [event(i, (1, 2)) for i in range(20)] + [
        event(50 + i, tuple(range(1, 20))) for i in range(20)
    ]

    derived = ar.derive(events, n_features=38, n_clusters=2, min_support=5)

    assert "narrow_signature" in derived.support
    assert "broad_signature" in derived.support


def test_events_without_interpretation_dimensions_fall_through_to_other() -> None:
    events = [event(i, (1, 2)) for i in range(20)] + [event(99, ())]

    derived = ar.derive(events, n_features=10, n_clusters=2, min_support=5)

    assert derived.label_for(events[-1]) == ar.OTHER_LABEL


def test_too_few_events_to_cluster_is_handled() -> None:
    derived = ar.derive([event(0, (1,))], n_features=10, n_clusters=6)

    assert derived.support == {ar.OTHER_LABEL: 1}


def make_series(series_id: str, n_steps: int, events: list[AnomalyEvent]) -> MetricSeries:
    labels = np.zeros(n_steps, dtype=np.int8)
    for item in events:
        labels[item.start : item.end] = 1
    rng = np.random.default_rng(0)
    return MetricSeries(
        source="test",
        series_id=series_id,
        values=rng.normal(size=(n_steps, 2)).astype(np.float32),
        point_labels=labels,
        event_ids=event_id_track(n_steps, events),
        events=events,
        feature_names=["a", "b"],
    )


def test_dataset_uses_only_anomalous_windows() -> None:
    events = [
        AnomalyEvent(event_id=1, source="test", series_id="s", start=100, end=300, label="alpha")
    ]
    series = [make_series("s", 1000, events)]
    index = build_index(series, 60, 20)

    data = ds.build(series, index, ("alpha",))

    assert len(data) == int(index.label.sum())
    assert set(data.labels.tolist()) == {0}


def test_every_row_carries_the_event_it_came_from() -> None:
    events = [
        AnomalyEvent(event_id=7, source="test", series_id="s", start=100, end=300, label="alpha")
    ]
    series = [make_series("s", 1000, events)]

    data = ds.build(series, build_index(series, 60, 20), ("alpha",))

    assert set(data.groups.tolist()) == {7}
    assert data.n_events == 1


def test_rows_with_an_unlisted_class_are_dropped() -> None:
    events = [
        AnomalyEvent(event_id=1, source="test", series_id="s", start=100, end=300, label="alpha"),
        AnomalyEvent(event_id=2, source="test", series_id="s", start=500, end=700, label="omega"),
    ]
    series = [make_series("s", 1000, events)]

    data = ds.build(series, build_index(series, 60, 20), ("alpha",))

    assert set(data.groups.tolist()) == {1}


def test_dataset_reports_support_in_events_not_rows() -> None:
    events = [
        AnomalyEvent(event_id=1, source="test", series_id="s", start=100, end=400, label="alpha")
    ]
    series = [make_series("s", 1000, events)]

    data = ds.build(series, build_index(series, 60, 20), ("alpha", "beta"))

    assert data.events_per_class() == {"alpha": 1, "beta": 0}
    assert data.rows_per_class()["alpha"] > 1


def test_a_split_with_no_anomalies_gives_an_empty_matrix() -> None:
    series = [make_series("s", 500, [])]

    data = ds.build(series, build_index(series, 60, 20), ("alpha",))

    assert len(data) == 0
    assert data.features.shape == (0, FeatureSpec(("a", "b")).n_columns)


def test_majority_vote_resolves_one_verdict_per_incident() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    predicted = np.array([0, 0, 1, 1, 1, 0])
    groups = np.array([10, 10, 10, 20, 20, 20])

    truth, verdict = event_predictions(labels, predicted, groups)

    assert truth.tolist() == [0, 1]
    assert verdict.tolist() == [0, 1]


def test_a_wrong_majority_produces_a_wrong_verdict() -> None:
    labels = np.array([1, 1, 1])
    predicted = np.array([0, 0, 1])
    groups = np.array([5, 5, 5])

    truth, verdict = event_predictions(labels, predicted, groups)

    assert truth.tolist() == [1]
    assert verdict.tolist() == [0]


def test_class_coverage_spots_a_split_that_is_missing_a_class() -> None:
    series = synthetic.generate(n_scenarios=20, days=2)
    lookup = ds.event_labels(series)
    names = tuple(sorted(synthetic.ROOT_CAUSES))

    splits, _ = choose_split(series, lookup, names, attempts=20)

    assert _class_coverage(splits, lookup, names) >= 1


def test_chosen_split_is_deterministic() -> None:
    series = synthetic.generate(n_scenarios=20, days=2)
    lookup = ds.event_labels(series)
    names = tuple(sorted(synthetic.ROOT_CAUSES))

    first, seed_a = choose_split(series, lookup, names, attempts=20)
    second, seed_b = choose_split(series, lookup, names, attempts=20)

    assert seed_a == seed_b
    assert [s.series_id for s in first.test] == [s.series_id for s in second.test]


def test_prepared_synthetic_labels_are_the_four_causes() -> None:
    prepared = prepare(ClassifierConfig(source="synthetic", n_scenarios=20, days=2))

    assert set(prepared.label_names) == set(synthetic.ROOT_CAUSES)
    assert prepared.archetypes is None


def test_training_writes_a_model_the_feature_spec_can_be_checked_against(tmp_path) -> None:
    config = ClassifierConfig(
        source="synthetic",
        n_scenarios=24,
        days=2,
        n_trials=2,
        n_folds=3,
        stride=40,
        models_root=tmp_path,
    )

    summary = train(config)

    spec = FeatureSpec.load(config.model_dir / "features.json")
    assert spec.metric_names == tuple(synthetic.FEATURE_NAMES)
    assert spec.columns[0] == "latency_ms_mean"
    assert (config.model_dir / "model.json").is_file()
    assert json.loads((config.model_dir / "labels.json").read_text()) == list(summary["classes"])
    assert summary["test_events"] > 0
    assert 0.0 <= summary["event_macro_f1"] <= 1.0


def test_inference_features_line_up_with_the_trained_columns(tmp_path) -> None:
    """The guard against the silent failure: same order at fit time and at predict time."""
    import xgboost as xgb

    config = ClassifierConfig(
        source="synthetic",
        n_scenarios=24,
        days=2,
        n_trials=2,
        n_folds=3,
        stride=40,
        models_root=tmp_path,
    )
    train(config)

    spec = FeatureSpec.load(config.model_dir / "features.json")
    model = xgb.XGBClassifier()
    model.load_model(config.model_dir / "model.json")

    spec.check(synthetic.FEATURE_NAMES)
    assert model.n_features_in_ == spec.n_columns
    with pytest.raises(ValueError, match="metric order"):
        spec.check(list(reversed(synthetic.FEATURE_NAMES)))


def test_card_reports_the_pooling_gain_and_the_synthetic_gap() -> None:
    summaries = [
        {
            "source": "smd",
            "classes": ["a", "b"],
            "test_events": 59,
            "test_windows": 1001,
            "event_macro_f1": 0.827,
            "event_macro_f1_ci": [0.73, 0.92],
            "event_accuracy": 0.83,
            "window_macro_f1": 0.580,
            "best_cv_macro_f1": 0.744,
            "min_events_per_class_per_split": 28,
            "events_per_class": {"a": 29, "b": 30},
            "confusion": [[29, 0], [10, 20]],
            "feature_importance": {"m24_std": 0.1},
        },
        {
            "source": "synthetic",
            "classes": ["c", "d"],
            "test_events": 300,
            "test_windows": 5014,
            "event_macro_f1": 0.997,
            "event_macro_f1_ci": [0.99, 1.0],
            "event_accuracy": 0.997,
            "window_macro_f1": 0.953,
            "best_cv_macro_f1": 0.945,
            "min_events_per_class_per_split": 43,
            "events_per_class": {"c": 150, "d": 150},
            "confusion": [[150, 0], [1, 149]],
            "feature_importance": {"queue_depth_std": 0.3},
        },
    ]

    text = render(summaries)

    assert "+0.247" in text or "+0.246" in text
    assert "separable by construction" in text
    assert "## smd" in text and "## synthetic" in text
