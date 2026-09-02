"""Feature extraction must agree with the declared column order, exactly.

This is the failure that produces no exception: a feature vector built in a different
order than the model was trained on still has the right shape, so nothing complains and
every prediction is quietly wrong.
"""

import numpy as np
import pytest

from drdoom.classify.features import (
    STATISTICS,
    FeatureSpec,
    extract_features,
    feature_order,
)


def ramped_window(n_metrics: int = 3, length: int = 60) -> np.ndarray:
    """Metric i is a rising ramp offset by i * 100, so every column is distinguishable."""
    steps = np.arange(length, dtype=np.float32)
    return np.stack([steps + metric * 100.0 for metric in range(n_metrics)], axis=1)[None, :, :]


def test_column_order_is_metric_major_and_statistic_minor() -> None:
    columns = feature_order(["a", "b"])

    assert columns == [
        "a_mean",
        "a_std",
        "a_min",
        "a_max",
        "a_slope",
        "a_half_diff",
        "b_mean",
        "b_std",
        "b_min",
        "b_max",
        "b_slope",
        "b_half_diff",
    ]


def test_extracted_values_land_in_their_declared_columns() -> None:
    metrics = ["m0", "m1", "m2"]
    windows = ramped_window(len(metrics))

    values = dict(zip(feature_order(metrics), extract_features(windows)[0], strict=True))

    for index, name in enumerate(metrics):
        offset = index * 100.0
        assert values[f"{name}_min"] == pytest.approx(offset)
        assert values[f"{name}_max"] == pytest.approx(offset + 59.0)
        assert values[f"{name}_mean"] == pytest.approx(offset + 29.5)
        assert values[f"{name}_slope"] == pytest.approx(1.0, abs=1e-4)
        assert values[f"{name}_half_diff"] == pytest.approx(30.0, abs=1e-3)


def test_column_count_matches_the_specification() -> None:
    spec = FeatureSpec(metric_names=("a", "b", "c"))
    windows = ramped_window(3)

    assert extract_features(windows).shape == (1, spec.n_columns)
    assert spec.n_columns == 3 * len(STATISTICS)


def test_reordering_the_metrics_reorders_the_features() -> None:
    windows = ramped_window(2)
    swapped = windows[:, :, ::-1]

    original = extract_features(windows)[0]
    reordered = extract_features(swapped)[0]

    block = len(STATISTICS)
    assert np.allclose(original[:block], reordered[block:])
    assert not np.allclose(original, reordered)


def test_slope_is_negative_for_a_falling_metric() -> None:
    steps = np.arange(60, dtype=np.float32)
    windows = np.stack([-steps], axis=1)[None, :, :]

    values = dict(zip(feature_order(["m"]), extract_features(windows)[0], strict=True))

    assert values["m_slope"] < 0
    assert values["m_half_diff"] < 0


def test_a_flat_metric_has_no_spread_and_no_trend() -> None:
    windows = np.full((1, 60, 1), 7.0, dtype=np.float32)

    values = dict(zip(feature_order(["m"]), extract_features(windows)[0], strict=True))

    assert values["m_std"] == pytest.approx(0.0)
    assert values["m_slope"] == pytest.approx(0.0, abs=1e-6)
    assert values["m_half_diff"] == pytest.approx(0.0)


def test_wrong_input_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected"):
        extract_features(np.zeros((60, 4), dtype=np.float32))


def test_a_single_timestep_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="two timesteps"):
        extract_features(np.zeros((1, 1, 4), dtype=np.float32))


def test_specification_round_trips_through_disk(tmp_path) -> None:
    spec = FeatureSpec(metric_names=("latency", "errors"))
    path = tmp_path / "features.json"
    spec.save(path)

    restored = FeatureSpec.load(path)

    assert restored == spec
    assert restored.columns == spec.columns


def test_loading_a_specification_whose_columns_drifted_fails(tmp_path) -> None:
    path = tmp_path / "features.json"
    FeatureSpec(metric_names=("a", "b")).save(path)
    payload = path.read_text(encoding="utf-8").replace("a_mean", "a_median")
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="do not match"):
        FeatureSpec.load(path)


def test_check_rejects_a_different_metric_order() -> None:
    spec = FeatureSpec(metric_names=("a", "b"))

    spec.check(["a", "b"])
    with pytest.raises(ValueError, match="metric order"):
        spec.check(["b", "a"])
