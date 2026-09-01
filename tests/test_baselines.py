"""Each baseline must rank an injected anomaly above quiet traffic."""

import numpy as np
import pytest

from drdoom.data.schema import MetricSeries
from drdoom.data.windows import Scaler, build_index
from drdoom.detect.baselines import (
    EwmaResidual,
    IsolationForestDetector,
    MaxDeviation,
    NaiveResidual,
    WindowSpread,
    all_baselines,
    summarise_windows,
)


def quiet_series(series_id: str = "a", n_steps: int = 4000, seed: int = 0) -> MetricSeries:
    rng = np.random.default_rng(seed)
    return MetricSeries(
        source="test",
        series_id=series_id,
        values=rng.normal(loc=[50.0, 5.0], scale=[2.0, 0.5], size=(n_steps, 2)).astype(np.float32),
        point_labels=np.zeros(n_steps, dtype=np.int8),
        event_ids=np.full(n_steps, -1, dtype=np.int32),
        feature_names=["a", "b"],
    )


def fitted(detector, train):
    index = build_index([train], window_size=60, stride=20)
    return detector.fit([train], index, Scaler.fit([train]))


@pytest.mark.parametrize("detector", all_baselines(), ids=lambda d: d.name)
def test_baseline_ranks_a_level_shift_above_quiet_traffic(detector) -> None:
    train = quiet_series(seed=0)
    probe = quiet_series("b", n_steps=2000, seed=1)
    probe.values[900:1100, 0] += 40.0

    fitted(detector, train)
    index = build_index([probe], window_size=60, stride=20)
    scores = detector.score([probe], index)

    disturbed = (index.start < 1100) & (index.start + 60 > 900)
    assert scores[disturbed].max() > np.percentile(scores[~disturbed], 99)


@pytest.mark.parametrize("detector", all_baselines(), ids=lambda d: d.name)
def test_baseline_returns_one_score_per_window(detector) -> None:
    train = quiet_series()
    fitted(detector, train)
    index = build_index([train], window_size=60, stride=20)

    scores = detector.score([train], index)

    assert scores.shape == (len(index),)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("detector", all_baselines(), ids=lambda d: d.name)
def test_scoring_before_fitting_is_an_error(detector) -> None:
    series = quiet_series()
    index = build_index([series], window_size=60, stride=20)

    with pytest.raises(RuntimeError, match="fitted"):
        detector.score([series], index)


def test_window_spread_reacts_to_variance_not_level() -> None:
    train = quiet_series(seed=0)
    probe = quiet_series("b", n_steps=2000, seed=2)
    probe.values[:, 0] += 30.0  # a constant offset changes level but not spread

    detector = WindowSpread()
    fitted(detector, train)
    index = build_index([probe], window_size=60, stride=20)
    baseline_index = build_index([quiet_series("c", 2000, seed=2)], window_size=60, stride=20)

    shifted = detector.score([probe], index)
    unshifted = detector.score([quiet_series("c", 2000, seed=2)], baseline_index)

    assert np.allclose(shifted, unshifted, atol=1e-4)


def test_max_deviation_reacts_to_level() -> None:
    train = quiet_series(seed=0)
    probe = quiet_series("b", n_steps=2000, seed=2)
    probe.values[:, 0] += 30.0

    detector = MaxDeviation()
    fitted(detector, train)
    index = build_index([probe], window_size=60, stride=20)

    assert detector.score([probe], index).min() > 5.0


def test_naive_residual_reacts_to_an_abrupt_step() -> None:
    train = quiet_series(seed=0)
    probe = quiet_series("b", n_steps=2000, seed=3)
    probe.values[1000:, 0] += 60.0

    detector = NaiveResidual()
    fitted(detector, train)
    index = build_index([probe], window_size=60, stride=20)
    scores = detector.score([probe], index)

    crossing = (index.start < 1000) & (index.start + 60 > 1000)
    assert scores[crossing].max() == scores.max()


def test_ewma_alpha_is_validated() -> None:
    with pytest.raises(ValueError, match="alpha"):
        EwmaResidual(alpha=0.0)


def test_summary_features_have_four_statistics_per_feature() -> None:
    windows = np.zeros((7, 60, 3), dtype=np.float32)

    assert summarise_windows(windows).shape == (7, 12)


def test_isolation_forest_is_reproducible_for_a_seed() -> None:
    train = quiet_series(seed=0)
    index = build_index([train], window_size=60, stride=20)
    scaler = Scaler.fit([train])

    first = IsolationForestDetector(seed=5).fit([train], index, scaler).score([train], index)
    second = IsolationForestDetector(seed=5).fit([train], index, scaler).score([train], index)

    assert np.allclose(first, second)


def test_empty_index_scores_to_an_empty_array() -> None:
    train = quiet_series()
    detector = MaxDeviation()
    fitted(detector, train)
    index = build_index([], window_size=60, stride=20)

    assert detector.score([train], index).shape == (0,)
