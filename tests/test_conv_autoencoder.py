"""The centred convolutional autoencoder and the fusion built on it."""

import numpy as np
import pytest
import torch

from drdoom.data.schema import MetricSeries, event_id_track
from drdoom.data.windows import Scaler, build_index
from drdoom.detect.baselines import NaiveResidual, WindowSpread
from drdoom.detect.conv_autoencoder import (
    CentredConvAutoencoder,
    ConvAutoencoderDetector,
    window_error,
)
from drdoom.detect.fusion import MaxFusion

FEATURES = ["a", "b", "c"]


def quiet_series(n_steps: int = 3000, seed: int = 0) -> MetricSeries:
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)
    values = np.stack(
        [np.sin(t / 15.0), np.cos(t / 20.0), 0.5 * np.sin(t / 7.0)], axis=1
    ) + 0.05 * rng.normal(size=(n_steps, 3))
    return MetricSeries(
        source="test",
        series_id=f"s{seed}",
        values=values.astype(np.float32),
        point_labels=np.zeros(n_steps, dtype=np.int8),
        event_ids=event_id_track(n_steps, []),
        events=[],
        feature_names=FEATURES,
    )


@pytest.fixture(scope="module")
def fitted() -> tuple[ConvAutoencoderDetector, list[MetricSeries], Scaler]:
    series = [quiet_series()]
    scaler = Scaler.fit(series)
    index = build_index(series, window_size=60, stride=10)
    detector = ConvAutoencoderDetector(epochs=3, seed=0).fit(series, index, scaler)
    return detector, series, scaler


def test_the_output_keeps_the_window_shape() -> None:
    model = CentredConvAutoencoder(n_features=38)

    assert model(torch.zeros(4, 60, 38)).shape == (4, 60, 38)


def test_the_score_ignores_a_constant_offset_on_a_metric() -> None:
    """Level drift between periods must not look like an incident."""
    model = CentredConvAutoencoder(n_features=3)
    windows = np.random.default_rng(1).normal(size=(8, 60, 3)).astype(np.float32)
    shifted = windows.copy()
    shifted[:, :, 1] += 25.0

    assert np.allclose(window_error(model, windows), window_error(model, shifted), atol=1e-4)


def test_a_step_inside_the_window_scores_above_quiet_traffic(fitted) -> None:
    detector, _, _ = fitted
    stepped = quiet_series(seed=5)
    stepped.values[1530:1560, 0] += 4.0
    index = build_index([stepped], window_size=60, stride=60)

    scores = detector.score([stepped], index)
    step_window = int(np.flatnonzero(index.start == 1500)[0])

    assert scores[step_window] > np.percentile(np.delete(scores, step_window), 90)


def test_a_checkpoint_round_trips(fitted, tmp_path) -> None:
    detector, series, scaler = fitted
    index = build_index(series, window_size=60, stride=30)
    detector.save(tmp_path / "conv.pt")

    loaded = ConvAutoencoderDetector.load(tmp_path / "conv.pt", scaler)

    assert np.allclose(loaded.score(series, index), detector.score(series, index))
    assert loaded.history == detector.history


def test_a_checkpoint_for_other_metrics_is_refused(fitted, tmp_path) -> None:
    detector, _, _ = fitted
    detector.save(tmp_path / "conv.pt")
    other = quiet_series()
    other.feature_names[:] = ["x", "y", "z"]

    with pytest.raises(ValueError, match="feature order"):
        ConvAutoencoderDetector.load(tmp_path / "conv.pt", Scaler.fit([other]))


def test_the_same_seed_gives_the_same_scores() -> None:
    series = [quiet_series(n_steps=1200)]
    scaler = Scaler.fit(series)
    index = build_index(series, window_size=60, stride=10)

    first = ConvAutoencoderDetector(epochs=2, seed=3).fit(series, index, scaler)
    second = ConvAutoencoderDetector(epochs=2, seed=3).fit(series, index, scaler)

    assert np.array_equal(first.score(series, index), second.score(series, index))


def test_every_epoch_is_scored_on_windows_it_did_not_train_on(fitted) -> None:
    detector, _, _ = fitted

    assert [entry["epoch"] for entry in detector.history] == [1, 2, 3]
    assert all(np.isfinite(entry["holdout"]) for entry in detector.history)


def test_a_window_that_is_not_a_multiple_of_four_is_refused() -> None:
    series = [quiet_series(n_steps=600)]
    with pytest.raises(ValueError, match="multiple of four"):
        ConvAutoencoderDetector(epochs=1).fit(
            series, build_index(series, window_size=62, stride=10), Scaler.fit(series)
        )


# --- fusion -----------------------------------------------------------------------------


def test_fusion_is_scaled_on_training_windows_only() -> None:
    train = [quiet_series(seed=0)]
    scaler = Scaler.fit(train)
    train_index = build_index(train, window_size=60, stride=10)
    members = [WindowSpread().fit(train, train_index, scaler), NaiveResidual()]
    members[1].fit(train, train_index, scaler)

    fusion = MaxFusion(members).fit(train, train_index, scaler)
    median_train = np.median(fusion.score(train, train_index))

    assert fusion.name == "window_spread+naive_residual"
    assert abs(median_train) < 1.0


def test_fusion_takes_the_most_alarmed_member() -> None:
    train = [quiet_series(seed=0)]
    scaler = Scaler.fit(train)
    index = build_index(train, window_size=60, stride=10)
    spread = WindowSpread().fit(train, index, scaler)
    jump = NaiveResidual().fit(train, index, scaler)
    fusion = MaxFusion([spread, jump]).fit(train, index, scaler)

    expected = np.maximum(
        (spread.score(train, index) - fusion.centres[0]) / fusion.spreads[0],
        (jump.score(train, index) - fusion.centres[1]) / fusion.spreads[1],
    )

    assert np.allclose(fusion.score(train, index), expected, atol=1e-5)


def test_a_fusion_of_one_is_refused() -> None:
    with pytest.raises(ValueError):
        MaxFusion([WindowSpread()])
