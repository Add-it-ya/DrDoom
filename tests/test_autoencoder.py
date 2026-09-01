"""The autoencoder, its checkpoint format, and how a checkpoint gets chosen."""

import numpy as np
import pytest
import torch

from drdoom.data import store
from drdoom.data.build import BuildConfig, build
from drdoom.data.windows import Scaler, build_index
from drdoom.detect.autoencoder import AutoencoderDetector, LSTMAutoencoder, reconstruction_error
from drdoom.detect.train import TrainConfig, selection_score, train


def test_reconstruction_keeps_the_input_shape() -> None:
    model = LSTMAutoencoder(n_features=4, hidden_size=8)

    output = model(torch.randn(5, 60, 4))

    assert output.shape == (5, 60, 4)


def test_bottleneck_is_the_hidden_state_not_the_sequence() -> None:
    model = LSTMAutoencoder(n_features=38, hidden_size=16)

    assert model.encoder.hidden_size == 16
    assert model.decoder.input_size == 16
    assert model.output.out_features == 38


def test_reconstruction_error_is_one_number_per_window() -> None:
    model = LSTMAutoencoder(n_features=3, hidden_size=8)
    windows = np.random.default_rng(0).normal(size=(7, 20, 3)).astype(np.float32)

    errors = reconstruction_error(model, windows, torch.device("cpu"))

    assert errors.shape == (7,)
    assert (errors >= 0).all()


def test_an_untrained_model_reconstructs_zero_input_best() -> None:
    model = LSTMAutoencoder(n_features=2, hidden_size=8)
    device = torch.device("cpu")
    quiet = np.zeros((4, 30, 2), dtype=np.float32)
    loud = np.full((4, 30, 2), 25.0, dtype=np.float32)

    assert (
        reconstruction_error(model, quiet, device).mean()
        < reconstruction_error(model, loud, device).mean()
    )


def test_checkpoint_round_trips(tmp_path) -> None:
    model = LSTMAutoencoder(n_features=2, hidden_size=8)
    detector = AutoencoderDetector(model)
    detector.scaler = Scaler(
        means=np.zeros(2, dtype=np.float32),
        stds=np.ones(2, dtype=np.float32),
        feature_names=("a", "b"),
    )
    path = tmp_path / "model.pt"
    detector.save(path)

    restored = AutoencoderDetector.load(path, detector.scaler)

    windows = np.random.default_rng(1).normal(size=(3, 20, 2)).astype(np.float32)
    device = torch.device("cpu")
    assert np.allclose(
        reconstruction_error(model, windows, device),
        reconstruction_error(restored.model, windows, device),
        atol=1e-6,
    )


def test_loading_against_a_different_feature_order_fails(tmp_path) -> None:
    detector = AutoencoderDetector(LSTMAutoencoder(n_features=2, hidden_size=8))
    detector.scaler = Scaler(
        means=np.zeros(2, dtype=np.float32),
        stds=np.ones(2, dtype=np.float32),
        feature_names=("a", "b"),
    )
    path = tmp_path / "model.pt"
    detector.save(path)
    swapped = Scaler(
        means=np.zeros(2, dtype=np.float32),
        stds=np.ones(2, dtype=np.float32),
        feature_names=("b", "a"),
    )

    with pytest.raises(ValueError, match="feature order"):
        AutoencoderDetector.load(path, swapped)


def test_saving_without_a_scaler_is_an_error(tmp_path) -> None:
    detector = AutoencoderDetector(LSTMAutoencoder(n_features=2, hidden_size=8))

    with pytest.raises(RuntimeError, match="scaler"):
        detector.save(tmp_path / "model.pt")


def test_clean_criterion_ignores_the_anomalous_windows() -> None:
    errors = np.array([1.0, 1.0, 100.0, 100.0])
    labels = np.array([0, 0, 1, 1])

    value, higher_is_better = selection_score("normal_val_loss", errors, labels)

    assert value == 1.0
    assert higher_is_better is False


def test_contaminated_criterion_is_moved_by_the_anomalous_windows() -> None:
    errors = np.array([1.0, 1.0, 100.0, 100.0])
    labels = np.array([0, 0, 1, 1])

    clean, _ = selection_score("normal_val_loss", errors, labels)
    contaminated, _ = selection_score("mixed_val_loss", errors, labels)

    assert contaminated > clean


def test_ranking_criterion_prefers_separable_errors() -> None:
    labels = np.array([0, 0, 1, 1])
    separable, higher_is_better = selection_score(
        "val_pr_auc", np.array([0.1, 0.2, 5.0, 6.0]), labels
    )
    muddled, _ = selection_score("val_pr_auc", np.array([5.0, 6.0, 0.1, 0.2]), labels)

    assert higher_is_better is True
    assert separable > muddled


def test_unknown_criterion_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown criterion"):
        selection_score("guesswork", np.array([1.0]), np.array([0]))


def test_training_produces_a_loadable_checkpoint(tmp_path) -> None:
    data_root = tmp_path / "data"
    build(
        BuildConfig(
            source="synthetic",
            strategy="held_out_series",
            n_scenarios=12,
            days=2,
            stride=40,
            output_root=data_root,
        )
    )
    config = TrainConfig(
        source="synthetic",
        strategy="held_out_series",
        max_epochs=2,
        max_train_windows=200,
        stride=40,
        hidden_size=8,
        data_root=data_root,
        models_root=tmp_path / "models",
    )

    summary = train(config)

    assert config.model_path.is_file()
    assert summary["best_epoch"] >= 1
    assert len(summary["history"]) >= 1
    scaler = Scaler.load(config.data_dir / "scaler.npz")
    detector = AutoencoderDetector.load(config.model_path, scaler)
    series = detector.scaler.feature_names
    assert len(series) == 4


def test_trained_detector_scores_every_window(tmp_path) -> None:
    data_root = tmp_path / "data"
    build(
        BuildConfig(
            source="synthetic",
            strategy="held_out_series",
            n_scenarios=12,
            days=2,
            stride=40,
            output_root=data_root,
        )
    )
    config = TrainConfig(
        source="synthetic",
        strategy="held_out_series",
        max_epochs=1,
        max_train_windows=200,
        stride=40,
        hidden_size=8,
        data_root=data_root,
        models_root=tmp_path / "models",
    )
    train(config)

    test_series = store.load_series(config.data_dir / "test")
    index = build_index(test_series, 60, 40)
    scaler = Scaler.load(config.data_dir / "scaler.npz")
    detector = AutoencoderDetector.load(config.model_path, scaler)

    scores = detector.score(test_series, index)

    assert scores.shape == (len(index),)
    assert np.isfinite(scores).all()
