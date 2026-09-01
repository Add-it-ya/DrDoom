"""Train the autoencoder, and choose a checkpoint without contaminating the choice.

The detector is semi-supervised: it never sees an anomaly during training, so that it
does not learn to reconstruct one. That argument only holds if checkpoint selection
respects it too. Selecting on the mean reconstruction error of a validation split that
contains anomalies rewards the model for reconstructing them well, which is exactly the
behaviour the design set out to avoid.

Three criteria are implemented so the difference can be measured rather than asserted:

``normal_val_loss``
    Mean reconstruction error over anomaly-free validation windows only. Correct, and
    available even when validation labels are withheld.
``val_pr_auc``
    Ranking quality of reconstruction error against validation labels. Uses labels, and
    selects for the thing the detector is actually for.
``mixed_val_loss``
    Mean reconstruction error over all validation windows, anomalies included. Kept
    because it is the intuitive default and the one worth showing the cost of.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from drdoom.config import get_settings
from drdoom.data import store
from drdoom.data.schema import MetricSeries
from drdoom.data.windows import Scaler, WindowIndex, build_index, materialise
from drdoom.detect.autoencoder import AutoencoderDetector, LSTMAutoencoder, reconstruction_error

logger = logging.getLogger(__name__)

Criterion = Literal["normal_val_loss", "val_pr_auc", "mixed_val_loss"]
CRITERIA: tuple[Criterion, ...] = ("normal_val_loss", "val_pr_auc", "mixed_val_loss")


@dataclass(frozen=True)
class TrainConfig:
    source: str = "synthetic"
    strategy: str = "held_out_series"
    criterion: Criterion = "normal_val_loss"
    hidden_size: int = 32
    batch_size: int = 256
    learning_rate: float = 1e-3
    max_epochs: int = 20
    patience: int = 4
    max_train_windows: int = 40000
    window_size: int = 60
    stride: int = 10
    seed: int = 42
    data_root: Path | None = None
    models_root: Path | None = None

    @property
    def data_dir(self) -> Path:
        root = self.data_root or get_settings().processed_data_dir
        return root / self.source / self.strategy

    @property
    def model_path(self) -> Path:
        root = self.models_root or get_settings().models_dir
        return root / self.source / self.strategy / f"{self.criterion}.pt"


@dataclass
class Split:
    series: list[MetricSeries]
    index: WindowIndex


def load_split(config: TrainConfig, name: str) -> Split:
    series = store.load_series(config.data_dir / name)
    return Split(series, build_index(series, config.window_size, config.stride))


def _subsample(index: WindowIndex, limit: int, seed: int) -> WindowIndex:
    """Cap the training set so an epoch stays a reasonable length on cpu."""
    if len(index) <= limit:
        return index
    rng = np.random.default_rng(seed)
    return index.subset(np.sort(rng.choice(len(index), size=limit, replace=False)))


def _shuffled_batches(
    series: list[MetricSeries], index: WindowIndex, scaler: Scaler, batch_size: int, seed: int
):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(index))
    for begin in range(0, len(order), batch_size):
        rows = np.sort(order[begin : begin + batch_size])
        yield scaler.transform(materialise(series, index.subset(rows)))


def _scored(
    model: LSTMAutoencoder,
    series: list[MetricSeries],
    index: WindowIndex,
    scaler: Scaler,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    errors = []
    for begin in range(0, len(index), batch_size):
        rows = np.arange(begin, min(begin + batch_size, len(index)))
        batch = scaler.transform(materialise(series, index.subset(rows)))
        errors.append(reconstruction_error(model, batch, device))
    return np.concatenate(errors) if errors else np.empty(0, dtype=np.float32)


def selection_score(
    criterion: Criterion, errors: np.ndarray, labels: np.ndarray
) -> tuple[float, bool]:
    """Return the criterion value and whether larger is better."""
    if criterion == "normal_val_loss":
        normal = errors[labels == 0]
        return float(normal.mean()) if len(normal) else float("inf"), False
    if criterion == "mixed_val_loss":
        return float(errors.mean()), False
    if criterion == "val_pr_auc":
        if not 0 < labels.mean() < 1:
            return 0.0, True
        return float(average_precision_score(labels, errors)), True
    raise ValueError(f"unknown criterion {criterion!r}")


def train(config: TrainConfig) -> dict:
    """Train one autoencoder and keep the checkpoint the criterion prefers."""
    torch.manual_seed(config.seed)
    device = torch.device("cpu")

    train_split = load_split(config, "train")
    val_split = load_split(config, "val")
    scaler = Scaler.load(config.data_dir / "scaler.npz")

    train_index = _subsample(train_split.index.normal_only(), config.max_train_windows, config.seed)
    logger.info(
        "training on %d normal windows, validating on %d", len(train_index), len(val_split.index)
    )

    n_features = train_split.series[0].n_features
    model = LSTMAutoencoder(n_features, hidden_size=config.hidden_size).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.MSELoss()

    best_value: float | None = None
    best_state: dict | None = None
    best_epoch = 0
    since_improvement = 0
    history: list[dict] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses = []
        for batch in _shuffled_batches(
            train_split.series, train_index, scaler, config.batch_size, config.seed + epoch
        ):
            tensor = torch.from_numpy(batch).to(device)
            optimiser.zero_grad()
            loss = loss_function(model(tensor), tensor)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())

        errors = _scored(model, val_split.series, val_split.index, scaler, device)
        value, higher_is_better = selection_score(config.criterion, errors, val_split.index.label)
        improved = (
            best_value is None
            or (higher_is_better and value > best_value)
            or (not higher_is_better and value < best_value)
        )

        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), config.criterion: value}
        )
        logger.info(
            "epoch %2d train_loss %.5f %s %.5f%s",
            epoch,
            float(np.mean(losses)),
            config.criterion,
            value,
            " (best)" if improved else "",
        )

        if improved:
            best_value, best_epoch = value, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= config.patience:
                logger.info("stopping early at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    detector = AutoencoderDetector(model, device=device)
    detector.scaler = scaler
    detector.save(config.model_path)

    summary = {
        "source": config.source,
        "strategy": config.strategy,
        "criterion": config.criterion,
        "best_epoch": best_epoch,
        "best_value": best_value,
        "train_windows": len(train_index),
        "history": history,
        "model_path": str(config.model_path),
    }
    config.model_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    logger.info("saved %s (best epoch %d)", config.model_path, best_epoch)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the autoencoder detector.")
    parser.add_argument("--source", default="synthetic", choices=["smd", "synthetic"])
    parser.add_argument(
        "--strategy", default="held_out_series", choices=["time_based", "held_out_series"]
    )
    parser.add_argument("--criterion", default="normal_val_loss", choices=[*CRITERIA, "all"])
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--max-train-windows", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    base = TrainConfig(
        source=args.source,
        strategy=args.strategy,
        hidden_size=args.hidden_size,
        max_epochs=args.max_epochs,
        max_train_windows=args.max_train_windows,
        seed=args.seed,
    )
    criteria = CRITERIA if args.criterion == "all" else (args.criterion,)
    for criterion in criteria:
        train(replace(base, criterion=criterion))


if __name__ == "__main__":
    main()
