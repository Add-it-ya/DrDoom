"""A convolutional autoencoder over the shape of a window, not its level.

The window statistic this project served until now, ``window_spread``, is the per-metric
spread of a window about its own mean. Read differently, it is the reconstruction error of
a model that predicts every window to be flat. This detector keeps that framing and learns
the model: it is given the window with each metric's mean removed, reconstructs it, and
scores the error. A network that learned nothing and predicted zeros would score exactly
the within-window spread, so anything it does learn can only explain away normal movement.

Removing the mean is the design choice that mattered. The real dataset drifts between its
training period and its labelled one, and an autoencoder over absolute levels (the LSTM
this project trained first) raises its error everywhere when the level moves. A centred
window is blind to level by construction: adding a constant to a metric leaves the score
unchanged, which a test holds it to.

Convolutions rather than a recurrent network because, in the experiments behind this
change, centring only helped a convolutional or linear model; a centred LSTM stayed level
with the uncentred ones. The model is small, stable across seeds, and scores in well under
a millisecond per window on a cpu.

The checkpoint is chosen by reconstruction error on anomaly-free windows the model did not
train on, held out from the training set itself, so no label ever shapes it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import numpy as np
import torch
from torch import nn

from drdoom.data.schema import MetricSeries
from drdoom.data.windows import Scaler, WindowIndex, materialise
from drdoom.detect.base import Detector

DEFAULT_EPOCHS = 12
HOLDOUT_FRACTION = 0.1


def centre(windows: torch.Tensor) -> torch.Tensor:
    """Remove each metric's mean over the window: ``(batch, time, features)``."""
    return windows - windows.mean(dim=1, keepdim=True)


class CentredConvAutoencoder(nn.Module):
    """Two strided convolutions down to a quarter of the window, and back up.

    The window length has to be a multiple of four so the decoder returns to it exactly.
    """

    def __init__(self, n_features: int, width: int = 64, latent_channels: int = 16) -> None:
        super().__init__()
        self.n_features = n_features
        self.width = width
        self.latent_channels = latent_channels
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, width, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(width, width, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(width, latent_channels, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, width, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(width, width, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(width, n_features, 3, padding=1),
        )

    def forward(self, centred: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(centred.transpose(1, 2))).transpose(1, 2)


def window_error(model: CentredConvAutoencoder, windows: np.ndarray) -> np.ndarray:
    """RMS reconstruction error over every cell of each centred window."""
    model.eval()
    with torch.no_grad():
        batch = centre(torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32)))
        return ((batch - model(batch)) ** 2).mean(dim=(1, 2)).sqrt().numpy()


class ConvAutoencoderDetector(Detector):
    """Scores a window by how badly its shape is reconstructed."""

    name = "conv_autoencoder"

    def __init__(
        self,
        model: CentredConvAutoencoder | None = None,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        max_train_windows: int = 60000,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_train_windows = max_train_windows
        self.seed = seed
        self.history: list[dict] = []

    # --- training ------------------------------------------------------------------

    def fit(self, series: list[MetricSeries], index: WindowIndex, scaler: Scaler) -> Self:
        """Train on anomaly-free windows, keeping the epoch that best fits unseen ones.

        ``index`` should hold normal windows only, as for every other detector. A tenth
        of them is set aside and never trained on; the epoch with the lowest error there
        is the one kept.
        """
        super().fit(series, index, scaler)
        if index.window_size % 4:
            raise ValueError("the window length must be a multiple of four")
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        order = rng.permutation(len(index))[: self.max_train_windows]
        holdout_size = max(1, int(len(order) * HOLDOUT_FRACTION))
        holdout = index.subset(np.sort(order[:holdout_size]))
        training = index.subset(np.sort(order[holdout_size:]))
        if not len(training):
            raise ValueError("too few windows to train on")

        self.model = CentredConvAutoencoder(series[0].n_features)
        optimiser = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=self.epochs)

        best_error, best_state = np.inf, None
        self.history = []
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            losses = []
            for rows in _batches(len(training), self.batch_size, rng):
                batch = torch.from_numpy(
                    scaler.transform(materialise(series, training.subset(rows)))
                )
                centred = centre(batch)
                loss = ((centred - self.model(centred)) ** 2).mean()
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimiser.step()
                losses.append(loss.item())
            schedule.step()

            held_out = float((self._errors(series, holdout) ** 2).mean())
            self.history.append(
                {"epoch": epoch, "train_loss": float(np.mean(losses)), "holdout": held_out}
            )
            if held_out < best_error:
                best_error = held_out
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        self.model.load_state_dict(best_state)
        return self

    # --- scoring -------------------------------------------------------------------

    def score(self, series: list[MetricSeries], index: WindowIndex) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{self.name} must be fitted before scoring")
        if not len(index):
            return np.empty(0, dtype=np.float32)
        return self._errors(series, index)

    def _errors(self, series: list[MetricSeries], index: WindowIndex) -> np.ndarray:
        return np.concatenate(
            [window_error(self.model, batch) for batch in self._scaled_batches(series, index)]
        ).astype(np.float32)

    # --- persistence ---------------------------------------------------------------

    def save(self, path: Path) -> None:
        if self.model is None or self.scaler is None:
            raise RuntimeError("cannot save a detector that has not been fitted")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "n_features": self.model.n_features,
                "width": self.model.width,
                "latent_channels": self.model.latent_channels,
                "feature_names": list(self.scaler.feature_names),
                "seed": self.seed,
                "epochs": self.epochs,
                "history": self.history,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, scaler: Scaler) -> ConvAutoencoderDetector:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        saved = tuple(payload["feature_names"])
        if saved != tuple(scaler.feature_names):
            raise ValueError(
                "checkpoint feature order does not match the scaler: "
                f"saved {saved[:3]}... vs scaler {tuple(scaler.feature_names)[:3]}..."
            )
        model = CentredConvAutoencoder(
            payload["n_features"], payload["width"], payload["latent_channels"]
        )
        model.load_state_dict(payload["state_dict"])
        detector = cls(model, epochs=payload["epochs"], seed=payload["seed"])
        detector.scaler = scaler
        detector.history = list(payload.get("history", []))
        return detector


def _batches(n_rows: int, batch_size: int, rng: np.random.Generator):
    order = rng.permutation(n_rows)
    for begin in range(0, n_rows, batch_size):
        yield np.sort(order[begin : begin + batch_size])
