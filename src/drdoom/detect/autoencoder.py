"""LSTM autoencoder detector.

The model compresses a window to a single hidden vector and reconstructs it from that
vector alone. Trained only on anomaly-free windows, it reconstructs normal behaviour
closely and unfamiliar behaviour poorly, so reconstruction error becomes the score.

Nothing here decides what counts as an anomaly. As with every other detector, the
threshold is chosen in ``evaluate`` against a false alarm budget.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from drdoom.data.schema import MetricSeries
from drdoom.data.windows import Scaler, WindowIndex
from drdoom.detect.base import BATCH_SIZE, Detector


class LSTMAutoencoder(nn.Module):
    """Sequence to vector to sequence, with the bottleneck at the hidden state."""

    def __init__(self, n_features: int, hidden_size: int = 32, num_layers: int = 1) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, n_features)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(batch)
        summary = hidden[-1]
        repeated = summary.unsqueeze(1).expand(-1, batch.shape[1], -1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


def reconstruction_error(
    model: LSTMAutoencoder, windows: np.ndarray, device: torch.device
) -> np.ndarray:
    """Mean squared error per window, averaged over timesteps and features."""
    model.eval()
    with torch.no_grad():
        batch = torch.from_numpy(np.ascontiguousarray(windows)).to(device)
        error = ((batch - model(batch)) ** 2).mean(dim=(1, 2))
    return error.cpu().numpy()


class AutoencoderDetector(Detector):
    """Wraps a trained autoencoder so it scores windows like any other detector."""

    name = "lstm_autoencoder"

    def __init__(self, model: LSTMAutoencoder, device: torch.device | None = None) -> None:
        super().__init__()
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

    def score(self, series: list[MetricSeries], index: WindowIndex) -> np.ndarray:
        if not len(index):
            return np.empty(0, dtype=np.float32)
        return np.concatenate(
            [
                reconstruction_error(self.model, batch, self.device)
                for batch in self._scaled_batches(series, index)
            ]
        )

    def save(self, path: Path) -> None:
        if self.scaler is None:
            raise RuntimeError("cannot save a detector that has no scaler")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "n_features": self.model.n_features,
                "hidden_size": self.model.hidden_size,
                "feature_names": list(self.scaler.feature_names),
            },
            path,
        )

    @classmethod
    def load(
        cls, path: Path, scaler: Scaler, device: torch.device | None = None
    ) -> AutoencoderDetector:
        payload = torch.load(path, map_location=device or "cpu", weights_only=True)
        saved = tuple(payload["feature_names"])
        if saved != tuple(scaler.feature_names):
            raise ValueError(
                "checkpoint feature order does not match the scaler: "
                f"saved {saved[:3]}... vs scaler {tuple(scaler.feature_names)[:3]}..."
            )
        model = LSTMAutoencoder(payload["n_features"], payload["hidden_size"])
        model.load_state_dict(payload["state_dict"])
        detector = cls(model, device=device)
        detector.scaler = scaler
        return detector


def batch_count(n_rows: int, batch_size: int = BATCH_SIZE) -> int:
    return max(1, (n_rows + batch_size - 1) // batch_size)
