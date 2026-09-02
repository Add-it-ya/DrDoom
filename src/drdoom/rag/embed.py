"""Text embedding, behind an interface so tests never download a model.

The real embedder is a sentence-transformers bi-encoder. It is imported lazily inside the
constructor: importing it at module scope would make the whole retrieval package depend on
a model download, and the tests that exercise chunking, ranking and fusion have no need of
one.

The hashing embedder is a deterministic fallback with no download and no learned
semantics. It exists so the pipeline can be exercised end to end offline, and it is
included in the results table as a floor -- if a learned encoder cannot beat hashed
character n-grams, it is not earning its place.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKEN = re.compile(r"[a-z0-9]+")


def normalise(vectors: np.ndarray) -> np.ndarray:
    """Scale rows to unit length so a dot product is a cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


class Embedder(Protocol):
    name: str
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return one unit-length row per text."""
        ...


class HashingEmbedder:
    """Hashed character n-grams. Deterministic, offline, and deliberately weak."""

    def __init__(self, dimension: int = 512, ngram: int = 4) -> None:
        self.name = f"hashing-{dimension}"
        self.dimension = dimension
        self.ngram = ngram

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        lowered = text.lower()
        for token in TOKEN.findall(lowered):
            padded = f" {token} "
            for start in range(max(1, len(padded) - self.ngram + 1)):
                gram = padded[start : start + self.ngram]
                bucket = int.from_bytes(
                    hashlib.blake2b(gram.encode(), digest_size=4).digest(), "little"
                )
                vector[bucket % self.dimension] += 1.0
        return vector

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return normalise(np.stack([self._vector(text) for text in texts]))


class SentenceTransformerEmbedder:
    """A learned bi-encoder. Downloads its weights on first use."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 64) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name.rsplit("/", maxsplit=1)[-1]
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vectors.astype(np.float32)
