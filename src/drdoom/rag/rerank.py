"""Reranking a shortlist with a cross-encoder.

A bi-encoder embeds the query and the passage separately, so the two never see each other
and the score is a similarity between two summaries. A cross-encoder reads the pair
together and scores the match directly, which is more accurate and far too slow to run
over a whole corpus. The usual arrangement, and the one here, is to let the fast
retrievers propose a shortlist and let the slow model reorder it.

Whether that reordering is worth its latency is an empirical question, so the identity
reranker is kept as the control and both appear in the results table.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from drdoom.rag.index import Hit

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]: ...


class NoReranker:
    """Keeps the retriever's order. The control condition."""

    name = "none"

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        return hits[:k]


class CrossEncoderReranker:
    """Scores each query-passage pair jointly and reorders by that score."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32) -> None:
        from sentence_transformers import CrossEncoder

        self.name = model_name.rsplit("/", maxsplit=1)[-1]
        self.batch_size = batch_size
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if not hits:
            return []
        scores = self._model.predict(
            [(query, hit.chunk.search_text) for hit in hits],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        order = np.argsort(-np.asarray(scores, dtype=np.float32))[:k]
        return [
            Hit(chunk=hits[int(position)].chunk, score=float(scores[int(position)]), rank=rank)
            for rank, position in enumerate(order, start=1)
        ]


class LengthPenaltyReranker:
    """Deterministic reranker used to exercise the plumbing without a download.

    Nudges very short passages down, which is a real if unsophisticated signal: a
    forty-word fragment rarely answers an operational question on its own.
    """

    name = "length_penalty"

    def __init__(self, target_chars: int = 600) -> None:
        self.target_chars = target_chars

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        def adjusted(hit: Hit) -> float:
            ratio = min(len(hit.chunk.text) / self.target_chars, 1.0)
            return (1.0 / hit.rank) * (0.5 + 0.5 * ratio)

        ordered = sorted(hits, key=adjusted, reverse=True)[:k]
        return [
            Hit(chunk=hit.chunk, score=adjusted(hit), rank=rank)
            for rank, hit in enumerate(ordered, start=1)
        ]
