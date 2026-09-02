"""Lexical, dense, and hybrid retrieval over the chunked corpus.

Three retrievers with one interface, so the results table can attribute any gain to a
specific component rather than to the stack as a whole.

BM25 is implemented here rather than pulled in. The scoring function is forty lines of
well-understood arithmetic, and precomputing the term weights into a sparse matrix makes a
query a column sum, which is both faster and easier to test than a dependency would be.

Fusion is reciprocal rank, which combines rankings rather than scores. Lexical and dense
scores live on incompatible scales, so any weighted sum of them needs calibration that
would have to be tuned and would then be one more thing to justify.

The corpus is a few thousand chunks, and an exact dot product over that is well under a
millisecond, so the dense index is plain numpy. A dedicated vector service earns its keep
when the corpus outgrows memory or needs concurrent writers; adding one now would be the
same unjustified machinery this project set out to avoid.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy import sparse

from drdoom.rag.embed import Embedder
from drdoom.rag.ingest import Chunk

TOKEN = re.compile(r"[a-z0-9]+")
RRF_K = 60


def tokenise(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk, with the score and rank that produced it."""

    chunk: Chunk
    score: float
    rank: int


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 10) -> list[Hit]: ...


def _to_hits(chunks: list[Chunk], order: np.ndarray, scores: np.ndarray) -> list[Hit]:
    return [
        Hit(chunk=chunks[int(position)], score=float(scores[int(position)]), rank=rank)
        for rank, position in enumerate(order, start=1)
    ]


class BM25Index:
    """Okapi BM25 with the term weights precomputed into a sparse matrix."""

    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        documents = [tokenise(chunk.search_text) for chunk in chunks]
        lengths = np.array([len(document) for document in documents], dtype=np.float32)
        average_length = float(lengths.mean()) if len(lengths) else 1.0

        self.vocabulary: dict[str, int] = {}
        rows, columns, weights = [], [], []
        document_frequency: Counter[str] = Counter()
        for document in documents:
            document_frequency.update(set(document))

        n_documents = max(len(documents), 1)
        for row, document in enumerate(documents):
            if not document:
                continue
            normaliser = self.k1 * (1 - self.b + self.b * lengths[row] / max(average_length, 1e-9))
            for term, frequency in Counter(document).items():
                column = self.vocabulary.setdefault(term, len(self.vocabulary))
                frequency_count = document_frequency[term]
                idf = math.log(
                    1.0 + (n_documents - frequency_count + 0.5) / (frequency_count + 0.5)
                )
                rows.append(row)
                columns.append(column)
                weights.append(idf * frequency * (self.k1 + 1) / (frequency + normaliser))

        shape = (len(documents), max(len(self.vocabulary), 1))
        self.weights = sparse.csc_matrix(
            (np.array(weights, dtype=np.float32), (rows, columns)), shape=shape
        )

    def search(self, query: str, k: int = 10) -> list[Hit]:
        columns = [
            self.vocabulary[term] for term in set(tokenise(query)) if term in self.vocabulary
        ]
        if not columns or not self.chunks:
            return []
        scores = np.asarray(self.weights[:, columns].sum(axis=1)).ravel()
        order = np.argsort(-scores)[:k]
        order = order[scores[order] > 0]
        return _to_hits(self.chunks, order, scores)


class DenseIndex:
    """Exact cosine similarity over unit-length embeddings."""

    name = "dense"

    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self.name = f"dense[{embedder.name}]"
        self.matrix = embedder.encode([chunk.search_text for chunk in chunks])

    def search(self, query: str, k: int = 10) -> list[Hit]:
        if not self.chunks:
            return []
        vector = self.embedder.encode([query])[0]
        scores = self.matrix @ vector
        order = np.argsort(-scores)[:k]
        return _to_hits(self.chunks, order, scores)


class HybridRetriever:
    """Reciprocal rank fusion over several retrievers."""

    def __init__(self, retrievers: list[Retriever], rrf_k: int = RRF_K, depth: int = 50) -> None:
        if not retrievers:
            raise ValueError("hybrid retrieval needs at least one retriever")
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.depth = depth
        self.name = "hybrid[" + "+".join(r.name for r in retrievers) + "]"

    def search(self, query: str, k: int = 10) -> list[Hit]:
        fused: dict[str, float] = {}
        seen: dict[str, Chunk] = {}
        for retriever in self.retrievers:
            for hit in retriever.search(query, self.depth):
                fused[hit.chunk.chunk_id] = fused.get(hit.chunk.chunk_id, 0.0) + 1.0 / (
                    self.rrf_k + hit.rank
                )
                seen[hit.chunk.chunk_id] = hit.chunk

        ranked = sorted(fused.items(), key=lambda item: -item[1])[:k]
        return [
            Hit(chunk=seen[chunk_id], score=score, rank=rank)
            for rank, (chunk_id, score) in enumerate(ranked, start=1)
        ]
