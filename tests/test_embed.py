"""Saving corpus embeddings, and trusting a saved matrix only while it still fits.

Embedding the corpus is the slow part of starting the service, so it is done once and
kept. The direction that matters is the other one: a matrix made from different text or a
different model must never be served, because nothing downstream could tell.
"""

import numpy as np
import pytest

from drdoom.rag.corpus import Document
from drdoom.rag.embed import HashingEmbedder, encode_cached
from drdoom.rag.index import DenseIndex
from drdoom.rag.ingest import chunk_all

TEXTS = ["restart the pods", "set a memory limit", "drain the node"]


class CountingEmbedder(HashingEmbedder):
    """The hashing embedder, counting how often it is actually asked to encode."""

    def __init__(self, dimension: int = 64) -> None:
        super().__init__(dimension=dimension)
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        return super().encode(texts)


def small_corpus() -> list:
    return chunk_all(
        [
            Document(
                doc_id="memory",
                source="test",
                path="memory.md",
                title="Assign Memory Resources",
                text="## Limits\n" + "Set a memory limit so the container is capped. " * 12,
                url="https://example.invalid/memory",
                licence="CC-BY-4.0",
            ),
            Document(
                doc_id="network",
                source="test",
                path="network.md",
                title="Declare Network Policy",
                text="## Policy\n" + "Network policy restricts traffic between pods. " * 12,
                url="https://example.invalid/network",
                licence="CC-BY-4.0",
            ),
        ]
    )


def test_a_saved_matrix_is_reused_when_nothing_changed(tmp_path) -> None:
    embedder = CountingEmbedder()
    path = tmp_path / "embeddings.npz"

    first = encode_cached(embedder, TEXTS, path)
    second = encode_cached(embedder, TEXTS, path)

    assert embedder.calls == 1
    assert np.array_equal(first, second)


def test_changed_text_is_recomputed_rather_than_trusted(tmp_path) -> None:
    embedder = CountingEmbedder()
    path = tmp_path / "embeddings.npz"
    encode_cached(embedder, TEXTS, path)
    changed = [*TEXTS[:2], "cordon the node"]

    matrix = encode_cached(embedder, changed, path)

    assert embedder.calls == 2
    assert np.array_equal(matrix, HashingEmbedder(dimension=64).encode(changed))


def test_a_different_model_is_recomputed_rather_than_trusted(tmp_path) -> None:
    path = tmp_path / "embeddings.npz"
    encode_cached(CountingEmbedder(dimension=64), TEXTS, path)
    other = CountingEmbedder(dimension=32)

    matrix = encode_cached(other, TEXTS, path)

    assert other.calls == 1
    assert matrix.shape == (len(TEXTS), 32)


def test_a_save_leaves_no_partial_file_behind(tmp_path) -> None:
    encode_cached(CountingEmbedder(), TEXTS, tmp_path / "embeddings.npz")

    assert [path.name for path in tmp_path.iterdir()] == ["embeddings.npz"]


def test_the_dense_index_ranks_a_saved_matrix_exactly_as_a_computed_one() -> None:
    chunks = small_corpus()
    embedder = HashingEmbedder()
    computed = DenseIndex(chunks, embedder)
    restored = DenseIndex(chunks, embedder, matrix=computed.matrix.copy())
    query = "network policy between pods"

    assert [hit.chunk.chunk_id for hit in restored.search(query, k=2)] == [
        hit.chunk.chunk_id for hit in computed.search(query, k=2)
    ]


def test_the_dense_index_refuses_a_matrix_that_does_not_fit() -> None:
    chunks = small_corpus()
    wrong = np.zeros((len(chunks) + 1, 512), dtype=np.float32)

    with pytest.raises(ValueError, match="shape"):
        DenseIndex(chunks, HashingEmbedder(), matrix=wrong)
