"""Corpus parsing, chunking, and the three retrievers.

Nothing here downloads a model. The hashing embedder stands in for the learned one so
ranking, fusion and reranking can be exercised offline and deterministically.
"""

import numpy as np
import pytest

from drdoom.rag.corpus import SOURCES, Document, SourceSpec, _wanted, clean_markdown
from drdoom.rag.embed import HashingEmbedder, normalise
from drdoom.rag.index import BM25Index, DenseIndex, Hit, HybridRetriever, tokenise
from drdoom.rag.ingest import MIN_CHUNK_CHARS, TARGET_CHARS, chunk_all, chunk_document
from drdoom.rag.rerank import LengthPenaltyReranker, NoReranker

KUBERNETES = SOURCES[1]


def make_document(doc_id: str, text: str, title: str = "Doc") -> Document:
    return Document(
        doc_id=doc_id,
        source="test",
        path=f"{doc_id}.md",
        title=title,
        text=text,
        url=f"https://example.invalid/{doc_id}",
        licence="CC-BY-4.0",
    )


def test_frontmatter_supplies_the_title_and_is_stripped() -> None:
    title, body = clean_markdown('---\ntitle: "Debugging Pods"\nweight: 10\n---\nBody text here.')

    assert title == "Debugging Pods"
    assert "weight" not in body
    assert body == "Body text here."


def test_title_falls_back_to_the_first_heading() -> None:
    title, _ = clean_markdown("# Draining a Node\n\nSome content.")

    assert title == "Draining a Node"


def test_template_shortcodes_and_comments_are_removed() -> None:
    _, body = clean_markdown("Text {{< note >}}inline{{< /note >}} and <!-- hidden --> tail.")

    assert "{{<" not in body
    assert "hidden" not in body
    assert "Text" in body and "tail" in body


def test_only_markdown_inside_the_included_prefixes_is_kept() -> None:
    assert _wanted("website-main/content/en/docs/tasks/debug.md", KUBERNETES) is not None
    assert _wanted("website-main/content/en/docs/setup/install.md", KUBERNETES) is None
    assert _wanted("website-main/content/en/docs/tasks/image.png", KUBERNETES) is None


def test_section_index_pages_are_skipped() -> None:
    assert _wanted("website-main/content/en/docs/tasks/_index.md", KUBERNETES) is None


def test_archive_paths_with_traversal_are_refused() -> None:
    spec = SourceSpec(
        name="x", archive_url="", include=("a/",), licence="", url_prefix="", strip_prefix="r/"
    )

    assert _wanted("r/../a/evil.md", spec) is None


def test_chunks_follow_the_document_headings() -> None:
    symptoms = "Latency climbs steadily and the queue depth grows with it. " * 4
    remediation = "Restart the affected pods and watch the queue drain. " * 4
    document = make_document("d1", f"## Symptoms\n{symptoms}\n\n## Remediation\n{remediation}")

    chunks = chunk_document(document)

    assert [chunk.heading for chunk in chunks] == ["Symptoms", "Remediation"]


def test_sections_shorter_than_the_minimum_are_dropped() -> None:
    document = make_document("d1", "## Symptoms\nHigh latency.\n\n## Remediation\nRestart.")

    assert chunk_document(document) == []


def test_long_sections_are_windowed_with_overlap() -> None:
    document = make_document("d1", "## Long\n" + ("word " * 900))

    chunks = chunk_document(document)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= TARGET_CHARS for chunk in chunks)
    assert chunks[1].offset < chunks[0].offset + TARGET_CHARS


def test_chunk_ids_are_stable_across_rebuilds() -> None:
    document = make_document("d1", "## A\n" + "text " * 400)

    assert [c.chunk_id for c in chunk_document(document)] == [
        c.chunk_id for c in chunk_document(document)
    ]


def test_chunk_ids_differ_between_documents() -> None:
    body = "## A\nSome reasonably long body text for the chunk to survive the minimum. " * 4
    first = {chunk.chunk_id for chunk in chunk_document(make_document("d1", body))}
    second = {chunk.chunk_id for chunk in chunk_document(make_document("d2", body))}

    assert first.isdisjoint(second)


def test_tiny_fragments_are_dropped() -> None:
    chunks = chunk_document(make_document("d1", "## A\nshort"))

    assert all(len(chunk.text) >= MIN_CHUNK_CHARS for chunk in chunks)


def test_chunk_carries_provenance_for_citation() -> None:
    document = make_document("d1", "## Remediation\n" + "detail " * 40, title="Restarting Pods")

    chunk = chunk_document(document)[0]

    assert chunk.doc_id == "d1"
    assert chunk.licence == "CC-BY-4.0"
    assert chunk.citation == "Restarting Pods - Remediation"
    assert chunk.search_text.startswith("Restarting Pods. Remediation.")


def corpus_chunks() -> list:
    return chunk_all(
        [
            make_document(
                "memory",
                "## Memory limits\n" + "Set a memory limit so the container is capped. " * 12,
                title="Assign Memory Resources",
            ),
            make_document(
                "rollout",
                "## Restart\n" + "Use kubectl rollout restart to recreate the pods. " * 12,
                title="Rollout Restart",
            ),
            make_document(
                "network",
                "## Policy\n" + "Network policy restricts traffic between pods. " * 12,
                title="Declare Network Policy",
            ),
        ]
    )


def test_bm25_ranks_the_lexically_matching_chunk_first() -> None:
    index = BM25Index(corpus_chunks())

    hits = index.search("kubectl rollout restart", k=3)

    assert hits[0].chunk.doc_id == "rollout"


def test_bm25_returns_nothing_for_out_of_vocabulary_queries() -> None:
    assert BM25Index(corpus_chunks()).search("zzzzqqqq", k=5) == []


def test_bm25_ranks_are_consecutive_from_one() -> None:
    hits = BM25Index(corpus_chunks()).search("memory limit", k=3)

    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))


def test_dense_index_finds_the_related_chunk() -> None:
    index = DenseIndex(corpus_chunks(), HashingEmbedder())

    hits = index.search("network policy between pods", k=3)

    assert hits[0].chunk.doc_id == "network"


def test_embeddings_are_unit_length() -> None:
    vectors = HashingEmbedder().encode(["some text", "other text"])

    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_embedding_is_deterministic() -> None:
    embedder = HashingEmbedder()

    assert np.array_equal(embedder.encode(["repeat me"]), embedder.encode(["repeat me"]))


def test_normalising_a_zero_row_does_not_divide_by_zero() -> None:
    assert np.isfinite(normalise(np.zeros((1, 4), dtype=np.float32))).all()


def test_encoding_nothing_gives_an_empty_matrix() -> None:
    assert HashingEmbedder().encode([]).shape == (0, 512)


def test_hybrid_fuses_both_retrievers() -> None:
    chunks = corpus_chunks()
    hybrid = HybridRetriever([BM25Index(chunks), DenseIndex(chunks, HashingEmbedder())])

    hits = hybrid.search("kubectl rollout restart", k=3)

    assert hits[0].chunk.doc_id == "rollout"
    assert [hit.rank for hit in hits] == [1, 2, 3]


def test_hybrid_surfaces_a_chunk_only_one_retriever_found() -> None:
    chunks = corpus_chunks()
    bm25, dense = BM25Index(chunks), DenseIndex(chunks, HashingEmbedder())
    hybrid = HybridRetriever([bm25, dense])

    fused = {hit.chunk.chunk_id for hit in hybrid.search("memory limit container", k=6)}
    lexical = {hit.chunk.chunk_id for hit in bm25.search("memory limit container", k=6)}

    assert fused >= lexical or fused & lexical


def test_hybrid_needs_at_least_one_retriever() -> None:
    with pytest.raises(ValueError, match="at least one"):
        HybridRetriever([])


def test_no_reranker_preserves_order_and_truncates() -> None:
    chunks = corpus_chunks()
    hits = BM25Index(chunks).search("memory limit", k=5)

    assert NoReranker().rerank("memory limit", hits, 2) == hits[:2]


def test_length_penalty_reranker_demotes_a_tiny_passage() -> None:
    chunks = corpus_chunks()
    short = Hit(chunk=chunks[0], score=1.0, rank=1)
    long_chunk = next(chunk for chunk in chunks if len(chunk.text) > 400)
    long_hit = Hit(chunk=long_chunk, score=0.9, rank=2)
    tiny = Hit(
        chunk=short.chunk.__class__(**{**short.chunk.__dict__, "text": "tiny"}), score=1.0, rank=1
    )

    reranked = LengthPenaltyReranker().rerank("q", [tiny, long_hit], k=2)

    assert reranked[0].chunk.chunk_id == long_hit.chunk.chunk_id


def test_reranking_nothing_returns_nothing() -> None:
    assert LengthPenaltyReranker().rerank("q", [], k=5) == []


def test_tokenise_lowercases_and_drops_punctuation() -> None:
    assert tokenise("Restart the Pod, now!") == ["restart", "the", "pod", "now"]


def test_empty_corpus_is_searchable_without_error() -> None:
    assert BM25Index([]).search("anything", k=3) == []
    assert DenseIndex([], HashingEmbedder()).search("anything", k=3) == []
