"""Retrieval metrics, and the labelled query set they run against."""

import json

import pytest

from drdoom.rag.evaluate import (
    LabelledQuery,
    build_configurations,
    evaluate,
    load_queries,
    render_markdown,
)
from drdoom.rag.index import Hit
from drdoom.rag.ingest import Chunk


def chunk(doc_id: str, text: str = "body text") -> Chunk:
    return Chunk(
        chunk_id=f"c-{doc_id}",
        doc_id=doc_id,
        source="test",
        title=doc_id,
        heading="",
        text=text,
        url="",
        licence="",
        offset=0,
    )


class FixedRetriever:
    """Returns a preset ranking, so metric arithmetic can be checked exactly."""

    name = "fixed"

    def __init__(self, order: list[str]) -> None:
        self.order = order

    def search(self, query: str, k: int = 10) -> list[Hit]:
        return [
            Hit(chunk=chunk(doc_id), score=1.0 / rank, rank=rank)
            for rank, doc_id in enumerate(self.order[:k], start=1)
        ]


def test_a_hit_at_rank_one_scores_everywhere() -> None:
    queries = [LabelledQuery("q", ("gold",))]

    metrics = evaluate("x", FixedRetriever(["gold", "a", "b"]), queries)

    assert metrics.hit_rate[1] == 1.0
    assert metrics.hit_rate[5] == 1.0
    assert metrics.mrr == 1.0


def test_a_hit_at_rank_three_misses_at_one() -> None:
    queries = [LabelledQuery("q", ("gold",))]

    metrics = evaluate("x", FixedRetriever(["a", "b", "gold"]), queries)

    assert metrics.hit_rate[1] == 0.0
    assert metrics.hit_rate[3] == 1.0
    assert metrics.mrr == pytest.approx(1 / 3)


def test_a_document_never_retrieved_scores_zero() -> None:
    metrics = evaluate("x", FixedRetriever(["a", "b"]), [LabelledQuery("q", ("gold",))])

    assert metrics.hit_rate[10] == 0.0
    assert metrics.mrr == 0.0


def test_recall_counts_the_fraction_of_gold_documents_found() -> None:
    queries = [LabelledQuery("q", ("gold1", "gold2"))]

    metrics = evaluate("x", FixedRetriever(["gold1", "a", "b"]), queries, ks=(3,))

    assert metrics.hit_rate[3] == 1.0
    assert metrics.recall[3] == pytest.approx(0.5)


def test_recall_reaches_one_when_both_gold_documents_are_found() -> None:
    queries = [LabelledQuery("q", ("gold1", "gold2"))]

    metrics = evaluate("x", FixedRetriever(["gold1", "gold2"]), queries, ks=(3,))

    assert metrics.recall[3] == pytest.approx(1.0)


def test_metrics_average_over_the_query_set() -> None:
    class Alternating:
        name = "alternating"

        def search(self, query: str, k: int = 10) -> list[Hit]:
            found = "gold" if query == "found" else "other"
            return [Hit(chunk=chunk(found), score=1.0, rank=1)]

    queries = [LabelledQuery("found", ("gold",)), LabelledQuery("missing", ("gold",))]

    metrics = evaluate("x", Alternating(), queries)

    assert metrics.hit_rate[1] == pytest.approx(0.5)
    assert metrics.n_queries == 2


def test_reranking_changes_the_measured_order() -> None:
    class ReverseReranker:
        name = "reverse"

        def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
            reordered = list(reversed(hits))[:k]
            return [
                Hit(chunk=hit.chunk, score=hit.score, rank=rank)
                for rank, hit in enumerate(reordered, start=1)
            ]

    queries = [LabelledQuery("q", ("gold",))]
    retriever = FixedRetriever(["a", "b", "gold"])

    before = evaluate("before", retriever, queries)
    after = evaluate("after", retriever, queries, reranker=ReverseReranker())

    assert before.hit_rate[1] == 0.0
    assert after.hit_rate[1] == 1.0


def test_intervals_are_reported_for_every_cutoff() -> None:
    metrics = evaluate("x", FixedRetriever(["gold"]), [LabelledQuery("q", ("gold",))])

    assert set(metrics.hit_rate_ci) == set(metrics.hit_rate)
    for low, high in metrics.hit_rate_ci.values():
        assert low <= high


def test_offline_configurations_download_nothing() -> None:
    chunks = [chunk("a", "memory limit " * 30), chunk("b", "network policy " * 30)]

    configurations = build_configurations(chunks, use_learned_models=False)

    names = [name for name, _, _ in configurations]
    assert names == ["bm25 only", "dense only (hashed n-grams)", "hybrid (bm25 + hashed)"]


def test_report_names_the_gain_from_each_component() -> None:
    rows = [
        {
            "configuration": name,
            "queries": 40,
            "hit_rate": {"1": 0.4, "3": 0.6, "5": hit, "10": 0.9},
            "recall": {"5": hit - 0.05},
            "mrr": mrr,
            "hit_rate_ci": {"5": [hit - 0.1, hit + 0.1]},
        }
        for name, hit, mrr in (
            ("bm25 only", 0.725, 0.539),
            ("dense only (hashed n-grams)", 0.525, 0.376),
            ("dense only (MiniLM)", 0.825, 0.630),
            ("hybrid (bm25 + MiniLM)", 0.850, 0.617),
            ("hybrid + cross-encoder rerank", 0.875, 0.699),
        )
    ]

    text = render_markdown(rows, n_chunks=5671, n_documents=430)

    assert "a learned encoder over hashing: hit@5 **+0.300**" in text
    assert "reranking the shortlist" in text
    assert "moves MRR considerably more than hit rate" in text
    assert "430 documents" in text


def test_labelled_query_set_is_present_and_well_formed() -> None:
    queries = load_queries()

    assert len(queries) >= 30
    assert all(query.query.strip() for query in queries)
    assert all(query.relevant for query in queries)


def test_labelled_query_set_records_its_limitations() -> None:
    from drdoom.config import get_settings

    payload = json.loads(
        (get_settings().project_root / "evals" / "retrieval_queries.json").read_text(
            encoding="utf-8"
        )
    )

    assert "limitations" in payload
    assert "annotator" in payload["limitations"]
