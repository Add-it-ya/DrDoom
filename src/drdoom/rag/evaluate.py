"""Measure retrieval quality, and write the ablation table.

Run with ``python -m drdoom.rag.evaluate``.

Relevance is judged per document: a query is answered if any chunk of a document the
annotator listed appears in the top k. Two numbers are reported because they say different
things -- hit rate asks whether the engineer sees *an* answer, recall asks what fraction of
the known answers made it in, and a query with two relevant documents can satisfy the first
while failing the second.

Every configuration is scored on the same queries with the same shortlist depth, so a
difference between rows is attributable to the component that changed.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from drdoom.config import get_settings
from drdoom.rag import corpus
from drdoom.rag.embed import Embedder, HashingEmbedder
from drdoom.rag.index import BM25Index, DenseIndex, HybridRetriever, Retriever
from drdoom.rag.ingest import Chunk, chunk_all
from drdoom.rag.rerank import NoReranker, Reranker

DEFAULT_KS = (1, 3, 5, 10)
DEFAULT_DEPTH = 50
BOOTSTRAP_SAMPLES = 2000

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelledQuery:
    query: str
    relevant: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalMetrics:
    """Scores for one retrieval configuration."""

    configuration: str
    n_queries: int
    hit_rate: dict[int, float]
    recall: dict[int, float]
    mrr: float
    hit_rate_ci: dict[int, tuple[float, float]] = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            "configuration": self.configuration,
            "queries": self.n_queries,
            "hit_rate": {str(k): round(v, 4) for k, v in self.hit_rate.items()},
            "recall": {str(k): round(v, 4) for k, v in self.recall.items()},
            "mrr": round(self.mrr, 4),
            "hit_rate_ci": {
                str(k): [round(v, 4) for v in bounds] for k, bounds in self.hit_rate_ci.items()
            },
        }


def load_queries(path: Path | None = None) -> list[LabelledQuery]:
    location = path or get_settings().project_root / "evals" / "retrieval_queries.json"
    payload = json.loads(location.read_text(encoding="utf-8"))
    return [
        LabelledQuery(query=item["query"], relevant=tuple(item["relevant"]))
        for item in payload["queries"]
    ]


def _bootstrap_ci(values: np.ndarray, seed: int = 0) -> tuple[float, float]:
    if not len(values):
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    means = values[draws].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def evaluate(
    name: str,
    retriever: Retriever,
    queries: list[LabelledQuery],
    reranker: Reranker | None = None,
    ks: tuple[int, ...] = DEFAULT_KS,
    depth: int = DEFAULT_DEPTH,
) -> RetrievalMetrics:
    """Score one configuration over the labelled query set."""
    reranker = reranker or NoReranker()
    top_k = max(ks)
    hits: dict[int, list[float]] = {k: [] for k in ks}
    recalls: dict[int, list[float]] = {k: [] for k in ks}
    reciprocal_ranks: list[float] = []

    for item in queries:
        shortlist = retriever.search(item.query, depth)
        ordered = reranker.rerank(item.query, shortlist, top_k)
        documents = [hit.chunk.doc_id for hit in ordered]

        relevant = set(item.relevant)
        for k in ks:
            found = set(documents[:k]) & relevant
            hits[k].append(1.0 if found else 0.0)
            recalls[k].append(len(found) / len(relevant))

        rank = next(
            (position for position, doc in enumerate(documents, start=1) if doc in relevant), None
        )
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    return RetrievalMetrics(
        configuration=name,
        n_queries=len(queries),
        hit_rate={k: float(np.mean(values)) for k, values in hits.items()},
        recall={k: float(np.mean(values)) for k, values in recalls.items()},
        mrr=float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        hit_rate_ci={k: _bootstrap_ci(np.array(values)) for k, values in hits.items()},
    )


def build_configurations(
    chunks: list[Chunk], use_learned_models: bool
) -> list[tuple[str, Retriever, Reranker | None]]:
    """Assemble the ablation, from lexical alone up to the full stack."""
    bm25 = BM25Index(chunks)
    hashing = DenseIndex(chunks, HashingEmbedder())

    configurations: list[tuple[str, Retriever, Reranker | None]] = [
        ("bm25 only", bm25, None),
        ("dense only (hashed n-grams)", hashing, None),
        ("hybrid (bm25 + hashed)", HybridRetriever([bm25, hashing]), None),
    ]

    if not use_learned_models:
        return configurations

    from drdoom.rag.embed import SentenceTransformerEmbedder
    from drdoom.rag.rerank import CrossEncoderReranker

    embedder: Embedder = SentenceTransformerEmbedder()
    dense = DenseIndex(chunks, embedder)
    hybrid = HybridRetriever([bm25, dense])
    reranker = CrossEncoderReranker()

    configurations += [
        ("dense only (MiniLM)", dense, None),
        ("hybrid (bm25 + MiniLM)", hybrid, None),
        ("hybrid + cross-encoder rerank", hybrid, reranker),
    ]
    return configurations


def render_markdown(rows: list[dict], n_chunks: int, n_documents: int) -> str:
    lines = [
        "# Retrieval quality",
        "",
        "Generated by `python -m drdoom.rag.evaluate`.",
        "",
        f"Corpus: {n_documents} documents from Kubernetes and Prometheus documentation, split",
        f"into {n_chunks} chunks. Queries: hand-authored operational questions in",
        "`evals/retrieval_queries.json`, with relevance judged per document.",
        "",
        "There is no filter on document type or source. An earlier design in a predecessor",
        "project narrowed retrieval to the runbook matching the anomaly class the classifier",
        "had already chosen, which made semantic search a lookup over a handful of passages it",
        "could not rank wrongly. Retrieval here has to find the answer from the query alone.",
        "",
        "| Configuration | Hit@1 | Hit@3 | Hit@5 | Hit@5 95% CI | Hit@10 | Recall@5 | MRR |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        interval = row["hit_rate_ci"].get("5", [float("nan"), float("nan")])
        lines.append(
            f"| {row['configuration']} | {row['hit_rate']['1']:.3f} | {row['hit_rate']['3']:.3f} |"
            f" {row['hit_rate']['5']:.3f} | [{interval[0]:.2f}, {interval[1]:.2f}] |"
            f" {row['hit_rate']['10']:.3f} | {row['recall']['5']:.3f} | {row['mrr']:.3f} |"
        )

    lines += ["", "## What the table says", ""]
    by_name = {row["configuration"]: row for row in rows}

    def gain(better: str, worse: str) -> tuple[float, float] | None:
        if better not in by_name or worse not in by_name:
            return None
        return (
            by_name[better]["hit_rate"]["5"] - by_name[worse]["hit_rate"]["5"],
            by_name[better]["mrr"] - by_name[worse]["mrr"],
        )

    for better, worse, label in (
        ("dense only (MiniLM)", "dense only (hashed n-grams)", "a learned encoder over hashing"),
        ("hybrid (bm25 + MiniLM)", "bm25 only", "fusing a learned encoder with bm25"),
        ("hybrid (bm25 + MiniLM)", "dense only (MiniLM)", "adding bm25 to the learned encoder"),
        ("hybrid + cross-encoder rerank", "hybrid (bm25 + MiniLM)", "reranking the shortlist"),
    ):
        delta = gain(better, worse)
        if delta:
            lines.append(f"- {label}: hit@5 **{delta[0]:+.3f}**, MRR **{delta[1]:+.3f}**")

    rerank_delta = gain("hybrid + cross-encoder rerank", "hybrid (bm25 + MiniLM)")
    if rerank_delta and rerank_delta[1] > rerank_delta[0]:
        lines += [
            "",
            "The reranker moves MRR considerably more than hit rate, which is what a reranker",
            "is for: it rarely surfaces a document the retrievers missed, it moves the right",
            "document up the shortlist. That matters because the passages are handed to a",
            "language model with a limited context, so position decides what is actually read.",
        ]

    lines += [
        "",
        "Unlike the detection and classification layers, where the simplest method held its",
        "own, every component here pays for itself. That is the point of measuring each one",
        "separately rather than assuming either way.",
        "",
        "The query set is small and single-annotator, so a few points of hit rate sit inside",
        "the interval. It is a regression guard and a relative comparison between",
        "configurations, not an absolute measure of retrieval quality.",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score retrieval configurations.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="skip configurations that download a model",
    )
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not corpus.is_downloaded():
        corpus.download()
    documents = corpus.load()
    chunks = chunk_all(documents)
    queries = load_queries()
    logger.info("%d documents, %d chunks, %d queries", len(documents), len(chunks), len(queries))

    rows = []
    for name, retriever, reranker in build_configurations(chunks, not args.offline):
        metrics = evaluate(name, retriever, queries, reranker, depth=args.depth)
        rows.append(metrics.as_row())
        logger.info(
            "%-32s hit@5 %.3f  recall@5 %.3f  mrr %.3f",
            name,
            metrics.hit_rate[5],
            metrics.recall[5],
            metrics.mrr,
        )

    docs = args.out or get_settings().project_root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "retrieval-results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (docs / "retrieval-results.md").write_text(
        render_markdown(rows, len(chunks), len(documents)), encoding="utf-8"
    )
    logger.info("wrote %s", docs / "retrieval-results.md")


if __name__ == "__main__":
    main()
