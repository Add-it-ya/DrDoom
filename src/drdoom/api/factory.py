"""Assemble the running system from whatever artefacts are on disk.

A demonstration that only works after five training commands is a demonstration nobody
runs. This falls back where it safely can and fails loudly where it cannot: the detector
degrades to a fitted baseline, and a missing document corpus is an error with the command
to fix it, because retrieval has no honest fallback.
"""

from __future__ import annotations

import logging

import numpy as np

from drdoom.agents.diagnosis import DiagnosisAgent
from drdoom.agents.graph import Investigator, checkpoint_path
from drdoom.agents.remediation import RemediationAgent
from drdoom.agents.reporting import ReportingAgent
from drdoom.agents.risk import RiskAssessor
from drdoom.agents.triage import Classifier, TriageAgent, window_to_series
from drdoom.audit import AuditLog
from drdoom.config import get_settings
from drdoom.data import synthetic
from drdoom.data.windows import Scaler, build_index
from drdoom.detect.base import Detector
from drdoom.detect.baselines import WindowSpread
from drdoom.detect.evaluate import select_threshold
from drdoom.executor import DryRunExecutor
from drdoom.llm.base import LLMProvider
from drdoom.llm.factory import build_provider_or_unavailable
from drdoom.rag import corpus
from drdoom.rag.index import BM25Index, DenseIndex, HybridRetriever, Retriever
from drdoom.rag.ingest import chunk_all

logger = logging.getLogger(__name__)

WINDOW = 60


def build_detector() -> tuple[Detector, float, list[str]]:
    """A baseline fitted on generated normal traffic, with a threshold that is measured.

    Measurement on the real dataset favoured a window statistic over the autoencoder, so
    the default here is that statistic rather than the more impressive option.

    The threshold is chosen the way every published result chooses one: on separately
    generated validation traffic, as the most sensitive value that stays inside the false
    alarm budget. A number typed in by hand is expressed in the units of one particular
    scaler, and quietly stops meaning anything when that scaler changes.
    """
    series = synthetic.generate(n_scenarios=6, days=2, seed=7)
    normal = build_index(series, WINDOW, stride=20).normal_only()
    detector = WindowSpread()
    detector.fit(series, normal, Scaler.fit(series))

    validation = synthetic.generate(n_scenarios=6, days=2, seed=8)
    index = build_index(validation, WINDOW)
    threshold = select_threshold(detector.score(validation, index), index, validation)
    logger.info("detector threshold %.4f chosen against the false alarm budget", threshold)
    return detector, threshold, list(synthetic.FEATURE_NAMES)


def build_classifier() -> Classifier | None:
    directory = get_settings().models_dir / "classifier" / "synthetic"
    if not (directory / "model.json").is_file():
        logger.info("no trained classifier at %s, incidents will be unclassified", directory)
        return None
    try:
        return Classifier.load(directory)
    except (ValueError, OSError):
        logger.exception("classifier could not be loaded, continuing without it")
        return None


def build_retriever(use_dense: bool = True) -> Retriever:
    """The retriever the service runs: BM25 and a learned encoder, fused.

    Fusion is the configuration the retrieval results measured as better than BM25 alone,
    and the one the evaluation suite scores, so it is also the one that serves requests.
    ``use_dense=False`` leaves the encoder out, for anywhere its weights cannot be loaded.

    Embedding the corpus takes minutes on a CPU, so the matrix is saved beside the corpus
    the first time and reused on every start after that.
    """
    if not corpus.is_downloaded():
        raise RuntimeError(
            "the document corpus is missing; run: python -c "
            "'from drdoom.rag import corpus; corpus.download()'"
        )
    chunks = chunk_all(corpus.load())
    lexical = BM25Index(chunks)
    if not use_dense:
        return lexical

    from drdoom.rag.embed import SentenceTransformerEmbedder, encode_cached

    embedder = SentenceTransformerEmbedder()
    matrix = encode_cached(
        embedder,
        [chunk.search_text for chunk in chunks],
        corpus.corpus_dir() / f"embeddings-{embedder.name}.npz",
    )
    return HybridRetriever([lexical, DenseIndex(chunks, embedder, matrix=matrix)])


def build_service(provider: LLMProvider | None = None, use_dense: bool = True):
    """Wire the whole system together for a real run."""
    from drdoom.agents.graph import make_checkpointer
    from drdoom.api.main import Service

    settings = get_settings()
    detector, threshold, feature_names = build_detector()
    retriever = build_retriever(use_dense=use_dense)
    model = provider or build_provider_or_unavailable(settings.llm_provider)
    reviewer = model
    if provider is None and (settings.risk_provider or settings.risk_model):
        reviewer = build_provider_or_unavailable(
            settings.risk_provider or settings.llm_provider, settings.risk_model
        )
    audit = AuditLog()

    checkpointer, connection = make_checkpointer(checkpoint_path())

    investigator = Investigator(
        TriageAgent(
            detector,
            threshold,
            feature_names,
            classifier=build_classifier(),
            window_size=WINDOW,
        ),
        DiagnosisAgent(retriever, model),
        RemediationAgent(retriever, model),
        ReportingAgent(model),
        checkpointer,
        executor=DryRunExecutor(),
        audit=audit,
        risk=RiskAssessor(reviewer),
    )
    logger.info(
        "service ready with provider %s, risk reviewed by %s/%s",
        model.name,
        reviewer.name,
        reviewer.model,
    )
    return Service(investigator=investigator, audit=audit, connection=connection)


def demo_window(anomalous: bool = True) -> np.ndarray:
    """A window shaped like the ones the detector was fitted on, for the dashboard."""
    scenario = synthetic.generate_scenario(99, days=1, seed=11)
    if not anomalous:
        quiet = np.flatnonzero(scenario.point_labels == 0)
        start = int(quiet[len(quiet) // 3])
        return scenario.values[start : start + WINDOW]
    event = scenario.events[0]
    start = max(0, event.start - WINDOW // 3)
    return scenario.values[start : start + WINDOW]


__all__ = ["build_detector", "build_retriever", "build_service", "demo_window", "window_to_series"]
