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
from drdoom.agents.triage import Classifier, TriageAgent, window_to_series
from drdoom.audit import AuditLog
from drdoom.config import get_settings
from drdoom.data import synthetic
from drdoom.data.windows import Scaler, build_index
from drdoom.detect.base import Detector
from drdoom.detect.baselines import WindowSpread
from drdoom.executor import DryRunExecutor
from drdoom.llm.base import LLMProvider
from drdoom.llm.factory import build_provider
from drdoom.rag import corpus
from drdoom.rag.index import BM25Index, DenseIndex, HybridRetriever, Retriever
from drdoom.rag.ingest import chunk_all

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 2.5
WINDOW = 60


def build_detector() -> tuple[Detector, float, list[str]]:
    """A baseline fitted on generated normal traffic.

    Measurement on the real dataset favoured a window statistic over the autoencoder, so
    the default here is that statistic rather than the more impressive option.
    """
    series = synthetic.generate(n_scenarios=6, days=2, seed=7)
    normal = build_index(series, WINDOW, stride=20).normal_only()
    detector = WindowSpread()
    detector.fit(series, normal, Scaler.fit(series))
    return detector, DEFAULT_THRESHOLD, list(synthetic.FEATURE_NAMES)


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


def build_retriever(use_dense: bool = False) -> Retriever:
    if not corpus.is_downloaded():
        raise RuntimeError(
            "the document corpus is missing; run: python -c "
            "'from drdoom.rag import corpus; corpus.download()'"
        )
    chunks = chunk_all(corpus.load())
    lexical = BM25Index(chunks)
    if not use_dense:
        return lexical

    from drdoom.rag.embed import SentenceTransformerEmbedder

    return HybridRetriever([lexical, DenseIndex(chunks, SentenceTransformerEmbedder())])


def build_service(provider: LLMProvider | None = None, use_dense: bool = False):
    """Wire the whole system together for a real run."""
    from drdoom.agents.graph import make_checkpointer
    from drdoom.api.main import Service

    settings = get_settings()
    detector, threshold, feature_names = build_detector()
    retriever = build_retriever(use_dense=use_dense)
    model = provider or build_provider(settings.llm_provider)
    audit = AuditLog()

    checkpointer, connection = make_checkpointer(checkpoint_path())

    investigator = Investigator(
        TriageAgent(detector, threshold, feature_names, classifier=build_classifier()),
        DiagnosisAgent(retriever, model),
        RemediationAgent(retriever, model),
        ReportingAgent(model),
        checkpointer,
        executor=DryRunExecutor(),
        audit=audit,
    )
    logger.info("service ready with provider %s", model.name)
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
