"""What the dashboard's two demonstration buttons actually do.

The service is assembled from generated traffic so it runs without the real dataset, and
its threshold is measured on separately generated validation traffic rather than typed in.
These tests hold the assembled detector to what the buttons promise.
"""

import pytest

from drdoom.agents.triage import TriageAgent
from drdoom.api.factory import WINDOW, build_detector, demo_window
from drdoom.data import synthetic


@pytest.fixture(scope="module")
def triage() -> TriageAgent:
    detector, threshold, feature_names = build_detector()
    return TriageAgent(detector, threshold, feature_names)


def test_the_disturbed_demo_window_raises_an_incident(triage) -> None:
    assert triage.run(demo_window(anomalous=True)).is_anomaly is True


def test_the_calm_demo_window_does_not(triage) -> None:
    assert triage.run(demo_window(anomalous=False)).is_anomaly is False


def test_the_threshold_is_identical_on_every_start() -> None:
    assert build_detector()[1] == build_detector()[1]


@pytest.mark.parametrize("anomalous", [True, False])
def test_the_demo_windows_fit_the_service_exactly(anomalous: bool) -> None:
    """The service refuses any other shape, so the dashboard's buttons must produce this one."""
    assert demo_window(anomalous=anomalous).shape == (WINDOW, len(synthetic.FEATURE_NAMES))


@pytest.mark.parametrize("kind", ["conv", "conv+naive", "window_spread"])
def test_every_configurable_detector_separates_the_demo_windows(kind: str) -> None:
    detector, threshold, names = build_detector(kind)
    agent = TriageAgent(detector, threshold, names)

    assert agent.run(demo_window(anomalous=True)).is_anomaly is True
    assert agent.run(demo_window(anomalous=False)).is_anomaly is False


def test_the_default_is_the_conv_autoencoder() -> None:
    assert build_detector()[0].name == "conv_autoencoder"


def test_a_detector_that_fails_to_fit_falls_back_to_the_baseline(monkeypatch) -> None:
    from drdoom.api import factory

    def broken(*_args, **_kwargs):
        raise RuntimeError("no torch here")

    real = factory.fit_detector
    monkeypatch.setattr(
        factory,
        "fit_detector",
        lambda kind, series, scaler: (
            broken() if kind != "window_spread" else real(kind, series, scaler)
        ),
    )

    assert factory.build_detector("conv")[0].name == "window_spread"


def _models_at(monkeypatch, root) -> None:
    from types import SimpleNamespace

    from drdoom.api import factory

    monkeypatch.setattr(factory, "get_settings", lambda: SimpleNamespace(models_dir=root))


def test_a_trained_classifier_is_loaded(monkeypatch, tmp_path) -> None:
    from drdoom.api.factory import build_classifier
    from drdoom.classify.train import ClassifierConfig, train

    train(
        ClassifierConfig(
            source="synthetic",
            n_scenarios=24,
            days=2,
            n_trials=2,
            n_folds=3,
            stride=40,
            models_root=tmp_path,
        )
    )
    _models_at(monkeypatch, tmp_path)

    classifier = build_classifier()

    assert classifier is not None
    cause, confidence = classifier.predict(demo_window(), list(synthetic.FEATURE_NAMES))
    assert cause in classifier.labels
    assert 0.0 <= confidence <= 1.0


def test_a_missing_classifier_is_a_warning(monkeypatch, tmp_path, caplog) -> None:
    """Every incident goes unclassified without it, which should not pass as routine."""
    from drdoom.api.factory import build_classifier

    _models_at(monkeypatch, tmp_path)

    with caplog.at_level("INFO", logger="drdoom.api.factory"):
        assert build_classifier() is None

    [record] = [r for r in caplog.records if "unclassified" in r.getMessage()]
    assert record.levelname == "WARNING"


def test_start_up_names_the_whole_configuration_in_one_line(monkeypatch, tmp_path, caplog) -> None:
    """Which detector, threshold, retriever, reranker, classifier and models are serving."""
    from drdoom.api import factory
    from drdoom.audit import AuditLog
    from drdoom.detect.baselines import WindowSpread
    from drdoom.llm.stub import StubProvider
    from drdoom.rag.corpus import Document
    from drdoom.rag.index import BM25Index
    from drdoom.rag.ingest import chunk_all

    document = Document(
        doc_id="k8s:memory",
        source="kubernetes",
        path="memory.md",
        title="Assign Memory Resources",
        text="## Limits\n" + "Set a memory limit on the container. " * 10,
        url="https://example.invalid/memory",
        licence="CC-BY-4.0",
    )
    names = list(synthetic.FEATURE_NAMES)
    monkeypatch.setattr(factory, "build_detector", lambda: (WindowSpread(), 0.1234, names))
    monkeypatch.setattr(
        factory, "build_retriever", lambda use_dense=True: BM25Index(chunk_all([document]))
    )
    monkeypatch.setattr(factory, "build_classifier", lambda: None)
    monkeypatch.setattr(factory, "checkpoint_path", lambda: tmp_path / "state.sqlite")
    monkeypatch.setattr(factory, "AuditLog", lambda: AuditLog(tmp_path / "audit.jsonl"))

    with caplog.at_level("INFO", logger="drdoom.api.factory"):
        service = factory.build_service(provider=StubProvider(model="stub-1"))
    service.connection.close()

    [line] = [r.getMessage() for r in caplog.records if r.getMessage().startswith("service ready")]
    assert line == (
        "service ready: detector window_spread at threshold 0.1234, retriever BM25Index, "
        "reranker none, classifier none, model stub/stub-1, risk reviewed by stub/stub-1"
    )
