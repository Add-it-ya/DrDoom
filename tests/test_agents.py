"""The four agents, and the one rule the model is not allowed to decide.

Every test runs against the stub provider: no key, no network, no cost.
"""

import json

import numpy as np
import pytest

from drdoom.agents.diagnosis import DiagnosisAgent, build_query, format_passages, to_citations
from drdoom.agents.remediation import RemediationAgent
from drdoom.agents.reporting import ReportingAgent
from drdoom.agents.schemas import Diagnosis, Postmortem, RemediationPlan
from drdoom.agents.triage import TriageAgent, TriageResult, window_to_series
from drdoom.data.windows import Scaler
from drdoom.detect.baselines import WindowSpread
from drdoom.llm.base import LLMInvalidOutputError, LLMUnavailableError
from drdoom.llm.stub import SequenceProvider, StubProvider
from drdoom.rag.corpus import Document
from drdoom.rag.index import BM25Index
from drdoom.rag.ingest import chunk_all

DIAGNOSIS_JSON = json.dumps(
    {
        "summary": "Memory climbed steadily until the container was killed.",
        "likely_cause": "memory leak",
        "confidence": "high",
        "next_action": "Restart the pods and set a memory limit.",
    }
)

PLAN_JSON = json.dumps(
    {
        "immediate_action": "Rolling restart of the affected pods",
        "risk_level": "high",
        "short_term_fix": "Set a memory limit",
        "long_term_fix": "Fix the cache eviction policy",
        "rollback": "Scale the previous replica set back up",
    }
)

POSTMORTEM_JSON = json.dumps(
    {
        "title": "Memory leak in the api service",
        "summary": "A leak exhausted container memory.",
        "what_happened": "Memory grew for forty minutes.",
        "root_cause": "Unbounded cache.",
        "action_taken": "Rolling restart after approval.",
        "prevention": "Add an eviction policy.",
    }
)


def corpus_retriever() -> BM25Index:
    documents = [
        Document(
            doc_id="k8s:memory",
            source="kubernetes",
            path="memory.md",
            title="Assign Memory Resources",
            text="## Limits\n"
            + "Set a memory limit so a container cannot grow without bound. " * 10,
            url="https://example.invalid/memory",
            licence="CC-BY-4.0",
        ),
        Document(
            doc_id="k8s:restart",
            source="kubernetes",
            path="restart.md",
            title="Rollout Restart",
            text="## Restart\n"
            + "Use a rollout restart to recreate the pods of a deployment. " * 10,
            url="https://example.invalid/restart",
            licence="CC-BY-4.0",
        ),
    ]
    return BM25Index(chunk_all(documents))


# --- the safety rule ---------------------------------------------------------------


@pytest.mark.parametrize(("risk", "expected"), [("low", False), ("medium", True), ("high", True)])
def test_approval_follows_from_risk_level(risk: str, expected: bool) -> None:
    plan = RemediationPlan(
        immediate_action="a", risk_level=risk, short_term_fix="b", long_term_fix="c", rollback="d"
    )

    assert plan.requires_approval is expected


def test_the_model_is_never_asked_whether_approval_is_needed() -> None:
    schema = RemediationPlan.model_json_schema()

    assert "requires_approval" not in schema["properties"]


def test_a_model_claiming_a_dangerous_action_is_safe_is_overruled() -> None:
    """The exact failure this rule exists for: high risk, approval declared unnecessary."""
    plan = RemediationPlan.model_validate(
        {
            "immediate_action": "Delete the persistent volume",
            "risk_level": "high",
            "short_term_fix": "b",
            "long_term_fix": "c",
            "rollback": "d",
            "requires_approval": False,
        }
    )

    assert plan.requires_approval is True


def test_the_derived_field_survives_serialisation() -> None:
    plan = RemediationPlan(
        immediate_action="a", risk_level="high", short_term_fix="b", long_term_fix="c", rollback="d"
    )

    restored = RemediationPlan.model_validate(plan.model_dump())

    assert restored.requires_approval is True


def test_an_invented_risk_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="risk_level"):
        RemediationPlan(
            immediate_action="a",
            risk_level="catastrophic",
            short_term_fix="b",
            long_term_fix="c",
            rollback="d",
        )


def test_a_plan_cannot_be_mutated_after_construction() -> None:
    plan = RemediationPlan(
        immediate_action="a", risk_level="low", short_term_fix="b", long_term_fix="c", rollback="d"
    )

    with pytest.raises(ValueError, match="frozen"):
        plan.risk_level = "high"


# --- triage ------------------------------------------------------------------------


def quiet_window(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(loc=50.0, scale=1.0, size=(60, 2)).astype(np.float32)


def fitted_detector(train: np.ndarray) -> WindowSpread:
    series, index = window_to_series(train, ["a", "b"])
    detector = WindowSpread()
    detector.fit(series, index, Scaler.fit(series))
    return detector


def test_a_calm_window_is_not_an_incident() -> None:
    window = quiet_window()
    agent = TriageAgent(fitted_detector(window), threshold=10.0, feature_names=["a", "b"])

    result = agent.run(quiet_window(seed=1))

    assert result.is_anomaly is False
    assert result.root_cause is None


def test_a_disturbed_window_is_an_incident() -> None:
    window = quiet_window()
    agent = TriageAgent(fitted_detector(window), threshold=0.5, feature_names=["a", "b"])
    disturbed = quiet_window(seed=2)
    disturbed[30:, 0] += 60.0

    assert agent.run(disturbed).is_anomaly is True


def test_the_classifier_is_only_consulted_for_incidents() -> None:
    class ExplodingClassifier:
        def predict(self, window, feature_names):
            raise AssertionError("must not be called for a calm window")

    window = quiet_window()
    agent = TriageAgent(
        fitted_detector(window),
        threshold=10.0,
        feature_names=["a", "b"],
        classifier=ExplodingClassifier(),
    )

    assert agent.run(quiet_window(seed=3)).is_anomaly is False


def test_a_failing_classifier_still_reports_the_incident() -> None:
    class BrokenClassifier:
        def predict(self, window, feature_names):
            raise ValueError("feature order does not match")

    window = quiet_window()
    agent = TriageAgent(
        fitted_detector(window),
        threshold=0.0,
        feature_names=["a", "b"],
        classifier=BrokenClassifier(),
    )

    result = agent.run(quiet_window(seed=4))

    assert result.is_anomaly is True
    assert result.root_cause is None


def test_triage_result_serialises() -> None:
    row = TriageResult(is_anomaly=True, score=1.5, threshold=1.0, root_cause="x", confidence=0.9)

    assert row.as_dict()["root_cause"] == "x"


def test_a_window_of_the_wrong_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="timesteps"):
        window_to_series(np.zeros(60, dtype=np.float32), ["a"])


# --- diagnosis ---------------------------------------------------------------------


def test_diagnosis_grounds_itself_in_retrieved_passages() -> None:
    agent = DiagnosisAgent(corpus_retriever(), StubProvider(default=DIAGNOSIS_JSON))

    result = agent.run("memory limit exceeded on the container", "memory_leak")

    assert result.degraded is False
    assert result.diagnosis.likely_cause == "memory leak"
    assert result.citations
    assert all(citation.url for citation in result.citations)


def test_the_prompt_carries_the_retrieved_text() -> None:
    provider = StubProvider(default=DIAGNOSIS_JSON)
    DiagnosisAgent(corpus_retriever(), provider).run("memory limit container", "memory_leak")

    assert "Documentation excerpts" in provider.calls[0][0].content


def test_diagnosis_degrades_instead_of_failing_when_the_provider_is_down() -> None:
    agent = DiagnosisAgent(
        corpus_retriever(), StubProvider(fail_with=LLMUnavailableError("unreachable"))
    )

    result = agent.run("memory limit container", "memory_leak")

    assert result.degraded is True
    assert result.citations
    assert result.diagnosis.confidence == "low"
    assert "no model" in result.diagnosis.summary.lower()


def test_the_degraded_diagnosis_does_not_pretend_to_diagnose() -> None:
    agent = DiagnosisAgent(
        corpus_retriever(), StubProvider(fail_with=LLMUnavailableError("unreachable"))
    )

    summary = agent.run("memory limit container", "memory_leak").diagnosis.summary

    assert "has not been summarised" in summary


def test_a_persistently_malformed_diagnosis_raises() -> None:
    agent = DiagnosisAgent(corpus_retriever(), StubProvider(default="I would rather not"))

    with pytest.raises(LLMInvalidOutputError):
        agent.run("memory limit container", "memory_leak")


def test_query_combines_symptoms_and_predicted_cause() -> None:
    assert build_query("memory_leak", "latency rising") == "latency rising memory leak"
    assert build_query(None, "latency rising") == "latency rising"


def test_passages_are_numbered_for_the_prompt() -> None:
    hits = corpus_retriever().search("memory limit", k=2)

    assert format_passages(hits).startswith("[1] ")


def test_citations_carry_provenance() -> None:
    citations = to_citations(corpus_retriever().search("memory limit", k=1))

    assert citations[0].licence == "CC-BY-4.0"
    assert citations[0].doc_id.startswith("k8s:")


# --- remediation -------------------------------------------------------------------


def test_remediation_produces_a_risk_rated_plan() -> None:
    agent = RemediationAgent(corpus_retriever(), StubProvider(default=PLAN_JSON))

    result = agent.run("Memory grew until the container was killed.", "memory_leak")

    assert result.plan.risk_level == "high"
    assert result.plan.requires_approval is True
    assert result.citations


def test_remediation_repairs_a_malformed_plan() -> None:
    provider = SequenceProvider(['{"immediate_action": "restart"}', PLAN_JSON])
    agent = RemediationAgent(corpus_retriever(), provider)

    result = agent.run("summary", "memory_leak")

    assert result.plan.immediate_action.startswith("Rolling restart")
    assert len(result.completions) == 2


def test_remediation_retrieves_against_fixing_language() -> None:
    provider = StubProvider(default=PLAN_JSON)
    RemediationAgent(corpus_retriever(), provider).run("summary", "memory_leak")

    assert "remediation plan" in provider.calls[0][0].content.lower()


def test_token_usage_is_reported() -> None:
    agent = RemediationAgent(corpus_retriever(), StubProvider(default=PLAN_JSON))

    assert agent.run("summary", "memory_leak").tokens > 0


# --- reporting ---------------------------------------------------------------------


def diagnosis_fixture() -> Diagnosis:
    return Diagnosis.model_validate_json(DIAGNOSIS_JSON)


def plan_fixture() -> RemediationPlan:
    return RemediationPlan.model_validate_json(PLAN_JSON)


def test_report_renders_markdown_with_sources() -> None:
    citations = to_citations(corpus_retriever().search("memory limit", k=2))
    agent = ReportingAgent(StubProvider(default=POSTMORTEM_JSON))

    result = agent.run(
        diagnosis_fixture(), plan_fixture(), "approved_by_human", "restart", citations
    )

    assert result.markdown.startswith("# Memory leak in the api service")
    assert "## Sources" in result.markdown
    assert "https://example.invalid" in result.markdown


def test_the_report_prompt_states_the_decision_and_what_ran() -> None:
    provider = StubProvider(default=POSTMORTEM_JSON)
    ReportingAgent(provider).run(diagnosis_fixture(), plan_fixture(), "rejected_by_human", None)

    prompt = provider.calls[0][0].content
    assert "rejected_by_human" in prompt
    assert "nothing was executed" in prompt


def test_reporting_degrades_to_the_recorded_facts() -> None:
    agent = ReportingAgent(StubProvider(fail_with=LLMUnavailableError("down")))

    result = agent.run(diagnosis_fixture(), plan_fixture(), "rejected_by_human", None)

    assert result.degraded is True
    assert "Decision: rejected_by_human" in result.postmortem.action_taken
    assert "Executed: nothing" in result.postmortem.action_taken


def test_postmortem_markdown_has_every_section() -> None:
    markdown = Postmortem.model_validate_json(POSTMORTEM_JSON).to_markdown()

    for heading in ("Summary", "What happened", "Root cause", "Action taken", "Prevention"):
        assert f"## {heading}" in markdown
