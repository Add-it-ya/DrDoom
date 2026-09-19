"""The http surface: authentication, idempotency, streaming, and what reaches the browser.

The pipeline itself is tested elsewhere. What is under test here is the layer around it.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from drdoom.agents.diagnosis import DiagnosisAgent
from drdoom.agents.graph import Investigator, make_checkpointer
from drdoom.agents.remediation import RemediationAgent
from drdoom.agents.reporting import ReportingAgent
from drdoom.agents.risk import RiskAssessor
from drdoom.agents.triage import TriageAgent, window_to_series
from drdoom.api.auth import KeyRing
from drdoom.api.main import Service, create_app, set_service
from drdoom.audit import AuditLog
from drdoom.data.windows import Scaler
from drdoom.detect.baselines import WindowSpread
from drdoom.llm.stub import StubProvider
from drdoom.rag.corpus import Document
from drdoom.rag.index import BM25Index
from drdoom.rag.ingest import chunk_all
from tests._restart_worker import DIAGNOSIS, PLAN, POSTMORTEM, RISK_LOW, disturbed_window

KEY = "test-key-value"
PRINCIPAL = "aditya"

# The payload a poisoned document could talk a model into producing.
HOSTILE = (
    "Memory grew steadily. <img src=x onerror=\"fetch('/incidents/x/approve',"
    "{method:'POST'})\"> <script>alert(1)</script>"
)

HOSTILE_DIAGNOSIS = json.dumps(
    {
        "summary": HOSTILE,
        "likely_cause": "memory leak",
        "confidence": "high",
        "next_action": "Restart the pods.",
    }
)
HOSTILE_POSTMORTEM = json.dumps(
    {
        "title": "Incident",
        "summary": HOSTILE,
        "what_happened": HOSTILE,
        "root_cause": "Unbounded cache.",
        "action_taken": "Restart.",
        "prevention": "Add eviction.",
    }
)


def build_service(tmp_path: Path, diagnosis=DIAGNOSIS, postmortem=POSTMORTEM) -> Service:
    quiet = np.random.default_rng(0).normal(50, 1, size=(60, 2)).astype(np.float32)
    series, index = window_to_series(quiet, ["a", "b"])
    detector = WindowSpread()
    detector.fit(series, index, Scaler.fit(series))

    documents = [
        Document(
            doc_id="k8s:memory",
            source="kubernetes",
            path="memory.md",
            title="Assign Memory Resources",
            text="## Limits\n" + "Set a memory limit on the container. " * 10,
            url="https://example.invalid/memory",
            licence="CC-BY-4.0",
        )
    ]
    retriever = BM25Index(chunk_all(documents))
    audit = AuditLog(tmp_path / "audit.jsonl")

    checkpointer, connection = make_checkpointer(tmp_path / "state.sqlite")
    investigator = Investigator(
        TriageAgent(detector, threshold=5.0, feature_names=["a", "b"], window_size=60),
        DiagnosisAgent(retriever, StubProvider(default=diagnosis)),
        RemediationAgent(retriever, StubProvider(default=PLAN)),
        ReportingAgent(StubProvider(default=postmortem)),
        checkpointer,
        audit=audit,
        risk=RiskAssessor(StubProvider(default=RISK_LOW)),
    )
    return Service(investigator=investigator, audit=audit, connection=connection)


@pytest.fixture
def client(tmp_path):
    service = build_service(tmp_path)
    app = create_app(service=service, keyring=KeyRing({KEY: PRINCIPAL}))
    with TestClient(app) as test_client:
        yield test_client
    set_service(None)


def window_payload(anomalous: bool = True) -> dict:
    window = (
        disturbed_window()
        if anomalous
        else np.random.default_rng(2).normal(50, 1, size=(60, 2)).astype(np.float32)
    )
    return {
        "values": window.tolist(),
        "feature_names": ["a", "b"],
        "symptoms": "latency climbing",
    }


# --- basics ------------------------------------------------------------------------


def test_health_needs_no_credential(client) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_reports_each_part_it_rests_on(client) -> None:
    components = client.get("/health").json()["components"]

    assert {"model", "approvals", "store", "detector", "classifier", "retriever"} <= set(components)
    assert components["model"]["ready"] is True
    assert components["approvals"]["ready"] is True
    assert components["store"]["ready"] is True


def test_a_calm_window_returns_without_an_incident(client) -> None:
    body = client.post("/investigate", json=window_payload(anomalous=False)).json()

    assert body["is_anomaly"] is False
    assert body["status"] == "no_incident"
    assert body["plan"] is None


def test_an_incident_stops_at_the_gate(client) -> None:
    body = client.post("/investigate", json=window_payload()).json()

    assert body["status"] == "awaiting_approval"
    assert body["plan"]["risk_level"] == "high"
    assert body["awaiting"]["plan_hash"]
    assert body["report"] is None


def test_an_incident_can_be_read_back(client) -> None:
    incident = client.post("/investigate", json=window_payload()).json()["incident_id"]

    body = client.get(f"/incidents/{incident}").json()

    assert body["status"] == "awaiting_approval"


def test_an_unknown_incident_is_not_found(client) -> None:
    assert client.get("/incidents/does-not-exist").status_code == 404


@pytest.mark.parametrize("values", [[], [[1.0, 2.0]], [[1.0, 2.0], [3.0]], [[], []]])
def test_a_malformed_window_is_rejected(client, values) -> None:
    response = client.post("/investigate", json={"values": values})

    assert response.status_code == 422


# --- what a window must be before anything is spent on it -------------------------


def with_value(value: float) -> dict:
    body = window_payload()
    body["values"][10][1] = value
    return body


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_missing_or_infinite_value_is_refused_before_any_model_call(tmp_path, bad) -> None:
    """NaN compares false against the threshold, so it used to read as an incident."""
    service = build_service(tmp_path)
    app = create_app(service=service, keyring=KeyRing({KEY: PRINCIPAL}))
    # Sent as raw json: Python's encoder, like many clients, writes NaN and Infinity as
    # bare tokens, and the server's parser accepts them.
    with TestClient(app) as local:
        response = local.post(
            "/investigate",
            content=json.dumps(with_value(bad)),
            headers={"Content-Type": "application/json"},
        )
    set_service(None)

    assert response.status_code == 422
    assert "must be finite" in response.json()["detail"][0]["msg"]
    assert service.investigator.diagnosis.provider.calls == []
    assert service.investigator.remediation.provider.calls == []
    assert service.audit.entries() == []


@pytest.mark.parametrize("rows", [2, 59, 61, 120])
def test_a_window_of_another_length_is_refused(client, rows) -> None:
    values = (disturbed_window().tolist() * 2)[:rows]

    response = client.post("/investigate", json={"values": values, "feature_names": ["a", "b"]})

    assert response.status_code == 422
    assert "expected 60 timesteps" in response.json()["detail"]


def test_a_window_with_another_number_of_metrics_is_refused(client) -> None:
    values = np.zeros((60, 3)).tolist()

    response = client.post("/investigate", json={"values": values})

    assert response.status_code == 422
    assert "expected 2 metrics" in response.json()["detail"]


def test_metric_names_in_another_order_are_refused_not_ignored(client) -> None:
    """Swapped columns used to be scored as if they were in the service's order."""
    body = window_payload() | {"feature_names": ["b", "a"]}

    response = client.post("/investigate", json=body)

    assert response.status_code == 422
    assert "in that order" in response.json()["detail"]


def test_a_window_without_names_is_taken_in_the_service_order(client) -> None:
    body = window_payload(anomalous=False)
    del body["feature_names"]

    assert client.post("/investigate", json=body).status_code == 200


def test_an_oversized_window_is_refused_before_it_becomes_an_array(client) -> None:
    response = client.post("/investigate", json={"values": [[1.0, 2.0]] * 5_001})

    assert response.status_code == 422


def test_a_refusal_does_not_echo_the_rejected_window(client) -> None:
    """The default handler returned every value back; the reason is enough."""
    response = client.post("/investigate", json={"values": [[1.0, 2.0]] * 5_001})

    assert "input" not in response.json()["detail"][0]
    assert len(response.content) < 1_000


def test_overlong_symptoms_are_refused(client) -> None:
    body = window_payload() | {"symptoms": "x" * 2_001}

    assert client.post("/investigate", json=body).status_code == 422


def test_a_bad_stream_request_fails_with_a_status_not_halfway_through(client) -> None:
    """Refused before the response starts, so the client is never told 200 first."""
    short = {"values": disturbed_window().tolist()[:30], "feature_names": ["a", "b"]}

    with client.stream("POST", "/investigate/stream", json=short) as stream:
        assert stream.status_code == 422


def test_the_gate_shows_the_command_approval_would_run(client) -> None:
    """A human approves a command, not a sentence."""
    awaiting = client.post("/investigate", json=window_payload()).json()["awaiting"]

    assert awaiting["action"] == "rollout_restart"
    assert awaiting["would_run"].startswith("kubectl rollout restart deployment/")


# --- authentication ----------------------------------------------------------------


def test_approving_without_a_key_is_unauthorised(client) -> None:
    incident = client.post("/investigate", json=window_payload()).json()["incident_id"]

    response = client.post(f"/incidents/{incident}/approve", json={"approved": True})

    assert response.status_code == 401


def test_approving_with_the_wrong_key_is_unauthorised(client) -> None:
    incident = client.post("/investigate", json=window_payload()).json()["incident_id"]

    response = client.post(
        f"/incidents/{incident}/approve",
        json={"approved": True},
        headers={"X-API-Key": "not-the-key"},
    )

    assert response.status_code == 401


def test_an_unauthorised_attempt_executes_nothing(client) -> None:
    incident = client.post("/investigate", json=window_payload()).json()["incident_id"]

    client.post(f"/incidents/{incident}/approve", json={"approved": True})

    assert client.get(f"/incidents/{incident}").json()["status"] == "awaiting_approval"
    assert client.get(f"/incidents/{incident}/audit").json()["entries"] == []


def test_a_valid_key_approves_and_is_recorded_by_name(client) -> None:
    incident = client.post("/investigate", json=window_payload()).json()["incident_id"]

    body = client.post(
        f"/incidents/{incident}/approve",
        json={"approved": True},
        headers={"X-API-Key": KEY},
    ).json()

    assert body["status"] == "complete"
    assert body["execution"]["executed"] is True
    entries = client.get(f"/incidents/{incident}/audit").json()["entries"]
    assert entries[0]["principal"] == PRINCIPAL


def test_an_empty_key_ring_accepts_nobody(tmp_path) -> None:
    service = build_service(tmp_path)
    app = create_app(service=service, keyring=KeyRing({}))
    with TestClient(app) as local:
        incident = local.post("/investigate", json=window_payload()).json()["incident_id"]
        response = local.post(
            f"/incidents/{incident}/approve",
            json={"approved": True},
            headers={"X-API-Key": KEY},
        )
    set_service(None)

    assert response.status_code == 401


# --- idempotency -------------------------------------------------------------------


def test_approving_twice_returns_the_same_outcome(client) -> None:
    """Networks retry. A recorded decision is returned, not applied again."""
    incident = client.post("/investigate", json=window_payload()).json()["incident_id"]
    headers = {"X-API-Key": KEY}

    first = client.post(f"/incidents/{incident}/approve", json={"approved": True}, headers=headers)
    second = client.post(f"/incidents/{incident}/approve", json={"approved": True}, headers=headers)

    assert first.status_code == second.status_code == 200
    assert first.json()["decision"] == second.json()["decision"]
    assert first.json()["report"] == second.json()["report"]


def test_a_repeat_does_not_execute_a_second_time(client) -> None:
    incident = client.post("/investigate", json=window_payload()).json()["incident_id"]
    headers = {"X-API-Key": KEY}
    client.post(f"/incidents/{incident}/approve", json={"approved": True}, headers=headers)
    client.post(f"/incidents/{incident}/approve", json={"approved": True}, headers=headers)

    assert len(client.get(f"/incidents/{incident}/audit").json()["entries"]) == 1


def test_a_reversal_after_the_fact_returns_the_recorded_decision(client) -> None:
    incident = client.post("/investigate", json=window_payload()).json()["incident_id"]
    headers = {"X-API-Key": KEY}
    client.post(f"/incidents/{incident}/approve", json={"approved": True}, headers=headers)

    body = client.post(
        f"/incidents/{incident}/approve", json={"approved": False}, headers=headers
    ).json()

    assert body["decision"] == "approved_by_human"


def test_approving_an_unknown_incident_is_not_found(client) -> None:
    response = client.post(
        "/incidents/nope/approve", json={"approved": True}, headers={"X-API-Key": KEY}
    )

    assert response.status_code == 404


# --- rejection ---------------------------------------------------------------------


def test_rejecting_escalates_and_executes_nothing(client) -> None:
    incident = client.post("/investigate", json=window_payload()).json()["incident_id"]

    body = client.post(
        f"/incidents/{incident}/approve", json={"approved": False}, headers={"X-API-Key": KEY}
    ).json()

    assert body["status"] == "rejected"
    assert body["execution"]["executed"] is False
    assert "escalated" in body["escalation"]


# --- streaming ---------------------------------------------------------------------


def test_the_stream_reports_each_stage_in_order(client) -> None:
    with client.stream("POST", "/investigate/stream", json=window_payload()) as stream:
        events = [line[7:].strip() for line in stream.iter_lines() if line.startswith("event: ")]

    assert events[0] == "accepted"
    assert events.index("triage") < events.index("diagnose") < events.index("remediate")
    assert "awaiting_approval" in events
    assert events[-1] == "done"


def test_a_calm_window_streams_only_triage(client) -> None:
    with client.stream(
        "POST", "/investigate/stream", json=window_payload(anomalous=False)
    ) as stream:
        events = [line[7:].strip() for line in stream.iter_lines() if line.startswith("event: ")]

    assert "diagnose" not in events
    assert events[-1] == "done"


def test_the_stream_does_not_leak_the_raw_window_or_the_token(client) -> None:
    with client.stream("POST", "/investigate/stream", json=window_payload()) as stream:
        body = "".join(stream.iter_text())

    assert '"window"' not in body
    assert '"approval"' not in body


# --- what reaches the browser ------------------------------------------------------


def test_hostile_model_output_is_returned_as_data_not_markup(tmp_path) -> None:
    """The api must not be the thing that sanitises, but it must not mangle either.

    Escaping here would hide the problem; the text is carried faithfully and the
    dashboard is responsible for never turning it into live markup.
    """
    service = build_service(tmp_path, diagnosis=HOSTILE_DIAGNOSIS, postmortem=HOSTILE_POSTMORTEM)
    app = create_app(service=service, keyring=KeyRing({KEY: PRINCIPAL}))

    with TestClient(app) as local:
        body = local.post("/investigate", json=window_payload()).json()
    set_service(None)

    assert body["diagnosis"]["summary"] == HOSTILE
    assert "<script>" in body["diagnosis"]["summary"]
    assert "&lt;script&gt;" not in body["diagnosis"]["summary"]


def test_the_dashboard_never_assigns_api_data_to_inner_html() -> None:
    """A structural guard: the sanitiser is the only route from model text to markup."""
    source = (Path(__file__).resolve().parents[1] / "web" / "index.html").read_text(
        encoding="utf-8"
    )

    assignments = re.findall(r"innerHTML\s*=\s*(.*?);", source, re.DOTALL)

    assert assignments, "expected at least one innerHTML assignment to check"
    for expression in assignments:
        assert "DOMPurify.sanitize" in expression, f"unsanitised assignment: {expression!r}"


def test_the_dashboard_loads_a_sanitiser() -> None:
    source = (Path(__file__).resolve().parents[1] / "web" / "index.html").read_text(
        encoding="utf-8"
    )

    assert "dompurify" in source.lower()
    assert source.index("purify.min.js") < source.index("DOMPurify.sanitize")


def test_the_dashboard_uses_text_content_for_plain_fields() -> None:
    source = (Path(__file__).resolve().parents[1] / "web" / "index.html").read_text(
        encoding="utf-8"
    )

    assert source.count("textContent") > 5


# --- metrics -----------------------------------------------------------------------


def test_metrics_report_traffic_and_audit_health(client) -> None:
    client.post("/investigate", json=window_payload(anomalous=False))

    body = client.get("/metrics").json()

    assert body["requests"]["investigate"] == 1
    assert body["audit_chain_intact"] is True
    assert body["uptime_seconds"] >= 0


def test_metrics_count_approvals(client) -> None:
    incident = client.post("/investigate", json=window_payload()).json()["incident_id"]
    client.post(
        f"/incidents/{incident}/approve", json={"approved": True}, headers={"X-API-Key": KEY}
    )

    assert client.get("/metrics").json()["requests"]["approve"] == 1


def test_the_demo_window_matches_the_expected_shape(client) -> None:
    body = client.get("/demo/window?anomalous=true").json()

    assert len(body["values"]) == 60
    assert len(body["feature_names"]) == len(body["values"][0])


def test_metrics_report_where_time_went(client) -> None:
    """The first question about a slow investigation is which stage was slow."""
    client.post("/investigate", json=window_payload())

    stages = client.get("/metrics").json()["stages"]

    assert "triage" in stages
    assert "diagnose" in stages
    assert stages["triage"]["count"] == 1
    assert stages["triage"]["p50_ms"] >= 0


def test_a_calm_run_times_only_the_stage_that_ran(client) -> None:
    client.post("/investigate", json=window_payload(anomalous=False))

    stages = client.get("/metrics").json()["stages"]

    assert "triage" in stages
    assert "diagnose" not in stages


# --- a run that stops partway -------------------------------------------------------


class BrokenRetriever:
    def search(self, query: str, k: int = 10):
        raise RuntimeError("index is gone at /secret/path")


@pytest.fixture
def broken(tmp_path):
    service = build_service(tmp_path)
    service.investigator.diagnosis.retriever = BrokenRetriever()
    app = create_app(service=service, keyring=KeyRing({KEY: PRINCIPAL}))
    with TestClient(app) as test_client:
        yield test_client
    set_service(None)


def test_a_run_that_stops_answers_500_with_where_to_look(broken) -> None:
    response = broken.post("/investigate", json=window_payload())

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["status"] == "failed"
    assert "/secret/path" not in response.text

    incident = broken.get(f"/incidents/{detail['incident_id']}").json()
    assert incident["status"] == "failed"
    assert incident["is_anomaly"] is True


def test_a_stream_that_stops_ends_with_a_failed_event(broken) -> None:
    with broken.stream("POST", "/investigate/stream", json=window_payload()) as stream:
        body = "".join(stream.iter_text())

    events = [line[7:].strip() for line in body.splitlines() if line.startswith("event: ")]
    assert events[-2:] == ["failed", "done"]
    assert '"status": "failed"' in body
    assert "/secret/path" not in body


# --- configuration --------------------------------------------------------------------


def test_a_service_without_a_model_reports_degraded_but_stays_up(tmp_path) -> None:
    from drdoom.llm.factory import UnavailableProvider

    service = build_service(tmp_path)
    missing = UnavailableProvider("no Groq credentials")
    for agent in (
        service.investigator.diagnosis,
        service.investigator.remediation,
        service.investigator.reporting,
    ):
        agent.provider = missing
    app = create_app(service=service, keyring=KeyRing({KEY: PRINCIPAL}))

    with TestClient(app) as local:
        health = local.get("/health")
        body = local.post("/investigate", json=window_payload()).json()
    set_service(None)

    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert health.json()["components"]["model"]["ready"] is False
    assert "Groq" not in health.text
    assert body["status"] == "awaiting_approval"
    assert body["degraded"] is True


def test_no_one_able_to_approve_reports_degraded(tmp_path) -> None:
    app = create_app(service=build_service(tmp_path), keyring=KeyRing({}))

    with TestClient(app) as local:
        body = local.get("/health").json()
    set_service(None)

    assert body["status"] == "degraded"
    assert body["components"]["approvals"]["ready"] is False


def test_a_store_that_does_not_answer_is_unavailable(tmp_path) -> None:
    service = build_service(tmp_path)
    app = create_app(service=service, keyring=KeyRing({KEY: PRINCIPAL}))

    with TestClient(app) as local:
        service.connection.close()
        response = local.get("/health")
    set_service(None)

    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"


def test_an_approval_key_written_in_the_local_env_file_is_accepted(tmp_path, monkeypatch) -> None:
    """The key ring used to be built at import, before the .env was ever read."""
    from drdoom.api import auth

    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("DRDOOM_API_KEYS=ops:from-the-file\n", encoding="utf-8")
    monkeypatch.setattr("drdoom.config.PROJECT_ROOT", project)
    monkeypatch.delenv(auth.KEYS_ENV, raising=False)

    app = create_app(service=build_service(tmp_path))
    try:
        with TestClient(app) as local:
            incident = local.post("/investigate", json=window_payload()).json()["incident_id"]
            response = local.post(
                f"/incidents/{incident}/approve",
                json={"approved": True},
                headers={"X-API-Key": "from-the-file"},
            )
    finally:
        set_service(None)
        auth.configure(KeyRing())

    assert response.status_code == 200
    assert response.json()["decision"] == "approved_by_human"


def test_a_missing_provider_key_starts_a_degraded_service_not_a_crash(monkeypatch) -> None:
    from drdoom.llm.base import LLMUnavailableError
    from drdoom.llm.factory import UnavailableProvider, build_provider_or_unavailable

    monkeypatch.setattr("drdoom.llm.factory.load_env_file", lambda: 0)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    provider = build_provider_or_unavailable("groq")

    assert isinstance(provider, UnavailableProvider)
    with pytest.raises(LLMUnavailableError):
        provider.complete([])


def test_health_reports_the_risk_assessor(client) -> None:
    assert client.get("/health").json()["components"]["risk_assessor"]["ready"] is True


def test_a_missing_risk_assessor_model_is_degraded_not_unsafe(tmp_path) -> None:
    from drdoom.llm.factory import UnavailableProvider

    service = build_service(tmp_path)
    service.investigator.risk.provider = UnavailableProvider("no key")
    app = create_app(service=service, keyring=KeyRing({KEY: PRINCIPAL}))

    with TestClient(app) as local:
        body = local.get("/health").json()
    set_service(None)

    assert body["status"] == "degraded"
    assert body["components"]["risk_assessor"]["ready"] is False


def test_the_gate_shows_how_the_risk_was_decided(client) -> None:
    body = client.post("/investigate", json=window_payload()).json()

    assert set(body["awaiting"]["risk"]) >= {"author", "floor", "assessor", "final"}
