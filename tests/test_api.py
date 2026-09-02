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
from tests._restart_worker import DIAGNOSIS, PLAN, POSTMORTEM, disturbed_window

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
        TriageAgent(detector, threshold=5.0, feature_names=["a", "b"]),
        DiagnosisAgent(retriever, StubProvider(default=diagnosis)),
        RemediationAgent(retriever, StubProvider(default=PLAN)),
        ReportingAgent(StubProvider(default=postmortem)),
        checkpointer,
        audit=audit,
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
    assert client.get("/health").json() == {"status": "ok"}


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


def test_a_reversal_after_the_fact_is_refused(client) -> None:
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
