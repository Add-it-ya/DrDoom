"""The http surface: authentication, idempotency, streaming, and what reaches the browser.

The pipeline itself is tested elsewhere. What is under test here is the layer around it.
"""

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
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
from drdoom.api.limits import RateLimiter
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
AUTH = {"X-API-Key": KEY}

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

    body = client.get(f"/incidents/{incident}", headers=AUTH).json()

    assert body["status"] == "awaiting_approval"


def test_an_unknown_incident_is_not_found(client) -> None:
    assert client.get("/incidents/does-not-exist", headers=AUTH).status_code == 404


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

    assert (
        client.get(f"/incidents/{incident}", headers=AUTH).json()["status"] == "awaiting_approval"
    )
    assert client.get(f"/incidents/{incident}/audit", headers=AUTH).json()["entries"] == []


def test_a_valid_key_approves_and_is_recorded_by_name(client) -> None:
    incident = client.post("/investigate", json=window_payload()).json()["incident_id"]

    body = client.post(
        f"/incidents/{incident}/approve",
        json={"approved": True},
        headers={"X-API-Key": KEY},
    ).json()

    assert body["status"] == "complete"
    assert body["execution"]["executed"] is True
    entries = client.get(f"/incidents/{incident}/audit", headers=AUTH).json()["entries"]
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


def test_an_expired_key_approves_nothing_and_health_says_so(tmp_path) -> None:
    from datetime import UTC, datetime

    expired = KeyRing({KEY: PRINCIPAL}, {KEY: datetime(2020, 1, 1, tzinfo=UTC)})
    app = create_app(service=build_service(tmp_path), keyring=expired)
    with TestClient(app) as local:
        incident = local.post("/investigate", json=window_payload()).json()["incident_id"]
        response = local.post(
            f"/incidents/{incident}/approve", json={"approved": True}, headers=AUTH
        )
        approvals = local.get("/health").json()["components"]["approvals"]
    set_service(None)

    assert response.status_code == 401
    assert approvals["ready"] is False


@pytest.mark.parametrize("path", ["/incidents/{id}", "/incidents/{id}/audit", "/metrics"])
def test_reading_what_the_service_knows_requires_a_key(client, path: str) -> None:
    """An incident shows the command approval would run, and its id is no secret."""
    incident = client.post("/investigate", json=window_payload()).json()["incident_id"]
    url = path.format(id=incident)

    assert client.get(url).status_code == 401
    assert client.get(url, headers={"X-API-Key": "not-the-key"}).status_code == 401
    assert client.get(url, headers=AUTH).status_code == 200


def test_starting_an_investigation_stays_open(client) -> None:
    """The demo has to be clickable; the caller already holds the window it sent."""
    assert client.get("/demo/window").status_code == 200
    assert client.post("/investigate", json=window_payload()).status_code == 200


def test_starting_too_many_investigations_is_refused_with_429(tmp_path) -> None:
    """Open to anyone, each run costs thousands of model tokens."""
    app = create_app(
        service=build_service(tmp_path),
        keyring=KeyRing({KEY: PRINCIPAL}),
        limiter=RateLimiter(per_caller=2, total=100),
    )
    with TestClient(app) as local:
        codes = [
            local.post("/investigate", json=window_payload(False)).status_code for _ in range(3)
        ]
        streamed = local.post("/investigate/stream", json=window_payload(False))
        keyed = local.post("/investigate", json=window_payload(False), headers=AUTH)
        started = local.get("/incidents", headers=AUTH).json()["total"]
    set_service(None)

    assert codes == [200, 200, 429]
    assert streamed.status_code == 429
    assert int(streamed.headers["retry-after"]) >= 1
    assert keyed.status_code == 200, "a caller with a key has its own allowance"
    assert started == 3, "a refused request starts nothing"


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

    assert len(client.get(f"/incidents/{incident}/audit", headers=AUTH).json()["entries"]) == 1


def approve_together(client, incident: str) -> list:
    barrier = threading.Barrier(2)

    def approve(_: int):
        barrier.wait()
        return client.post(f"/incidents/{incident}/approve", json={"approved": True}, headers=AUTH)

    with ThreadPoolExecutor(2) as pool:
        return list(pool.map(approve, range(2)))


def test_two_approvals_at_once_execute_once(client) -> None:
    """Checked and then resumed, both were accepted and the plan ran twice in 20 of 20."""
    for _ in range(3):
        incident = client.post("/investigate", json=window_payload()).json()["incident_id"]

        responses = approve_together(client, incident)

        codes = sorted(response.status_code for response in responses)
        assert codes in ([200, 200], [200, 409])
        for response in responses:
            if response.status_code == 409:
                assert "already being recorded" in response.json()["detail"]
        entries = client.get(f"/incidents/{incident}/audit", headers=AUTH).json()["entries"]
        assert len(entries) == 1


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


def web(name: str) -> str:
    return (Path(__file__).resolve().parents[1] / "web" / name).read_text(encoding="utf-8")


def test_the_dashboard_never_assigns_api_data_to_inner_html() -> None:
    """A structural guard: the sanitiser is the only route from model text to markup."""
    source = web("app.js")

    assignments = re.findall(r"innerHTML\s*=\s*(.*?);", source, re.DOTALL)

    assert assignments, "expected at least one innerHTML assignment to check"
    for expression in assignments:
        assert "DOMPurify.sanitize" in expression, f"unsanitised assignment: {expression!r}"


def test_the_dashboard_loads_a_sanitiser_before_its_script() -> None:
    page = web("index.html")

    assert page.index("purify.min.js") < page.index('<script src="app.js">')
    assert "DOMPurify.sanitize" in web("app.js")


def test_every_script_from_elsewhere_is_pinned_by_hash() -> None:
    """A compromised CDN would otherwise replace the sanitiser the page relies on."""
    external = re.findall(r"<script\s[^>]*src=\"https?://[^>]*>", web("index.html"))

    assert len(external) == 2
    for tag in external:
        assert re.search(r'integrity="sha(384|512)-[A-Za-z0-9+/]+=*"', tag), tag
        assert 'crossorigin="anonymous"' in tag, tag


def test_the_dashboard_uses_text_content_for_plain_fields() -> None:
    assert web("app.js").count("textContent") > 5


def test_the_page_has_nothing_inline_for_the_policy_to_allow() -> None:
    """The policy refuses inline script and style, so the page must not need either."""
    page = web("index.html")

    assert "<style" not in page
    assert all("src=" in tag for tag in re.findall(r"<script[^>]*>", page))
    assert not re.search(r"\son[a-z]+\s*=", page), "inline event handler"
    assert not re.search(r"\sstyle\s*=", page), "inline style attribute"


def test_the_policy_allows_exactly_the_scripts_the_page_loads() -> None:
    from drdoom.api.headers import CDN_SCRIPTS

    loaded = re.findall(r'<script\s[^>]*src="(https?://[^"]+)"', web("index.html"))

    assert sorted(loaded) == sorted(CDN_SCRIPTS)


def test_the_dashboard_is_served_with_a_content_security_policy(client) -> None:
    response = client.get("/")
    policy = response.headers["content-security-policy"]
    directives = dict(item.strip().split(" ", 1) for item in policy.split(";"))

    assert response.status_code == 200
    assert "unsafe-inline" not in policy and "unsafe-eval" not in policy
    assert directives["default-src"] == "'none'"
    assert directives["script-src"].split()[0] == "'self'"
    assert directives["connect-src"] == "'self'"
    assert directives["img-src"] == "'self'"
    assert directives["frame-ancestors"] == "'none'"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_api_responses_carry_the_policy_too(client) -> None:
    """A json body opened directly in a browser is a page as well."""
    assert "content-security-policy" in client.get("/health").headers
    assert "content-security-policy" in client.get("/incidents", headers=AUTH).headers


def test_the_interactive_docs_are_left_out_of_the_policy(client) -> None:
    """They load their own scripts from another CDN; they show the schema, not model text."""
    response = client.get("/docs")

    assert response.status_code == 200
    assert "content-security-policy" not in response.headers
    assert response.headers["x-content-type-options"] == "nosniff"


# --- listing -----------------------------------------------------------------------


def start(client, anomalous: bool = True) -> str:
    return client.post("/investigate", json=window_payload(anomalous)).json()["incident_id"]


def listed(page: dict) -> list[str]:
    return [item["incident_id"] for item in page["incidents"]]


def test_listing_incidents_requires_a_key(client) -> None:
    """The list maps everything that has gone wrong, so it is not public."""
    start(client)

    assert client.get("/incidents").status_code == 401
    assert client.get("/incidents", headers={"X-API-Key": "not-the-key"}).status_code == 401


def test_incidents_are_listed_newest_first(client) -> None:
    started = [start(client, anomalous) for anomalous in (True, False, True)]

    page = client.get("/incidents", headers=AUTH).json()

    assert listed(page) == list(reversed(started))
    assert page["total"] == 3


def test_the_list_is_paged(client) -> None:
    first, second, third = (start(client) for _ in range(3))

    front = client.get("/incidents?limit=2", headers=AUTH).json()
    back = client.get("/incidents?limit=2&offset=2", headers=AUTH).json()

    assert listed(front) == [third, second]
    assert listed(back) == [first]
    assert front["total"] == back["total"] == 3


def test_a_listed_incident_says_where_it_stands(client) -> None:
    incident = start(client)

    [item] = client.get("/incidents", headers=AUTH).json()["incidents"]

    assert item["incident_id"] == incident
    assert item["status"] == "awaiting_approval"
    assert item["is_anomaly"] is True
    assert (
        item["risk_level"]
        == client.get(f"/incidents/{incident}", headers=AUTH).json()["risk"]["final"]
    )
    assert item["updated_at"]


def test_a_calm_window_is_listed_as_no_incident(client) -> None:
    start(client, anomalous=False)

    [item] = client.get("/incidents", headers=AUTH).json()["incidents"]

    assert item["status"] == "no_incident"
    assert item["is_anomaly"] is False


@pytest.mark.parametrize("query", ["limit=0", "limit=101", "offset=-1"])
def test_an_unreasonable_page_is_refused(client, query: str) -> None:
    assert client.get(f"/incidents?{query}", headers=AUTH).status_code == 422


def test_the_dashboard_sends_the_key_when_listing() -> None:
    source = web("app.js")

    listing = source[source.index('fetch("/incidents?') :]

    assert '"X-API-Key"' in listing[: listing.index(");")]


def test_the_dashboard_sends_the_key_when_opening_an_incident() -> None:
    source = web("app.js")

    opening = source[source.index('fetch("/incidents/" + encodeURIComponent(id)') :]

    assert '"X-API-Key"' in opening[: opening.index(");")]


# --- metrics -----------------------------------------------------------------------


def test_metrics_report_traffic_and_audit_health(client) -> None:
    client.post("/investigate", json=window_payload(anomalous=False))

    body = client.get("/metrics", headers=AUTH).json()

    assert body["requests"]["investigate"] == 1
    assert body["audit_chain_intact"] is True
    assert body["uptime_seconds"] >= 0


def test_metrics_count_approvals(client) -> None:
    incident = client.post("/investigate", json=window_payload()).json()["incident_id"]
    client.post(
        f"/incidents/{incident}/approve", json={"approved": True}, headers={"X-API-Key": KEY}
    )

    body = client.get("/metrics", headers=AUTH).json()
    entries = client.get(f"/incidents/{incident}/audit", headers=AUTH).json()["entries"]

    assert body["requests"]["approve"] == 1
    assert body["audit_head"] == f"1:{entries[0]['entry_hash']}"


def test_the_demo_window_matches_the_expected_shape(client) -> None:
    body = client.get("/demo/window?anomalous=true").json()

    assert len(body["values"]) == 60
    assert len(body["feature_names"]) == len(body["values"][0])


def test_metrics_report_where_time_went(client) -> None:
    """The first question about a slow investigation is which stage was slow."""
    client.post("/investigate", json=window_payload())

    stages = client.get("/metrics", headers=AUTH).json()["stages"]

    assert "triage" in stages
    assert "diagnose" in stages
    assert stages["triage"]["count"] == 1
    assert stages["triage"]["p50_ms"] >= 0


def test_a_calm_run_times_only_the_stage_that_ran(client) -> None:
    client.post("/investigate", json=window_payload(anomalous=False))

    stages = client.get("/metrics", headers=AUTH).json()["stages"]

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

    incident = broken.get(f"/incidents/{detail['incident_id']}", headers=AUTH).json()
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


def test_health_names_the_reranker(client) -> None:
    assert client.get("/health").json()["components"]["retriever"]["reranker"] == "none"


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
