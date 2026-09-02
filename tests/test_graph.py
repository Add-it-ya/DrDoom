"""The investigation pipeline: routing, suspension, and surviving a restart.

The durability test is the one that justifies the whole design. If suspended state did
not outlive the process, a plain sequence of function calls would do the same job with
less machinery.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from drdoom.agents.diagnosis import DiagnosisAgent
from drdoom.agents.graph import (
    APPROVED,
    AUTO_APPROVED,
    REJECTED,
    Investigator,
    open_checkpointer,
)
from drdoom.agents.remediation import RemediationAgent
from drdoom.agents.reporting import ReportingAgent
from drdoom.agents.triage import TriageAgent, window_to_series
from drdoom.data.windows import Scaler
from drdoom.detect.baselines import WindowSpread
from drdoom.llm.stub import StubProvider
from drdoom.rag.corpus import Document
from drdoom.rag.index import BM25Index
from drdoom.rag.ingest import chunk_all
from tests._restart_worker import DIAGNOSIS, PLAN, POSTMORTEM, disturbed_window

WORKER = Path(__file__).parent / "_restart_worker.py"

LOW_RISK_PLAN = json.dumps(
    {
        "immediate_action": "Record a note on the dashboard",
        "risk_level": "low",
        "short_term_fix": "a",
        "long_term_fix": "b",
        "rollback": "c",
    }
)


def calm_window(seed: int = 2) -> np.ndarray:
    return np.random.default_rng(seed).normal(50, 1, size=(60, 2)).astype(np.float32)


def make_investigator(checkpointer, plan_json: str = PLAN, threshold: float = 5.0) -> Investigator:
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

    return Investigator(
        TriageAgent(detector, threshold=threshold, feature_names=["a", "b"]),
        DiagnosisAgent(retriever, StubProvider(default=DIAGNOSIS)),
        RemediationAgent(retriever, StubProvider(default=plan_json)),
        ReportingAgent(StubProvider(default=POSTMORTEM)),
        checkpointer,
    )


# --- routing -----------------------------------------------------------------------


def test_a_calm_window_stops_after_triage(tmp_path) -> None:
    with open_checkpointer(tmp_path / "s.sqlite") as checkpointer:
        outcome = make_investigator(checkpointer).start(calm_window(), "all quiet", "calm")

    assert outcome.status == "no_incident"
    assert outcome.is_anomaly is False


def test_no_model_is_called_when_nothing_is_wrong(tmp_path) -> None:
    """The conditional edge is what keeps a quiet system from costing anything."""
    with open_checkpointer(tmp_path / "s.sqlite") as checkpointer:
        investigator = make_investigator(checkpointer)
        outcome = investigator.start(calm_window(), "all quiet", "calm")

    assert investigator.diagnosis.provider.calls == []
    assert investigator.remediation.provider.calls == []
    assert "diagnosis" not in outcome.state
    assert outcome.tokens == 0


def test_an_incident_runs_through_to_the_approval_gate(tmp_path) -> None:
    with open_checkpointer(tmp_path / "s.sqlite") as checkpointer:
        outcome = make_investigator(checkpointer).start(
            disturbed_window(), "latency climbing", "incident"
        )

    assert outcome.status == "awaiting_approval"
    assert outcome.plan["risk_level"] == "high"
    assert outcome.report is None


def test_the_gate_asks_a_question_a_human_can_answer(tmp_path) -> None:
    with open_checkpointer(tmp_path / "s.sqlite") as checkpointer:
        outcome = make_investigator(checkpointer).start(
            disturbed_window(), "latency climbing", "incident"
        )

    assert set(outcome.pending) >= {"question", "immediate_action", "risk_level", "rollback"}
    assert outcome.pending["risk_level"] == "high"


def test_a_low_risk_plan_never_stops(tmp_path) -> None:
    with open_checkpointer(tmp_path / "s.sqlite") as checkpointer:
        outcome = make_investigator(checkpointer, plan_json=LOW_RISK_PLAN).start(
            disturbed_window(), "minor blip", "minor"
        )

    assert outcome.status == "complete"
    assert outcome.state["decision"] == AUTO_APPROVED
    assert outcome.report


# --- decisions ---------------------------------------------------------------------


def test_approving_completes_the_investigation(tmp_path) -> None:
    with open_checkpointer(tmp_path / "s.sqlite") as checkpointer:
        investigator = make_investigator(checkpointer)
        investigator.start(disturbed_window(), "latency climbing", "incident")

        outcome = investigator.resume("incident", approved=True)

    assert outcome.status == "complete"
    assert outcome.state["decision"] == APPROVED
    assert outcome.report.startswith("# ")


def test_rejecting_is_recorded_as_a_rejection(tmp_path) -> None:
    with open_checkpointer(tmp_path / "s.sqlite") as checkpointer:
        investigator = make_investigator(checkpointer)
        investigator.start(disturbed_window(), "latency climbing", "incident")

        outcome = investigator.resume("incident", approved=False)

    assert outcome.state["decision"] == REJECTED
    assert outcome.report


def test_the_decision_reaches_the_written_report(tmp_path) -> None:
    with open_checkpointer(tmp_path / "s.sqlite") as checkpointer:
        investigator = make_investigator(checkpointer)
        investigator.start(disturbed_window(), "latency climbing", "incident")
        investigator.resume("incident", approved=False)

    prompt = investigator.reporting.provider.calls[0][0].content
    assert REJECTED in prompt


def test_threads_do_not_interfere(tmp_path) -> None:
    with open_checkpointer(tmp_path / "s.sqlite") as checkpointer:
        investigator = make_investigator(checkpointer)
        investigator.start(disturbed_window(), "first", "one")
        investigator.start(disturbed_window(), "second", "two")

        investigator.resume("one", approved=True)

        assert investigator.status("one").status == "complete"
        assert investigator.status("two").status == "awaiting_approval"


def test_status_reads_without_advancing(tmp_path) -> None:
    with open_checkpointer(tmp_path / "s.sqlite") as checkpointer:
        investigator = make_investigator(checkpointer)
        investigator.start(disturbed_window(), "latency climbing", "incident")

        first = investigator.status("incident")
        second = investigator.status("incident")

    assert first.status == second.status == "awaiting_approval"
    assert first.report is None


def test_tokens_accumulate_across_nodes(tmp_path) -> None:
    with open_checkpointer(tmp_path / "s.sqlite") as checkpointer:
        investigator = make_investigator(checkpointer)
        investigator.start(disturbed_window(), "latency climbing", "incident")
        outcome = investigator.resume("incident", approved=True)

    assert outcome.tokens > 0


# --- durability --------------------------------------------------------------------


def test_state_survives_rebuilding_every_object(tmp_path) -> None:
    database = tmp_path / "s.sqlite"

    with open_checkpointer(database) as first:
        make_investigator(first).start(disturbed_window(), "latency climbing", "incident")

    with open_checkpointer(database) as second:
        rebuilt = make_investigator(second)
        recovered = rebuilt.status("incident")
        outcome = rebuilt.resume("incident", approved=True)

    assert recovered.status == "awaiting_approval"
    assert recovered.plan["immediate_action"] == "Rolling restart of the affected pods"
    assert outcome.status == "complete"
    assert outcome.report


def test_an_investigation_resumes_in_a_different_process(tmp_path) -> None:
    """The property a plain function call cannot offer.

    One interpreter suspends at the gate and exits. A second interpreter, sharing nothing
    but the sqlite file, answers and finishes the work.
    """
    database = tmp_path / "s.sqlite"

    started = subprocess.run(
        [sys.executable, str(WORKER), "start", str(database), "incident"],
        capture_output=True,
        text=True,
        check=True,
    )
    suspended = json.loads(started.stdout)

    resumed = subprocess.run(
        [sys.executable, str(WORKER), "resume", str(database), "incident", "true"],
        capture_output=True,
        text=True,
        check=True,
    )
    finished = json.loads(resumed.stdout)

    assert suspended["status"] == "awaiting_approval"
    assert suspended["report"] is None
    assert suspended["pending"]["risk_level"] == "high"

    assert finished["status"] == "complete"
    assert finished["decision"] == APPROVED
    assert finished["report"].startswith("# Memory leak in the api service")


def test_a_rejection_also_survives_a_restart(tmp_path) -> None:
    database = tmp_path / "s.sqlite"

    subprocess.run(
        [sys.executable, str(WORKER), "start", str(database), "incident"],
        capture_output=True,
        text=True,
        check=True,
    )
    resumed = subprocess.run(
        [sys.executable, str(WORKER), "resume", str(database), "incident", "false"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert json.loads(resumed.stdout)["decision"] == REJECTED


def test_the_checkpoint_file_is_created_on_demand(tmp_path) -> None:
    database = tmp_path / "nested" / "deeper" / "s.sqlite"

    with open_checkpointer(database) as checkpointer:
        make_investigator(checkpointer).start(calm_window(), "quiet", "calm")

    assert database.is_file()


def test_resuming_an_unknown_thread_does_not_invent_a_report(tmp_path) -> None:
    with open_checkpointer(tmp_path / "s.sqlite") as checkpointer:
        investigator = make_investigator(checkpointer)

        assert investigator.status("never-existed").report is None


@pytest.mark.parametrize("approved", [True, False])
def test_every_decision_produces_a_report(tmp_path, approved: bool) -> None:
    with open_checkpointer(tmp_path / "s.sqlite") as checkpointer:
        investigator = make_investigator(checkpointer)
        investigator.start(disturbed_window(), "latency climbing", "incident")
        outcome = investigator.resume("incident", approved=approved)

    assert outcome.status == "complete"
    assert outcome.report
