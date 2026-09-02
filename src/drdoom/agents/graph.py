"""The investigation pipeline: one definition, suspendable, durable.

An investigation pauses in the middle to ask a human a question, and the answer may not
arrive for hours. A request handler cannot block on that, and a predecessor project
resolved the tension by keeping the graph for a command-line demonstration and hand-rolling
the same sequence again behind its web api -- two definitions of one pipeline, with the
conditional routing live in neither place that ran.

The framework already solves this. ``interrupt`` suspends the graph mid-run, the
checkpointer writes the suspended state to sqlite, and a later call resumes the same
thread from where it stopped. Nothing is held in process memory between the two, so a
restart, a deploy, or a second worker picking up the request all behave the same. That is
the property a plain function call cannot offer, and the reason this is a graph at all.

The state is deliberately plain json. A numpy array or a pydantic model in the state would
serialise inconsistently or not at all; models are validated at the edges and stored as
dictionaries.
"""

from __future__ import annotations

import logging
import operator
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

import numpy as np
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from drdoom.agents.diagnosis import DiagnosisAgent
from drdoom.agents.remediation import RemediationAgent
from drdoom.agents.reporting import ReportingAgent
from drdoom.agents.schemas import Citation, Diagnosis, RemediationPlan
from drdoom.agents.triage import TriageAgent
from drdoom.config import get_settings

logger = logging.getLogger(__name__)

Status = Literal["no_incident", "awaiting_approval", "complete"]

AUTO_APPROVED = "auto_approved"
APPROVED = "approved_by_human"
REJECTED = "rejected_by_human"


class InvestigationState(TypedDict, total=False):
    """Everything one investigation knows, in a form sqlite can hold."""

    window: list[list[float]]
    feature_names: list[str]
    symptoms: str
    triage: dict[str, Any]
    diagnosis: dict[str, Any]
    citations: list[dict[str, Any]]
    plan: dict[str, Any]
    decision: str
    report: str
    degraded: bool
    tokens: Annotated[int, operator.add]


@dataclass(frozen=True)
class Investigation:
    """The outcome of starting or resuming, and what is expected next."""

    thread_id: str
    status: Status
    state: dict[str, Any]
    pending: dict[str, Any] | None = None

    @property
    def is_anomaly(self) -> bool:
        return bool(self.state.get("triage", {}).get("is_anomaly", False))

    @property
    def plan(self) -> dict[str, Any] | None:
        return self.state.get("plan")

    @property
    def report(self) -> str | None:
        return self.state.get("report")

    @property
    def tokens(self) -> int:
        return int(self.state.get("tokens", 0))


class Investigator:
    """Owns the compiled graph and the agents its nodes call."""

    def __init__(
        self,
        triage: TriageAgent,
        diagnosis: DiagnosisAgent,
        remediation: RemediationAgent,
        reporting: ReportingAgent,
        checkpointer: SqliteSaver,
    ) -> None:
        self.triage = triage
        self.diagnosis = diagnosis
        self.remediation = remediation
        self.reporting = reporting
        self.graph = self._build().compile(checkpointer=checkpointer)

    # --- nodes ---------------------------------------------------------------------

    def _triage_node(self, state: InvestigationState) -> dict:
        window = np.asarray(state["window"], dtype=np.float32)
        result = self.triage.run(window)
        logger.info("triage: anomaly=%s cause=%s", result.is_anomaly, result.root_cause)
        return {"triage": result.as_dict()}

    def _diagnose_node(self, state: InvestigationState) -> dict:
        triage = state["triage"]
        outcome = self.diagnosis.run(state["symptoms"], triage.get("root_cause"))
        return {
            "diagnosis": outcome.diagnosis.model_dump(),
            "citations": [citation.model_dump() for citation in outcome.citations],
            "degraded": outcome.degraded,
            "tokens": outcome.tokens,
        }

    def _remediate_node(self, state: InvestigationState) -> dict:
        outcome = self.remediation.run(
            state["diagnosis"]["summary"], state["triage"].get("root_cause")
        )
        return {"plan": outcome.plan.model_dump(), "tokens": outcome.tokens}

    def _approval_node(self, state: InvestigationState) -> dict:
        plan = state["plan"]
        if not plan["requires_approval"]:
            logger.info("plan is low risk, proceeding without a human")
            return {"decision": AUTO_APPROVED}

        # Everything below this line runs only after a human answers. The graph
        # suspends here and the state is on disk until it does.
        answer = interrupt(
            {
                "question": "Approve this remediation?",
                "immediate_action": plan["immediate_action"],
                "risk_level": plan["risk_level"],
                "rollback": plan["rollback"],
            }
        )
        decision = APPROVED if answer else REJECTED
        logger.info("human decision recorded: %s", decision)
        return {"decision": decision}

    def _report_node(self, state: InvestigationState) -> dict:
        diagnosis = Diagnosis.model_validate(state["diagnosis"])
        plan = RemediationPlan.model_validate(state["plan"])
        citations = [Citation.model_validate(item) for item in state.get("citations", [])]
        outcome = self.reporting.run(
            diagnosis, plan, state["decision"], executed=None, citations=citations
        )
        return {"report": outcome.markdown, "tokens": outcome.tokens}

    @staticmethod
    def _route_after_triage(state: InvestigationState) -> str:
        return "diagnose" if state["triage"]["is_anomaly"] else END

    # --- wiring --------------------------------------------------------------------

    def _build(self) -> StateGraph:
        graph = StateGraph(InvestigationState)
        graph.add_node("triage", self._triage_node)
        graph.add_node("diagnose", self._diagnose_node)
        graph.add_node("remediate", self._remediate_node)
        graph.add_node("approval", self._approval_node)
        graph.add_node("report", self._report_node)

        graph.add_edge(START, "triage")
        graph.add_conditional_edges(
            "triage", self._route_after_triage, {"diagnose": "diagnose", END: END}
        )
        graph.add_edge("diagnose", "remediate")
        graph.add_edge("remediate", "approval")
        graph.add_edge("approval", "report")
        graph.add_edge("report", END)
        return graph

    # --- driving -------------------------------------------------------------------

    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def _outcome(self, thread_id: str, result: dict) -> Investigation:
        interrupts = result.get("__interrupt__") or []
        clean = {key: value for key, value in result.items() if not key.startswith("__")}

        if interrupts:
            return Investigation(
                thread_id=thread_id,
                status="awaiting_approval",
                state=clean,
                pending=dict(interrupts[0].value),
            )
        status: Status = "complete" if clean.get("report") else "no_incident"
        return Investigation(thread_id=thread_id, status=status, state=clean)

    def start(
        self,
        window: np.ndarray,
        symptoms: str,
        thread_id: str,
        feature_names: list[str] | None = None,
    ) -> Investigation:
        """Run until the pipeline finishes or stops to ask for approval."""
        payload: InvestigationState = {
            "window": np.asarray(window, dtype=np.float32).tolist(),
            "feature_names": feature_names or list(self.triage.feature_names),
            "symptoms": symptoms,
            "tokens": 0,
            "degraded": False,
        }
        return self._outcome(thread_id, self.graph.invoke(payload, config=self._config(thread_id)))

    def resume(self, thread_id: str, approved: bool) -> Investigation:
        """Answer a suspended investigation and run it to completion."""
        result = self.graph.invoke(Command(resume=approved), config=self._config(thread_id))
        return self._outcome(thread_id, result)

    def status(self, thread_id: str) -> Investigation:
        """Read a thread without advancing it."""
        snapshot = self.graph.get_state(self._config(thread_id))
        values = dict(snapshot.values or {})
        pending = None
        if snapshot.interrupts:
            pending = dict(snapshot.interrupts[0].value)
            return Investigation(thread_id, "awaiting_approval", values, pending)
        status: Status = "complete" if values.get("report") else "no_incident"
        return Investigation(thread_id, status, values)


def checkpoint_path() -> Path:
    return get_settings().project_root / "state" / "investigations.sqlite"


@contextmanager
def open_checkpointer(path: Path | None = None) -> Iterator[SqliteSaver]:
    """Open the durable store, creating its directory if needed."""
    location = path or checkpoint_path()
    location.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(location, check_same_thread=False)
    try:
        yield SqliteSaver(connection)
    finally:
        connection.close()
