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

Approval and rejection lead to different places. An approved plan reaches an executor
carrying a token issued at the moment of the decision; a rejected one is escalated and
never reaches the executor at all. Both are written to the audit log, because a review
needs to see the refusals as much as the actions.

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
from dataclasses import asdict, dataclass
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
from drdoom.audit import AuditLog
from drdoom.config import get_settings
from drdoom.executor import ApprovalToken, DryRunExecutor, plan_hash

logger = logging.getLogger(__name__)

Status = Literal["no_incident", "awaiting_approval", "complete", "rejected"]

AUTO_APPROVED = "auto_approved"
APPROVED = "approved_by_human"
REJECTED = "rejected_by_human"

POLICY_PRINCIPAL = "policy:low_risk"
UNKNOWN_PRINCIPAL = "unknown"


class InvestigationState(TypedDict, total=False):
    """Everything one investigation knows, in a form sqlite can hold."""

    window: list[list[float]]
    feature_names: list[str]
    symptoms: str
    thread_id: str
    principal: str
    triage: dict[str, Any]
    diagnosis: dict[str, Any]
    citations: list[dict[str, Any]]
    plan: dict[str, Any]
    decision: str
    approval: dict[str, Any]
    execution: dict[str, Any]
    escalation: str
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
    def execution(self) -> dict[str, Any] | None:
        return self.state.get("execution")

    @property
    def executed(self) -> bool:
        return bool((self.state.get("execution") or {}).get("executed", False))

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
        executor: DryRunExecutor | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self.triage = triage
        self.diagnosis = diagnosis
        self.remediation = remediation
        self.reporting = reporting
        self.executor = executor or DryRunExecutor()
        self.audit = audit or AuditLog()
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
        """Decide, and on approval mint the token that authorises this exact plan."""
        plan = RemediationPlan.model_validate(state["plan"])

        if not plan.requires_approval:
            logger.info("plan is low risk, proceeding without a human")
            token = ApprovalToken.issue(plan, POLICY_PRINCIPAL)
            return {"decision": AUTO_APPROVED, "approval": asdict(token)}

        # Everything below this line runs only after a human answers. The graph
        # suspends here and the state is on disk until it does.
        answer = interrupt(
            {
                "question": "Approve this remediation?",
                "immediate_action": plan.immediate_action,
                "risk_level": plan.risk_level,
                "rollback": plan.rollback,
                "plan_hash": plan_hash(plan),
            }
        )
        approved = bool(answer)
        principal = state.get("principal", UNKNOWN_PRINCIPAL)
        logger.info("human decision recorded: approved=%s by %s", approved, principal)

        if not approved:
            return {"decision": REJECTED}
        return {"decision": APPROVED, "approval": asdict(ApprovalToken.issue(plan, principal))}

    def _execute_node(self, state: InvestigationState) -> dict:
        """Run the approved plan, and write what happened to the audit log."""
        plan = RemediationPlan.model_validate(state["plan"])
        approval = state.get("approval")
        token = ApprovalToken(**approval) if approval else None

        result = self.executor.execute(plan, token)
        self.audit.record(
            incident_id=state.get("thread_id", "") or "unknown",
            principal=token.principal if token else UNKNOWN_PRINCIPAL,
            decision=state["decision"],
            risk_level=plan.risk_level,
            immediate_action=plan.immediate_action,
            plan_hash=plan_hash(plan),
            executed=result.executed,
            execution=result.summary,
        )
        return {"execution": result.as_dict()}

    def _escalate_node(self, state: InvestigationState) -> dict:
        """A rejection is a decision with consequences, not a quiet ending."""
        plan = RemediationPlan.model_validate(state["plan"])
        principal = state.get("principal", UNKNOWN_PRINCIPAL)
        message = (
            f"Remediation rejected by {principal}. Nothing was executed. "
            "The incident remains open and is escalated to the on-call engineer."
        )
        self.audit.record(
            incident_id=state.get("thread_id", "") or "unknown",
            principal=principal,
            decision=REJECTED,
            risk_level=plan.risk_level,
            immediate_action=plan.immediate_action,
            plan_hash=plan_hash(plan),
            executed=False,
            execution="nothing was executed (rejected)",
        )
        logger.warning("incident escalated after rejection")
        return {"escalation": message, "execution": {"executed": False, "kind": "rejected"}}

    def _report_node(self, state: InvestigationState) -> dict:
        diagnosis = Diagnosis.model_validate(state["diagnosis"])
        plan = RemediationPlan.model_validate(state["plan"])
        citations = [Citation.model_validate(item) for item in state.get("citations", [])]
        execution = state.get("execution") or {}
        executed = execution.get("command") if execution.get("executed") else None

        outcome = self.reporting.run(
            diagnosis, plan, state["decision"], executed=executed, citations=citations
        )
        markdown = outcome.markdown
        if state.get("escalation"):
            markdown += f"\n> {state['escalation']}\n"
        return {"report": markdown, "tokens": outcome.tokens}

    @staticmethod
    def _route_after_triage(state: InvestigationState) -> str:
        return "diagnose" if state["triage"]["is_anomaly"] else END

    @staticmethod
    def _route_after_approval(state: InvestigationState) -> str:
        return "escalate" if state["decision"] == REJECTED else "execute"

    # --- wiring --------------------------------------------------------------------

    def _build(self) -> StateGraph:
        graph = StateGraph(InvestigationState)
        graph.add_node("triage", self._triage_node)
        graph.add_node("diagnose", self._diagnose_node)
        graph.add_node("remediate", self._remediate_node)
        graph.add_node("approval", self._approval_node)
        graph.add_node("execute", self._execute_node)
        graph.add_node("escalate", self._escalate_node)
        graph.add_node("report", self._report_node)

        graph.add_edge(START, "triage")
        graph.add_conditional_edges(
            "triage", self._route_after_triage, {"diagnose": "diagnose", END: END}
        )
        graph.add_edge("diagnose", "remediate")
        graph.add_edge("remediate", "approval")
        graph.add_conditional_edges(
            "approval", self._route_after_approval, {"execute": "execute", "escalate": "escalate"}
        )
        graph.add_edge("execute", "report")
        graph.add_edge("escalate", "report")
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
        return Investigation(thread_id=thread_id, status=self._status(clean), state=clean)

    @staticmethod
    def _status(state: dict[str, Any]) -> Status:
        if not state.get("report"):
            return "no_incident"
        return "rejected" if state.get("decision") == REJECTED else "complete"

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
            "thread_id": thread_id,
            "tokens": 0,
            "degraded": False,
        }
        return self._outcome(thread_id, self.graph.invoke(payload, config=self._config(thread_id)))

    def resume(
        self, thread_id: str, approved: bool, principal: str = UNKNOWN_PRINCIPAL
    ) -> Investigation:
        """Answer a suspended investigation and run it to completion."""
        self.graph.update_state(self._config(thread_id), {"principal": principal})
        result = self.graph.invoke(Command(resume=approved), config=self._config(thread_id))
        return self._outcome(thread_id, result)

    def stream_start(
        self,
        window: np.ndarray,
        symptoms: str,
        thread_id: str,
        feature_names: list[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield one event per completed node, so a caller can show progress.

        The alternative is a single request that returns nothing for the length of the
        whole pipeline, which is indistinguishable from a hang.
        """
        payload: InvestigationState = {
            "window": np.asarray(window, dtype=np.float32).tolist(),
            "feature_names": feature_names or list(self.triage.feature_names),
            "symptoms": symptoms,
            "thread_id": thread_id,
            "tokens": 0,
            "degraded": False,
        }
        yield from self._stream(payload, thread_id)

    def stream_resume(
        self, thread_id: str, approved: bool, principal: str = UNKNOWN_PRINCIPAL
    ) -> Iterator[dict[str, Any]]:
        """Answer a suspended investigation, streaming the remaining nodes."""
        self.graph.update_state(self._config(thread_id), {"principal": principal})
        yield from self._stream(Command(resume=approved), thread_id)

    def _stream(self, payload: Any, thread_id: str) -> Iterator[dict[str, Any]]:
        for chunk in self.graph.stream(
            payload, config=self._config(thread_id), stream_mode="updates"
        ):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    yield {"event": "awaiting_approval", "data": dict(update[0].value)}
                else:
                    yield {"event": node, "data": _public(update)}
        yield {"event": "done", "data": {"status": self.status(thread_id).status}}

    def status(self, thread_id: str) -> Investigation:
        """Read a thread without advancing it."""
        snapshot = self.graph.get_state(self._config(thread_id))
        values = dict(snapshot.values or {})
        if snapshot.interrupts:
            return Investigation(
                thread_id, "awaiting_approval", values, dict(snapshot.interrupts[0].value)
            )
        return Investigation(thread_id, self._status(values), values)


def _public(update: Any) -> dict[str, Any]:
    """Trim a node update to what a client can be shown.

    The raw window is large and uninteresting to a reader, and the approval token is
    evidence rather than display material.
    """
    if not isinstance(update, dict):
        return {}
    hidden = {"window", "feature_names", "approval"}
    return {key: value for key, value in update.items() if key not in hidden}


def checkpoint_path() -> Path:
    return get_settings().project_root / "state" / "investigations.sqlite"


def make_checkpointer(path: Path | None = None) -> tuple[SqliteSaver, sqlite3.Connection]:
    """Open the durable store and hand back both halves.

    The connection is returned because its lifetime is the caller's problem. A long-lived
    service must keep it referenced; borrowing the context manager and discarding it
    leaves the connection to be closed by garbage collection, which fails later and
    somewhere else.
    """
    location = path or checkpoint_path()
    location.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(location, check_same_thread=False)
    return SqliteSaver(connection), connection


@contextmanager
def open_checkpointer(path: Path | None = None) -> Iterator[SqliteSaver]:
    """Open the durable store for the length of a block."""
    saver, connection = make_checkpointer(path)
    try:
        yield saver
    finally:
        connection.close()
