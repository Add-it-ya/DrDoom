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

Between writing a plan and asking about it, the plan's risk is decided again. The author's
rating is one of three: a policy floor for the action it names and an independent
assessment are taken as well, and the most cautious stands. The plan the gate shows, and
whose hash an approval carries, is the plan with that final rating, so the rating a human
approved is part of what they approved. See ``drdoom.agents.risk``.

Approval and rejection lead to different places. An approved plan reaches an executor
carrying a token issued at the moment of the decision; a rejected one is escalated and
never reaches the executor at all. Both are written to the audit log, because a review
needs to see the refusals as much as the actions.

A run that stops partway is reported as ``failed``, never as ``no_incident``. The two
used to be indistinguishable -- both are a state with no report -- so an incident whose
diagnosis crashed read the same as a quiet window. What separates them is triage: only a
window triage cleared is not an incident.

The state is deliberately plain json. A numpy array or a pydantic model in the state would
serialise inconsistently or not at all; models are validated at the edges and stored as
dictionaries.
"""

from __future__ import annotations

import logging
import operator
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

import numpy as np
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from drdoom.agents.diagnosis import DiagnosisAgent
from drdoom.agents.remediation import RemediationAgent
from drdoom.agents.reporting import ReportingAgent
from drdoom.agents.risk import RiskAssessor, highest
from drdoom.agents.schemas import Citation, Diagnosis, RemediationPlan
from drdoom.agents.triage import TriageAgent
from drdoom.audit import AuditLog
from drdoom.config import get_settings
from drdoom.executor import ApprovalToken, DryRunExecutor, plan_hash, risk_floor
from drdoom.observability import Counters, Timings, incident_context, timed

logger = logging.getLogger(__name__)

Status = Literal["no_incident", "awaiting_approval", "complete", "rejected", "failed"]

AUTO_APPROVED = "auto_approved"
APPROVED = "approved_by_human"
REJECTED = "rejected_by_human"

POLICY_PRINCIPAL = "policy:low_risk"
UNKNOWN_PRINCIPAL = "unknown"

# What a caller is told when a run stops. The exception itself goes to the log, where the
# incident id ties it to this run, and not to a client that may be a browser.
STOPPED_DETAIL = "the investigation stopped at an internal error; the service log has the cause"
_STOPPED: dict[str, Any] = {}

# The right to answer an approval gate is claimed with a row in the checkpoint database, so
# it holds between threads, processes and restarts alike. A claim that is never finished
# is left only by a process that died mid-decision, and a decision takes seconds, so after
# this long another caller may take it over.
CLAIM_EXPIRY_SECONDS = 15 * 60
_CLAIMS_TABLE = (
    "CREATE TABLE IF NOT EXISTS decision_claims "
    "(thread_id TEXT PRIMARY KEY, principal TEXT NOT NULL, claimed_at REAL NOT NULL)"
)


class DecisionTakenError(RuntimeError):
    """Another caller already holds the decision on this incident."""


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
    risk: dict[str, Any]
    decision: str
    approval: dict[str, Any]
    execution: dict[str, Any]
    escalation: str
    report: str
    degraded: bool
    tokens: Annotated[int, operator.add]
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]
    model: str


@dataclass(frozen=True)
class Investigation:
    """The outcome of starting or resuming, and what is expected next."""

    thread_id: str
    status: Status
    state: dict[str, Any]
    pending: dict[str, Any] | None = None
    # When the stored state last changed. Known when read back from the store, not when
    # returned by the run that produced it.
    updated_at: str | None = None

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

    @property
    def usage(self) -> dict[str, Any]:
        """Token and cost accounting for this incident."""
        from drdoom.llm.pricing import describe

        return describe(
            self.state.get("model", "unknown"),
            int(self.state.get("input_tokens", 0)),
            int(self.state.get("output_tokens", 0)),
        )


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
        counters: Counters | None = None,
        risk: RiskAssessor | None = None,
    ) -> None:
        self.triage = triage
        self.diagnosis = diagnosis
        self.remediation = remediation
        self.reporting = reporting
        # A separate call, not a separate model, unless one is configured: a fresh context
        # that never sees the author's rating already removes the anchoring.
        self.risk = risk or RiskAssessor(remediation.provider)
        self.executor = executor or DryRunExecutor()
        self.audit = audit or AuditLog()
        self.counters = counters or Counters()
        self.checkpointer = checkpointer
        self.graph = self._build().compile(checkpointer=checkpointer)
        with self.checkpointer.cursor() as cursor:
            cursor.execute(_CLAIMS_TABLE)

    # --- nodes ---------------------------------------------------------------------

    def _triage_node(self, state: InvestigationState) -> dict:
        with timed("triage", self._timings(state)):
            return self._triage(state)

    def _triage(self, state: InvestigationState) -> dict:
        window = np.asarray(state["window"], dtype=np.float32)
        result = self.triage.run(window)
        logger.info("triage: anomaly=%s cause=%s", result.is_anomaly, result.root_cause)
        return {"triage": result.as_dict()}

    def _diagnose_node(self, state: InvestigationState) -> dict:
        with timed("diagnose", self._timings(state)):
            return self._diagnose(state)

    def _diagnose(self, state: InvestigationState) -> dict:
        triage = state["triage"]
        outcome = self.diagnosis.run(state["symptoms"], triage.get("root_cause"))
        return {
            "diagnosis": outcome.diagnosis.model_dump(),
            "citations": [citation.model_dump() for citation in outcome.citations],
            "degraded": outcome.degraded,
            **_usage(outcome.completions),
        }

    def _remediate_node(self, state: InvestigationState) -> dict:
        with timed("remediate", self._timings(state)):
            return self._remediate(state)

    def _remediate(self, state: InvestigationState) -> dict:
        outcome = self.remediation.run(
            state["diagnosis"]["summary"], state["triage"].get("root_cause")
        )
        update = {"plan": outcome.plan.model_dump(), **_usage(outcome.completions)}
        if outcome.degraded:
            # Set only on failure, so a plan that worked cannot clear a diagnosis that did not.
            update["degraded"] = True
        return update

    def _assess_node(self, state: InvestigationState) -> dict:
        with timed("assess_risk", self._timings(state)):
            return self._assess(state)

    def _assess(self, state: InvestigationState) -> dict:
        """Rate the plan again, independently, and keep the most cautious rating."""
        plan = RemediationPlan.model_validate(state["plan"])
        author = plan.risk_level
        floor = risk_floor(plan.action)
        record: dict[str, Any] = {"author": author, "floor": floor}
        update: dict[str, Any] = {}

        if author == "high":
            # Nothing can rate it higher, so there is nothing an assessment could change.
            record |= {"assessor": None, "assessor_status": "not_needed"}
            final = author
        else:
            outcome = self.risk.run(
                state["triage"],
                Diagnosis.model_validate(state["diagnosis"]),
                plan,
                self.executor.preview(plan),
                [Citation.model_validate(item) for item in state.get("citations", [])],
            )
            assessment = outcome.assessment
            record |= {
                "assessor": assessment.risk_level if assessment else None,
                "assessor_status": outcome.failure or "assessed",
                "worst_case": assessment.worst_case if assessment else None,
                "reasons": list(assessment.reasons) if assessment else [],
            }
            final = highest(floor, outcome.level, author)
            update |= _usage(outcome.completions)
            if outcome.degraded:
                update["degraded"] = True

        record["final"] = final
        if final != author:
            logger.info("risk raised from %s to %s (floor %s)", author, final, floor)
        update |= {
            "plan": plan.model_copy(update={"risk_level": final}).model_dump(),
            "risk": record,
        }
        return update

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
                "action": plan.action,
                "would_run": self.executor.preview(plan),
                "risk_level": plan.risk_level,
                "risk": state.get("risk"),
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
        with timed("execute", self._timings(state)):
            return self._execute(state)

    def _execute(self, state: InvestigationState) -> dict:
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
            risk=state.get("risk"),
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
            risk=state.get("risk"),
        )
        logger.warning("incident escalated after rejection")
        return {"escalation": message, "execution": {"executed": False, "kind": "rejected"}}

    def _report_node(self, state: InvestigationState) -> dict:
        with timed("report", self._timings(state)):
            return self._report(state)

    def _report(self, state: InvestigationState) -> dict:
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
        return {"report": markdown, **_usage(outcome.completions)}

    def _timings(self, state: InvestigationState) -> Timings:
        """A per-call collector that also feeds the process-wide counters."""
        collector = Timings()
        collector.record = _forwarding(collector, self.counters)  # type: ignore[method-assign]
        return collector

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
        graph.add_node("assess_risk", self._assess_node)
        graph.add_node("approval", self._approval_node)
        graph.add_node("execute", self._execute_node)
        graph.add_node("escalate", self._escalate_node)
        graph.add_node("report", self._report_node)

        graph.add_edge(START, "triage")
        graph.add_conditional_edges(
            "triage", self._route_after_triage, {"diagnose": "diagnose", END: END}
        )
        graph.add_edge("diagnose", "remediate")
        graph.add_edge("remediate", "assess_risk")
        graph.add_edge("assess_risk", "approval")
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
        if state.get("report"):
            return "rejected" if state.get("decision") == REJECTED else "complete"
        triage = state.get("triage")
        if not state or (triage is not None and not triage.get("is_anomaly")):
            return "no_incident"
        # There is a state, and either triage never finished or it found an incident, yet
        # no report was written and nothing is waiting for a human: the run stopped.
        return "failed"

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
            "input_tokens": 0,
            "output_tokens": 0,
            "degraded": False,
        }
        with incident_context(thread_id):
            result = self.graph.invoke(payload, config=self._config(thread_id))
        return self._outcome(thread_id, result)

    def resume(
        self, thread_id: str, approved: bool, principal: str = UNKNOWN_PRINCIPAL
    ) -> Investigation:
        """Answer a suspended investigation and run it to completion, at most once.

        Checking that the incident waits at the gate and then resuming it are two steps,
        and two approvals arriving together both passed the check: in 20 of 20 trials both
        were accepted and the plan ran twice. The right to answer is therefore claimed
        first, atomically; a caller that loses gets ``DecisionTakenError``.
        """
        self._claim(thread_id, principal)
        try:
            self.graph.update_state(self._config(thread_id), {"principal": principal})
            with incident_context(thread_id):
                result = self.graph.invoke(Command(resume=approved), config=self._config(thread_id))
        except Exception:
            self._release_if_unanswered(thread_id)
            raise
        return self._outcome(thread_id, result)

    def _claim(self, thread_id: str, principal: str) -> None:
        """Take the one right to answer this incident's gate, or raise ``DecisionTakenError``.

        An insert that the primary key refuses is a compare-and-set SQLite performs
        atomically, whichever thread or process gets there first.

        An old claim is taken over only while the incident still waits at the gate. A
        claim whose incident moved past the gate was not abandoned but finished, and the
        decision it recorded stands.
        """
        now = time.time()
        with self.checkpointer.cursor() as cursor:
            cursor.execute(
                "INSERT OR IGNORE INTO decision_claims VALUES (?, ?, ?)",
                (thread_id, principal, now),
            )
            if cursor.rowcount == 1:
                return
        if self.status(thread_id).status == "awaiting_approval":
            with self.checkpointer.cursor() as cursor:
                cursor.execute(
                    "UPDATE decision_claims SET principal = ?, claimed_at = ? "
                    "WHERE thread_id = ? AND claimed_at < ?",
                    (principal, now, thread_id, now - CLAIM_EXPIRY_SECONDS),
                )
                if cursor.rowcount == 1:
                    logger.warning("incident %s: took over a decision abandoned mid-way", thread_id)
                    return
        raise DecisionTakenError(f"a decision on incident {thread_id} is already being recorded")

    def _release_if_unanswered(self, thread_id: str) -> None:
        """Give the claim back when a failed resume left the incident waiting at the gate.

        Nothing past the gate ran in that case, so the decision can safely be tried again.
        Once the gate has been passed the claim stays, whatever happened after it.
        """
        if self.status(thread_id).status != "awaiting_approval":
            return
        with self.checkpointer.cursor() as cursor:
            cursor.execute("DELETE FROM decision_claims WHERE thread_id = ?", (thread_id,))

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
            "input_tokens": 0,
            "output_tokens": 0,
            "degraded": False,
        }
        yield from self._stream(payload, thread_id)

    def stream_resume(
        self, thread_id: str, approved: bool, principal: str = UNKNOWN_PRINCIPAL
    ) -> Iterator[dict[str, Any]]:
        """Answer a suspended investigation, streaming the remaining nodes, at most once."""
        self._claim(thread_id, principal)
        try:
            self.graph.update_state(self._config(thread_id), {"principal": principal})
            yield from self._stream(Command(resume=approved), thread_id)
        finally:
            self._release_if_unanswered(thread_id)

    def _stream(self, payload: Any, thread_id: str) -> Iterator[dict[str, Any]]:
        """Advance the graph one node at a time, inside the incident's log context.

        The context is entered around each step rather than around the whole generator.
        A server drives a streaming response from a thread pool, one step per call and not
        necessarily in the same context, so a variable set once at the top would neither
        reach the later steps nor be safe to reset. Each step is where the work happens,
        so each step is what has to be tagged.

        A step that raises ends the stream with a ``failed`` event instead of cutting the
        connection mid-response. The status line that follows says ``failed`` too.
        """
        updates = self.graph.stream(payload, config=self._config(thread_id), stream_mode="updates")
        while True:
            with incident_context(thread_id):
                try:
                    chunk = next(updates, None)
                except Exception:
                    logger.exception("investigation stopped at an internal error")
                    chunk = _STOPPED
            if chunk is None:
                break
            if chunk is _STOPPED:
                yield {"event": "failed", "data": {"detail": STOPPED_DETAIL}}
                break
            for node, update in chunk.items():
                if node == "__interrupt__":
                    yield {"event": "awaiting_approval", "data": dict(update[0].value)}
                else:
                    yield {"event": node, "data": _public(update)}
        with incident_context(thread_id):
            status = self.status(thread_id).status
        yield {"event": "done", "data": {"status": status}}

    def status(self, thread_id: str) -> Investigation:
        """Read a thread without advancing it."""
        snapshot = self.graph.get_state(self._config(thread_id))
        values = dict(snapshot.values or {})
        if snapshot.interrupts:
            return Investigation(
                thread_id,
                "awaiting_approval",
                values,
                dict(snapshot.interrupts[0].value),
                updated_at=snapshot.created_at,
            )
        return Investigation(
            thread_id, self._status(values), values, updated_at=snapshot.created_at
        )

    def recent(self, limit: int, offset: int = 0) -> tuple[list[Investigation], int]:
        """Stored investigations, newest first, and how many there are altogether.

        LangGraph has no call that lists threads, so this reads the key columns of its
        checkpoint table. Checkpoint ids are time-ordered, so a thread's smallest id marks
        when that investigation started. The API tests pin the order, so a change to
        either fails there rather than in front of someone.
        """
        with self.checkpointer.cursor(transaction=False) as cursor:
            (total,) = cursor.execute(
                "SELECT COUNT(DISTINCT thread_id) FROM checkpoints WHERE checkpoint_ns = ''"
            ).fetchone()
            rows = cursor.execute(
                "SELECT thread_id FROM checkpoints WHERE checkpoint_ns = '' "
                "GROUP BY thread_id ORDER BY MIN(checkpoint_id) DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self.status(thread_id) for (thread_id,) in rows], int(total)

    def prune(self, older_than: timedelta, now: datetime | None = None) -> int:
        """Delete stored investigations that finished longer ago than ``older_than``.

        Every step of every investigation is kept, about 57 KiB of checkpoints each, and
        nothing removed them. Only finished ones go: an incident waiting at the gate is
        kept however old, because a decision is still owed on it. The audit log is not
        touched; it is the record of what was decided and outlives the working state.
        """
        cutoff = (now or datetime.now(UTC)) - older_than
        with self.checkpointer.cursor(transaction=False) as cursor:
            threads = [
                thread_id
                for (thread_id,) in cursor.execute(
                    "SELECT DISTINCT thread_id FROM checkpoints WHERE checkpoint_ns = ''"
                ).fetchall()
            ]
        removed = 0
        for thread_id in threads:
            investigation = self.status(thread_id)
            if investigation.status == "awaiting_approval" or not investigation.updated_at:
                continue
            if datetime.fromisoformat(investigation.updated_at) >= cutoff:
                continue
            self.checkpointer.delete_thread(thread_id)
            with self.checkpointer.cursor() as cursor:
                cursor.execute("DELETE FROM decision_claims WHERE thread_id = ?", (thread_id,))
            removed += 1
        return removed


def _forwarding(collector: Timings, counters: Counters):
    """Record a stage on the per-call collector and the process counters at once."""

    def record(stage: str, milliseconds: float) -> None:
        collector.stages[stage] = round(milliseconds, 1)
        counters.observe(stage, milliseconds)

    return record


def _usage(completions: list) -> dict[str, Any]:
    """Roll a node's completions into the counters the state accumulates.

    The model is recorded only when one answered, so a node that degraded does not erase
    the name of the model an earlier node used.
    """
    usage: dict[str, Any] = {
        "tokens": sum(c.total_tokens for c in completions),
        "input_tokens": sum(c.input_tokens for c in completions),
        "output_tokens": sum(c.output_tokens for c in completions),
    }
    if completions:
        usage["model"] = completions[0].model
    return usage


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
