"""The http surface over the investigation pipeline.

Nothing about the pipeline is reimplemented here. The routes start a graph run, resume a
suspended one, and read state; the ordering, the routing and the approval gate all live in
the graph. That is the whole point of the previous stage: a predecessor project kept its
graph for a command-line demonstration and hand-wrote the sequence again behind its api,
and the two drifted.

Three properties this layer is responsible for.

**Approving requires authentication**, and the authenticated name is what the audit log
records. Reading is open; deciding is not.

**Approving twice is safe.** Networks retry. An approval that has already been recorded
returns the same outcome rather than a 404 or a second execution.

**Model output is returned as data, never as markup.** The api hands back the text it
generated; turning that into html is the browser's job, and the dashboard does it through
a sanitiser. See the note in ``web/dashboard.html``.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from drdoom.agents.graph import Investigation, Investigator
from drdoom.api.auth import KeyRing, Principal, configure, require_principal
from drdoom.audit import AuditLog
from drdoom.config import get_settings
from drdoom.observability import configure_logging, incident_context

logger = logging.getLogger(__name__)

WINDOW_LENGTH = 60
TERMINAL = {"complete", "rejected"}
_FEATURES = ("latency_ms", "error_rate_pct", "cpu_pct", "queue_depth")


# --- request and response shapes ---------------------------------------------------


class MetricWindow(BaseModel):
    """A window of telemetry to investigate."""

    values: list[list[float]] = Field(
        description="One row per timestep, one column per metric",
    )
    feature_names: list[str] | None = None
    symptoms: str = Field(default="", description="What the reporter observed")

    @field_validator("values")
    @classmethod
    def check_shape(cls, values: list[list[float]]) -> list[list[float]]:
        if len(values) < 2:
            raise ValueError("a window needs at least two timesteps")
        widths = {len(row) for row in values}
        if len(widths) != 1:
            raise ValueError("every row must have the same number of metrics")
        if not widths.pop():
            raise ValueError("a window needs at least one metric")
        return values


class Decision(BaseModel):
    approved: bool


class InvestigationView(BaseModel):
    """What a client is told about an investigation."""

    incident_id: str
    status: str
    is_anomaly: bool
    triage: dict[str, Any] | None = None
    diagnosis: dict[str, Any] | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    plan: dict[str, Any] | None = None
    decision: str | None = None
    execution: dict[str, Any] | None = None
    escalation: str | None = None
    report: str | None = None
    degraded: bool = False
    tokens: int = 0
    usage: dict[str, Any] | None = None
    awaiting: dict[str, Any] | None = None

    @classmethod
    def of(cls, investigation: Investigation) -> InvestigationView:
        state = investigation.state
        return cls(
            incident_id=investigation.thread_id,
            status=investigation.status,
            is_anomaly=investigation.is_anomaly,
            triage=state.get("triage"),
            diagnosis=state.get("diagnosis"),
            citations=state.get("citations", []),
            plan=state.get("plan"),
            decision=state.get("decision"),
            execution=state.get("execution"),
            escalation=state.get("escalation"),
            report=state.get("report"),
            degraded=bool(state.get("degraded", False)),
            tokens=investigation.tokens,
            usage=investigation.usage,
            awaiting=investigation.pending,
        )


# --- application state -------------------------------------------------------------


@dataclass
class Service:
    """What the routes need, assembled once at startup."""

    investigator: Investigator
    audit: AuditLog
    connection: Any = None  # held so the checkpointer's sqlite handle outlives startup
    started_at: float = field(default_factory=time.monotonic)
    counters: dict[str, int] = field(default_factory=dict)

    def count(self, name: str) -> None:
        self.counters[name] = self.counters.get(name, 0) + 1

    @property
    def stage_latencies(self) -> dict[str, Any]:
        return self.investigator.counters.snapshot()


_service: Service | None = None


def set_service(service: Service | None) -> None:
    global _service
    _service = service


def get_service() -> Service:
    if _service is None:  # pragma: no cover - only when misconfigured
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="service is not ready"
        )
    return _service


CurrentService = Annotated[Service, Depends(get_service)]
Approver = Annotated[Principal, Depends(require_principal)]


def _window(payload: MetricWindow) -> np.ndarray:
    return np.asarray(payload.values, dtype=np.float32)


def _sse(events: Iterator[dict[str, Any]]) -> Iterator[str]:
    for event in events:
        yield f"event: {event['event']}\ndata: {json.dumps(event['data'], default=str)}\n\n"


# --- routes ------------------------------------------------------------------------


def create_app(service: Service | None = None, keyring: KeyRing | None = None) -> FastAPI:
    """Build the application. Passing a service skips startup assembly, which tests use."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = get_settings()
        configure_logging(settings.log_level, structured=settings.environment != "local")
        if _service is None:
            from drdoom.api.factory import build_service

            set_service(build_service())
        yield
        set_service(None)

    if service is not None:
        set_service(service)
    configure(keyring or KeyRing.from_environment())

    app = FastAPI(
        title="DrDoom",
        summary="Autonomous incident response with a human approval gate",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics(current: CurrentService) -> dict[str, Any]:
        """Traffic, where time goes, and whether the decision record is intact."""
        valid, reason = current.audit.verify()
        latencies = current.stage_latencies
        return {
            "uptime_seconds": round(time.monotonic() - current.started_at, 1),
            "requests": dict(current.counters),
            "stages": latencies["stages"],
            "audit_entries": len(current.audit.entries()),
            "audit_chain_intact": valid,
            "audit_chain_detail": reason,
        }

    @app.post("/investigate", response_model=InvestigationView)
    def investigate(payload: MetricWindow, current: CurrentService) -> InvestigationView:
        """Start an investigation and return where it stopped."""
        current.count("investigate")
        incident_id = uuid.uuid4().hex[:12]
        outcome = current.investigator.start(
            _window(payload), payload.symptoms, incident_id, payload.feature_names
        )
        logger.info("incident %s finished in state %s", incident_id, outcome.status)
        return InvestigationView.of(outcome)

    @app.post("/investigate/stream")
    def investigate_stream(payload: MetricWindow, current: CurrentService) -> StreamingResponse:
        """The same run, delivered a stage at a time."""
        current.count("investigate_stream")
        incident_id = uuid.uuid4().hex[:12]

        def events() -> Iterator[dict[str, Any]]:
            yield {"event": "accepted", "data": {"incident_id": incident_id}}
            yield from current.investigator.stream_start(
                _window(payload), payload.symptoms, incident_id, payload.feature_names
            )

        return StreamingResponse(
            _sse(events()),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/demo/window")
    def demo_window(anomalous: bool = True) -> dict[str, Any]:
        """A window shaped like the detector expects, so the dashboard has input."""
        from drdoom.api.factory import demo_window as make_window

        window = make_window(anomalous=anomalous)
        return {
            "values": window.tolist(),
            "feature_names": list(_FEATURES),
            "symptoms": (
                "latency and queue depth climbing over the last half hour"
                if anomalous
                else "routine check, nothing reported"
            ),
        }

    @app.get("/incidents/{incident_id}", response_model=InvestigationView)
    def read_incident(incident_id: str, current: CurrentService) -> InvestigationView:
        with incident_context(incident_id):
            outcome = current.investigator.status(incident_id)
        if outcome.status == "no_incident" and not outcome.state:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such incident")
        return InvestigationView.of(outcome)

    @app.post("/incidents/{incident_id}/approve", response_model=InvestigationView)
    def approve(
        incident_id: str,
        decision: Decision,
        current: CurrentService,
        principal: Approver,
    ) -> InvestigationView:
        """Answer a suspended investigation. Requires a credential; repeats are safe."""
        current.count("approve")
        existing = current.investigator.status(incident_id)

        if not existing.state:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such incident")

        if existing.status in TERMINAL:
            # Networks retry. A decision that is already recorded is returned as it
            # stands rather than applied a second time.
            logger.info("incident %s already decided, returning the recorded outcome", incident_id)
            return InvestigationView.of(existing)

        if existing.status != "awaiting_approval":
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"incident {incident_id} is not waiting for a decision",
            )

        outcome = current.investigator.resume(
            incident_id, approved=decision.approved, principal=principal.name
        )
        logger.info("incident %s decided by %s", incident_id, principal.name)
        return InvestigationView.of(outcome)

    @app.get("/incidents/{incident_id}/audit")
    def incident_audit(incident_id: str, current: CurrentService) -> dict:
        entries = current.audit.for_incident(incident_id)
        valid, reason = current.audit.verify()
        return {
            "incident_id": incident_id,
            "entries": [entry.payload() | {"entry_hash": entry.entry_hash} for entry in entries],
            "chain_intact": valid,
            "chain_detail": reason,
        }

    dashboard = get_settings().project_root / "web"
    if dashboard.is_dir():
        app.mount("/", StaticFiles(directory=dashboard, html=True), name="dashboard")

    @app.exception_handler(ValueError)
    async def value_error(request: Request, error: ValueError):  # pragma: no cover
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(error)) from error

    return app


app = create_app()
