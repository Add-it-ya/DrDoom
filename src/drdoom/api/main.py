"""The http surface over the investigation pipeline.

Nothing about the pipeline is reimplemented here. The routes start a graph run, resume a
suspended one, and read state; the ordering, the routing and the approval gate all live in
the graph. That is the whole point of the previous stage: a predecessor project kept its
graph for a command-line demonstration and hand-wrote the sequence again behind its api,
and the two drifted.

Three properties this layer is responsible for.

**Approving requires authentication**, and the authenticated name is what the audit log
records. So does reading what the service knows: an incident, its audit trail, the list
of incidents and the metrics. Open are only health, the demo window, and starting an
investigation, whose caller already has the window. The keys are read at startup, after
the local .env has been loaded, so a key written there is one the service accepts.

**Approving twice is safe.** Networks retry. An approval that has already been recorded
returns the same outcome rather than a 404 or a second execution. Two decisions arriving
together cannot both run: the first claims the gate, and the second gets 409, or the
first one's outcome if it has already finished.

**A window is checked before anything is spent on it.** Values must be finite and the
shape must be the one the detector's threshold was calibrated for, or the request is
refused with 422 before the graph starts. A malformed payload costs no retrieval and no
model call, and never reaches the approval gate as a phantom incident.

**Model output is returned as data, never as markup.** The api hands back the text it
generated; turning that into html is the browser's job, and the dashboard does it through
a sanitiser. See the note in ``web/index.html``. Every response carries a Content Security
Policy as the layer behind the sanitiser (``api/headers.py``).

**Health describes the parts, not the process.** ``/health`` says whether a model is
configured, whether anyone can approve, and whether the investigation store answers. A
service without a model still works in its degraded form, so that is ``degraded`` with a
200; a store that does not answer means nothing works, so that is a 503.
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from drdoom.agents.graph import STOPPED_DETAIL, DecisionTakenError, Investigation, Investigator
from drdoom.api.auth import (
    KeyRing,
    PresentedKey,
    Principal,
    configure,
    current_keyring,
    require_principal,
)
from drdoom.api.headers import SecurityHeaders
from drdoom.api.limits import RateLimiter
from drdoom.audit import AuditLog
from drdoom.config import get_settings, load_env_file
from drdoom.llm.factory import UnavailableProvider
from drdoom.observability import configure_logging, incident_context

logger = logging.getLogger(__name__)

# An upper bound checked before the service-specific shape, so an oversized body is
# refused without being turned into an array. A day of four metrics is 5760 cells.
MAX_CELLS = 10_000
MAX_SYMPTOMS = 2_000
TERMINAL = {"complete", "rejected"}
MAX_PAGE = 100


# --- request and response shapes ---------------------------------------------------


class MetricWindow(BaseModel):
    """A window of telemetry to investigate."""

    values: list[list[float]] = Field(
        description="One row per timestep, one column per metric",
    )
    feature_names: list[str] | None = Field(
        default=None,
        description="Metric names in column order; if given, must match the service's order",
    )
    symptoms: str = Field(
        default="", max_length=MAX_SYMPTOMS, description="What the reporter observed"
    )

    @field_validator("values")
    @classmethod
    def check_shape(cls, values: list[list[float]]) -> list[list[float]]:
        if len(values) < 2:
            raise ValueError("a window needs at least two timesteps")
        widths = {len(row) for row in values}
        if len(widths) != 1:
            raise ValueError("every row must have the same number of metrics")
        width = widths.pop()
        if not width:
            raise ValueError("a window needs at least one metric")
        if len(values) * width > MAX_CELLS:
            raise ValueError(f"a window may hold at most {MAX_CELLS} values")
        for row_number, row in enumerate(values):
            for column, value in enumerate(row):
                if not math.isfinite(value):
                    raise ValueError(
                        f"values must be finite; got {value} at timestep {row_number}, "
                        f"column {column}"
                    )
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
    risk: dict[str, Any] | None = None

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
            risk=state.get("risk"),
            decision=state.get("decision"),
            execution=state.get("execution"),
            escalation=state.get("escalation"),
            report=state.get("report"),
            degraded=bool(state.get("degraded", False)),
            tokens=investigation.tokens,
            usage=investigation.usage,
            awaiting=investigation.pending,
        )


class IncidentSummary(BaseModel):
    """One row of the incident list: enough to pick one out, not the whole record."""

    incident_id: str
    status: str
    updated_at: str | None = None
    is_anomaly: bool
    likely_cause: str | None = None
    action: str | None = None
    risk_level: str | None = None

    @classmethod
    def of(cls, investigation: Investigation) -> IncidentSummary:
        state = investigation.state
        plan = state.get("plan") or {}
        return cls(
            incident_id=investigation.thread_id,
            status=investigation.status,
            updated_at=investigation.updated_at,
            is_anomaly=investigation.is_anomaly,
            likely_cause=(state.get("diagnosis") or {}).get("likely_cause"),
            action=plan.get("immediate_action"),
            risk_level=(state.get("risk") or {}).get("final") or plan.get("risk_level"),
        )


class IncidentPage(BaseModel):
    incidents: list[IncidentSummary]
    total: int
    limit: int
    offset: int


# --- application state -------------------------------------------------------------


@dataclass
class Service:
    """What the routes need, assembled once at startup."""

    investigator: Investigator
    audit: AuditLog
    connection: Any = None  # held so the checkpointer's sqlite handle outlives startup
    started_at: float = field(default_factory=time.monotonic)

    def count(self, name: str) -> None:
        # The investigator's counters take a lock; a plain dict here lost increments when
        # two requests landed together.
        self.investigator.counters.increment(name)

    @property
    def stage_latencies(self) -> dict[str, Any]:
        return self.investigator.counters.snapshot()

    def health(self) -> tuple[str, dict[str, Any]]:
        """The overall verdict, and what each part it rests on reports.

        Nothing here names a key or repeats an error message: the endpoint is public.
        """
        triage = self.investigator.triage
        provider = self.investigator.diagnosis.provider
        reviewer = self.investigator.risk.provider
        model_ready = not isinstance(provider, UnavailableProvider)
        reviewer_ready = not isinstance(reviewer, UnavailableProvider)
        approvals_ready = current_keyring().usable() > 0
        try:
            if self.connection is not None:
                self.connection.execute("select 1").fetchone()
            store_ready = True
        except Exception:
            logger.exception("the investigation store did not answer")
            store_ready = False

        components = {
            "model": {"ready": model_ready, "provider": provider.name, "name": provider.model},
            # Without it every plan below high is treated as high, so nothing is unsafe,
            # but every plan waits for a human.
            "risk_assessor": {
                "ready": reviewer_ready,
                "provider": reviewer.name,
                "name": reviewer.model,
            },
            "approvals": {"ready": approvals_ready},
            "store": {"ready": store_ready},
            "detector": {"ready": True, "name": triage.detector.name},
            "classifier": {"ready": triage.classifier is not None},
            "retriever": {
                "ready": True,
                "name": type(self.investigator.diagnosis.retriever).__name__,
                "reranker": self.investigator.diagnosis.reranker.name,
            },
        }
        if not store_ready:
            return "unavailable", components
        # The classifier is optional: without it incidents are unclassified, which the
        # dashboard shows, and nothing else changes.
        ready = model_ready and reviewer_ready and approvals_ready
        return ("ok" if ready else "degraded"), components


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
# Reading takes the same credential as deciding, but the reader's name is not recorded, so
# the check is declared on the route rather than taken as an argument.
KEY_REQUIRED = [Depends(require_principal)]


def limit_investigations(request: Request, presented: PresentedKey) -> None:
    """Refuse with 429 once a caller, or everyone together, has started too many.

    A caller with a valid key is counted under its name, so a proxy in front of several
    operators does not make them share one address's allowance.
    """
    limiter: RateLimiter | None = request.app.state.limiter
    if limiter is None:
        return
    principal = current_keyring().resolve(presented)
    address = request.client.host if request.client else "unknown"
    caller = f"key:{principal.name}" if principal else f"address:{address}"
    wait = limiter.admit(caller)
    if wait is not None:
        logger.warning("refused an investigation from %s: rate limit", caller)
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "too many investigations started; try again later",
            headers={"Retry-After": str(max(1, math.ceil(wait)))},
        )


RATE_LIMITED = [Depends(limit_investigations)]


def _window(payload: MetricWindow, current: Service) -> np.ndarray:
    """The payload as an array the detector can score, or a 422 saying why not.

    Checked here, before the graph starts, so that a streaming request fails with a status
    code rather than an error event halfway through a response that already said 200.
    """
    triage = current.investigator.triage
    if payload.feature_names is not None and list(payload.feature_names) != triage.feature_names:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"feature_names must be {triage.feature_names} in that order, "
            f"got {list(payload.feature_names)}",
        )
    window = np.asarray(payload.values, dtype=np.float32)
    try:
        triage.check(window)
    except ValueError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error
    return window


def _stopped(incident_id: str, error: Exception) -> HTTPException:
    """A run that raised, logged under its incident and answered with where to look.

    The id is returned so the partial state can be read back, and reads as ``failed``.
    """
    with incident_context(incident_id):
        logger.error("investigation stopped at an internal error", exc_info=error)
    return HTTPException(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        {"incident_id": incident_id, "status": "failed", "message": STOPPED_DETAIL},
    )


def _sse(events: Iterator[dict[str, Any]]) -> Iterator[str]:
    for event in events:
        yield f"event: {event['event']}\ndata: {json.dumps(event['data'], default=str)}\n\n"


# --- routes ------------------------------------------------------------------------


def create_app(
    service: Service | None = None,
    keyring: KeyRing | None = None,
    limiter: RateLimiter | None = None,
) -> FastAPI:
    """Build the application. Passing a service skips startup assembly, which tests use.

    Without a limiter, one is built from the settings; a limit of zero lifts it.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = get_settings()
        configure_logging(settings.log_level, structured=settings.environment != "local")
        if keyring is None:
            # Read here and not when the module is imported. Until now the local .env was
            # loaded only inside provider construction, after the key ring had already been
            # built, so a key written there was never accepted.
            load_env_file()
            configure(KeyRing.from_environment())
        if _service is None:
            from drdoom.api.factory import build_service

            set_service(build_service())
        yield
        set_service(None)

    if service is not None:
        set_service(service)
    if keyring is not None:
        configure(keyring)

    app = FastAPI(
        title="DrDoom",
        summary="Autonomous incident response with a human approval gate",
        lifespan=lifespan,
    )
    settings = get_settings()
    app.state.limiter = limiter or RateLimiter(
        settings.investigate_per_minute, settings.investigate_per_minute_total
    )
    docs = (app.docs_url, app.redoc_url, app.swagger_ui_oauth2_redirect_url)
    app.add_middleware(SecurityHeaders, exempt=frozenset(path for path in docs if path))

    @app.get("/health")
    def health() -> JSONResponse:
        if _service is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"status": "starting"}
            )
        verdict, components = _service.health()
        code = status.HTTP_503_SERVICE_UNAVAILABLE if verdict == "unavailable" else 200
        return JSONResponse(status_code=code, content={"status": verdict, "components": components})

    @app.get("/metrics", dependencies=KEY_REQUIRED)
    def metrics(current: CurrentService) -> dict[str, Any]:
        """Traffic, where time goes, and whether the decision record is intact.

        Requires a credential: request counts and the size of the audit log describe how
        the service is used, which is nobody else's business.
        """
        valid, reason = current.audit.verify()
        snapshot = current.stage_latencies
        return {
            "uptime_seconds": round(time.monotonic() - current.started_at, 1),
            "requests": snapshot["events"],
            "stages": snapshot["stages"],
            "audit_entries": len(current.audit.entries()),
            # Worth recording elsewhere: `python -m drdoom.audit --anchor` checks against it.
            "audit_head": current.audit.head(),
            "audit_chain_intact": valid,
            "audit_chain_detail": reason,
        }

    @app.post("/investigate", response_model=InvestigationView, dependencies=RATE_LIMITED)
    def investigate(payload: MetricWindow, current: CurrentService) -> InvestigationView:
        """Start an investigation and return where it stopped."""
        current.count("investigate")
        window = _window(payload, current)
        incident_id = uuid.uuid4().hex[:12]
        try:
            outcome = current.investigator.start(
                window, payload.symptoms, incident_id, payload.feature_names
            )
        except Exception as error:
            raise _stopped(incident_id, error) from error
        logger.info("incident %s finished in state %s", incident_id, outcome.status)
        return InvestigationView.of(outcome)

    @app.post("/investigate/stream", dependencies=RATE_LIMITED)
    def investigate_stream(payload: MetricWindow, current: CurrentService) -> StreamingResponse:
        """The same run, delivered a stage at a time."""
        current.count("investigate_stream")
        window = _window(payload, current)
        incident_id = uuid.uuid4().hex[:12]

        def events() -> Iterator[dict[str, Any]]:
            yield {"event": "accepted", "data": {"incident_id": incident_id}}
            yield from current.investigator.stream_start(
                window, payload.symptoms, incident_id, payload.feature_names
            )

        return StreamingResponse(
            _sse(events()),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/demo/window")
    def demo_window(anomalous: bool = True) -> dict[str, Any]:
        """A window shaped like the detector expects, so the dashboard has input.

        The names come from the generator that made the values, so they always describe
        this window. A service configured for other metrics refuses it with a 422.
        """
        from drdoom.api.factory import demo_window as make_window
        from drdoom.data.synthetic import FEATURE_NAMES

        window = make_window(anomalous=anomalous)
        return {
            "values": window.tolist(),
            "feature_names": list(FEATURE_NAMES),
            "symptoms": (
                "latency and queue depth climbing over the last half hour"
                if anomalous
                else "routine check, nothing reported"
            ),
        }

    @app.get("/incidents", response_model=IncidentPage, dependencies=KEY_REQUIRED)
    def list_incidents(
        current: CurrentService,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE)] = 20,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> IncidentPage:
        """Recent investigations, newest first. Requires a credential.

        A list of every incident is a map of what has gone wrong and what was done about
        it.
        """
        investigations, total = current.investigator.recent(limit, offset)
        return IncidentPage(
            incidents=[IncidentSummary.of(item) for item in investigations],
            total=total,
            limit=limit,
            offset=offset,
        )

    @app.get(
        "/incidents/{incident_id}", response_model=InvestigationView, dependencies=KEY_REQUIRED
    )
    def read_incident(incident_id: str, current: CurrentService) -> InvestigationView:
        """One investigation as it stands. Requires a credential.

        An incident holds the diagnosis, the command approval would run and the plan's
        hash, and its identifier is twelve hex characters, not a secret.
        """
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

        try:
            outcome = current.investigator.resume(
                incident_id, approved=decision.approved, principal=principal.name
            )
        except DecisionTakenError as error:
            # Another request is answering this gate. If it has finished, its outcome is
            # the answer to this one as well; if not, this one arrived second.
            settled = current.investigator.status(incident_id)
            if settled.status in TERMINAL:
                return InvestigationView.of(settled)
            raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error
        except Exception as error:
            raise _stopped(incident_id, error) from error
        logger.info("incident %s decided by %s", incident_id, principal.name)
        return InvestigationView.of(outcome)

    @app.get("/incidents/{incident_id}/audit", dependencies=KEY_REQUIRED)
    def incident_audit(incident_id: str, current: CurrentService) -> dict:
        """Who decided what for one incident. Requires a credential."""
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

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
        """Say where a request is wrong and why, without echoing what was sent.

        The default handler returns the rejected input. For a window that is up to ten
        thousand numbers sent straight back, and when the input held NaN it cannot be
        serialised at all: the refusal itself failed with a 500.
        """
        detail = [
            {"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]}
            for item in error.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detail": detail}
        )

    @app.exception_handler(ValueError)
    async def value_error(request: Request, error: ValueError):  # pragma: no cover
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(error)) from error

    return app


app = create_app()
