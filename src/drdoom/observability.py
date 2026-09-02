"""Structured logging and stage timing, keyed by incident.

A project about operational visibility that logs unstructured prose to stdout is not
observable, and the irony is worth avoiding. Two things make the difference:

**Every line carries the incident it belongs to.** Without that, a concurrent run
interleaves with another and neither can be reconstructed afterwards. The incident id is
set once when a run starts and travels through a context variable, so callers do not have
to thread it through every function.

**Every stage is timed.** Where time goes in an investigation is the first question asked
when one is slow, and it is unanswerable from a log that only records what happened.

Deliberately not OpenTelemetry. A collector, an exporter and a backend is a lot of moving
parts for a service that currently answers one question, and the structured records here
carry the same fields a span would. The seam to swap in a real tracer is one function.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

current_incident: ContextVar[str] = ContextVar("current_incident", default="-")

# Attributes the logging module puts on every record. Anything else a caller attached
# through `extra` is ours and belongs in the output.
RESERVED = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
    }
)  # fmt: skip


class JsonFormatter(logging.Formatter):
    """One json object per line, with the incident id on every record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "incident": current_incident.get(),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", structured: bool = True) -> None:
    """Install the log format for the process. Plain text stays available for a terminal."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if structured else logging.Formatter("%(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    # These two are healthy but noisy at info level.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


@contextmanager
def incident_context(incident_id: str) -> Iterator[None]:
    """Tag every log line emitted inside this block with an incident."""
    token = current_incident.set(incident_id)
    try:
        yield
    finally:
        current_incident.reset(token)


@dataclass
class Timings:
    """How long each stage of an investigation took."""

    stages: dict[str, float] = field(default_factory=dict)

    def record(self, stage: str, milliseconds: float) -> None:
        self.stages[stage] = round(milliseconds, 1)

    @property
    def total_ms(self) -> float:
        return round(sum(self.stages.values()), 1)

    def as_dict(self) -> dict[str, Any]:
        return {"stages_ms": dict(self.stages), "total_ms": self.total_ms}


@contextmanager
def timed(stage: str, timings: Timings | None = None) -> Iterator[None]:
    """Time a stage, log how long it took, and record it if a collector was given."""
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - started) * 1000
        if timings is not None:
            timings.record(stage, elapsed)
        logging.getLogger("drdoom.timing").info(
            "stage complete", extra={"stage": stage, "duration_ms": round(elapsed, 1)}
        )


class Counters:
    """Process-wide counters and stage latencies for the metrics endpoint."""

    def __init__(self) -> None:
        self.events: dict[str, int] = {}
        self.latencies: dict[str, list[float]] = {}

    def increment(self, name: str, amount: int = 1) -> None:
        self.events[name] = self.events.get(name, 0) + amount

    def observe(self, stage: str, milliseconds: float) -> None:
        self.latencies.setdefault(stage, []).append(milliseconds)

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float:
        ordered = sorted(values)
        index = min(int(fraction * len(ordered)), len(ordered) - 1)
        return round(ordered[index], 1)

    def snapshot(self) -> dict[str, Any]:
        return {
            "events": dict(self.events),
            "stages": {
                stage: {
                    "count": len(values),
                    "p50_ms": self._percentile(values, 0.50),
                    "p95_ms": self._percentile(values, 0.95),
                }
                for stage, values in sorted(self.latencies.items())
                if values
            },
        }
