"""One half of an investigation, run as its own process.

Used by the durability test: one invocation starts an investigation and stops at the
approval gate, a second invocation in a fresh interpreter resumes it. Nothing is shared
between them but the sqlite file, which is the point.

Not named ``test_*`` so pytest does not collect it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from drdoom.agents.diagnosis import DiagnosisAgent
from drdoom.agents.graph import Investigator, open_checkpointer
from drdoom.agents.remediation import RemediationAgent
from drdoom.agents.reporting import ReportingAgent
from drdoom.agents.triage import TriageAgent, window_to_series
from drdoom.audit import AuditLog
from drdoom.data.windows import Scaler
from drdoom.detect.baselines import WindowSpread
from drdoom.llm.stub import StubProvider
from drdoom.rag.corpus import Document
from drdoom.rag.index import BM25Index
from drdoom.rag.ingest import chunk_all

DIAGNOSIS = json.dumps(
    {
        "summary": "Memory grew steadily until the container was killed.",
        "likely_cause": "memory leak",
        "confidence": "high",
        "next_action": "Restart the pods.",
    }
)
PLAN = json.dumps(
    {
        "immediate_action": "Rolling restart of the affected pods",
        "risk_level": "high",
        "short_term_fix": "Set a memory limit",
        "long_term_fix": "Fix the cache eviction policy",
        "rollback": "Scale the previous replica set back up",
    }
)
POSTMORTEM = json.dumps(
    {
        "title": "Memory leak in the api service",
        "summary": "A leak exhausted container memory.",
        "what_happened": "Memory grew for forty minutes.",
        "root_cause": "Unbounded cache.",
        "action_taken": "Rolling restart after approval.",
        "prevention": "Add an eviction policy.",
    }
)

FEATURES = ["a", "b"]
THRESHOLD = 5.0


def build(checkpointer, audit_path: Path) -> Investigator:
    quiet = np.random.default_rng(0).normal(50, 1, size=(60, 2)).astype(np.float32)
    series, index = window_to_series(quiet, FEATURES)
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
        TriageAgent(detector, threshold=THRESHOLD, feature_names=FEATURES),
        DiagnosisAgent(retriever, StubProvider(default=DIAGNOSIS)),
        RemediationAgent(retriever, StubProvider(default=PLAN)),
        ReportingAgent(StubProvider(default=POSTMORTEM)),
        checkpointer,
        audit=AuditLog(audit_path),
    )


def disturbed_window() -> np.ndarray:
    window = np.random.default_rng(1).normal(50, 1, size=(60, 2)).astype(np.float32)
    window[30:, 0] += 60.0
    return window


def main() -> None:
    action, database, thread_id = sys.argv[1], Path(sys.argv[2]), sys.argv[3]

    with open_checkpointer(database) as checkpointer:
        investigator = build(checkpointer, database.with_name("audit.jsonl"))
        if action == "start":
            outcome = investigator.start(
                disturbed_window(), "latency and memory climbing", thread_id
            )
        elif action == "resume":
            outcome = investigator.resume(thread_id, approved=sys.argv[4] == "true")
        else:
            raise SystemExit(f"unknown action {action!r}")

        print(
            json.dumps(
                {
                    "status": outcome.status,
                    "pending": outcome.pending,
                    "decision": outcome.state.get("decision"),
                    "report": outcome.report,
                    "tokens": outcome.tokens,
                }
            )
        )


if __name__ == "__main__":
    main()
