"""Appends to an audit log, run in its own process by the concurrency test.

Not named ``test_*`` so pytest does not collect it.
"""

from __future__ import annotations

from pathlib import Path

from drdoom.audit import AuditLog


def append_many(path: str, worker: int, count: int) -> None:
    log = AuditLog(Path(path))
    for n in range(count):
        log.record(
            incident_id=f"w{worker}-{n}",
            principal="aditya",
            decision="approved_by_human",
            risk_level="low",
            immediate_action="restart the pods",
            plan_hash="0" * 64,
            executed=True,
            execution="kubectl rollout restart deployment/api",
        )
