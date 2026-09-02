"""The record of who decided what, and whether it has been altered since.

Every change-control process is built around an artefact like this, and a predecessor
project had none: approving and rejecting left no trace beyond a status string on an
object that was then deleted. What follows is the smallest thing that answers the
questions an incident review actually asks -- who approved this, what exactly did they
approve, when, and did anything run.

The file is append-only in the ordinary sense (opened for append, never rewritten), and
tamper-evident in a stronger one: each entry carries the hash of the entry before it, so
editing or removing any earlier line breaks the chain from that point on. That does not
make the log unalterable, which no local file can be. It makes alteration detectable,
which is what a review needs.

The hash of the plan is recorded, not just its text. That is what lets a review confirm
the plan that ran is the plan that was shown to the approver.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from drdoom.config import get_settings

logger = logging.getLogger(__name__)

GENESIS = "0" * 64


@dataclass(frozen=True)
class AuditEntry:
    """One decision, and its place in the chain."""

    timestamp: str
    incident_id: str
    principal: str
    decision: str
    risk_level: str
    immediate_action: str
    plan_hash: str
    executed: bool
    execution: str
    previous_hash: str
    entry_hash: str

    def payload(self) -> dict:
        """Everything the entry hash covers, which is everything but the hash itself."""
        record = asdict(self)
        record.pop("entry_hash")
        return record


def compute_entry_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AuditLog:
    """An append-only decision log backed by one json-lines file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or get_settings().project_root / "state" / "audit.jsonl"

    def entries(self) -> list[AuditEntry]:
        if not self.path.is_file():
            return []
        return [
            AuditEntry(**json.loads(line))
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def last_hash(self) -> str:
        entries = self.entries()
        return entries[-1].entry_hash if entries else GENESIS

    def record(
        self,
        *,
        incident_id: str,
        principal: str,
        decision: str,
        risk_level: str,
        immediate_action: str,
        plan_hash: str,
        executed: bool,
        execution: str,
    ) -> AuditEntry:
        """Append one decision. The file is only ever opened for append."""
        payload = {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "incident_id": incident_id,
            "principal": principal,
            "decision": decision,
            "risk_level": risk_level,
            "immediate_action": immediate_action,
            "plan_hash": plan_hash,
            "executed": executed,
            "execution": execution,
            "previous_hash": self.last_hash(),
        }
        entry = AuditEntry(**payload, entry_hash=compute_entry_hash(payload))

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(entry), sort_keys=True) + "\n")

        logger.info(
            "audit: %s %s by %s (plan %s)",
            incident_id,
            decision,
            principal,
            plan_hash[:12],
        )
        return entry

    def verify(self) -> tuple[bool, str]:
        """Walk the chain and report the first break, if there is one."""
        previous = GENESIS
        for position, entry in enumerate(self.entries()):
            if entry.previous_hash != previous:
                return False, f"entry {position} does not follow the one before it"
            if compute_entry_hash(entry.payload()) != entry.entry_hash:
                return False, f"entry {position} has been altered"
            previous = entry.entry_hash
        return True, "chain intact"

    def for_incident(self, incident_id: str) -> list[AuditEntry]:
        return [entry for entry in self.entries() if entry.incident_id == incident_id]
