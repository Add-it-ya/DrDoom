"""Execution, and the proof of approval it demands first.

A predecessor project claimed an enforced human-approval gate, and approving or rejecting
produced the same outcome: a status string changed and a report was written either way.
Nothing executed, so nothing was gated. This module is what the gate stands in front of.

Two properties make the claim testable rather than decorative.

**Execution requires a token that names the plan it approves.** An ``ApprovalToken``
carries the hash of the exact plan a human saw. Executing a different plan with that token
fails, so approving a rolling restart cannot be turned into approval for deleting a
volume. This is the substitution attack the gate exists to prevent, and it is a type error
here rather than a review comment.

**Only recognised actions run.** The model writes its action as prose, and prose is not a
command. Actions are matched against a small catalogue; anything unrecognised is refused
and reported as refused. A system arranged around not trusting the model should not end by
executing whatever sentence it produced.

Nothing here touches a real cluster. Commands are rendered and returned, never run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from drdoom.agents.schemas import RemediationPlan

logger = logging.getLogger(__name__)

DEFAULT_TARGET = "affected-deployment"


class NotApprovedError(Exception):
    """Execution was attempted without a valid approval for that exact plan."""


def plan_hash(plan: RemediationPlan) -> str:
    """A stable fingerprint of a plan, including its derived approval requirement.

    Canonical json with sorted keys, so the same plan hashes identically across
    processes and the hash can be compared to what a human was shown.
    """
    payload = json.dumps(plan.model_dump(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ApprovalToken:
    """Evidence that a named principal approved one specific plan."""

    plan_hash: str
    principal: str
    decided_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @classmethod
    def issue(cls, plan: RemediationPlan, principal: str) -> ApprovalToken:
        return cls(plan_hash=plan_hash(plan), principal=principal)

    def authorises(self, plan: RemediationPlan) -> bool:
        return self.plan_hash == plan_hash(plan)


@dataclass(frozen=True)
class ActionSpec:
    """One action the system knows how to perform."""

    kind: str
    keywords: tuple[str, ...]
    template: str
    description: str


CATALOGUE: tuple[ActionSpec, ...] = (
    ActionSpec(
        kind="rollout_restart",
        keywords=("rolling restart", "rollout restart", "restart the pods", "recreate the pods"),
        template="kubectl rollout restart deployment/{target}",
        description="Recreate the pods of a deployment without changing its image",
    ),
    ActionSpec(
        kind="rollout_undo",
        keywords=("roll back", "rollback", "previous version", "undo the deploy"),
        template="kubectl rollout undo deployment/{target}",
        description="Return a deployment to its previous revision",
    ),
    ActionSpec(
        kind="scale_out",
        keywords=("scale up", "scale out", "add replicas", "increase replicas"),
        template="kubectl scale deployment/{target} --replicas=6",
        description="Raise the replica count of a deployment",
    ),
    ActionSpec(
        kind="set_memory_limit",
        keywords=("memory limit", "limit memory", "memory cap"),
        template="kubectl set resources deployment/{target} --limits=memory=2Gi",
        description="Apply a memory limit to a deployment",
    ),
    ActionSpec(
        kind="cordon_node",
        keywords=("cordon", "drain the node", "drain node"),
        template="kubectl drain node/{target} --ignore-daemonsets",
        description="Move workloads off a node for maintenance",
    ),
)


@dataclass(frozen=True)
class ExecutionResult:
    """What execution did, or why it did nothing."""

    executed: bool
    kind: str
    command: str | None = None
    detail: str = ""
    dry_run: bool = True

    def as_dict(self) -> dict:
        return {
            "executed": self.executed,
            "kind": self.kind,
            "command": self.command,
            "detail": self.detail,
            "dry_run": self.dry_run,
        }

    @property
    def summary(self) -> str:
        if not self.executed:
            return f"nothing was executed ({self.detail})"
        return f"{self.command} (dry run)"


def match_action(text: str) -> ActionSpec | None:
    """Find the catalogue entry an action describes, if any."""
    lowered = re.sub(r"\s+", " ", text.lower())
    return next(
        (spec for spec in CATALOGUE if any(keyword in lowered for keyword in spec.keywords)),
        None,
    )


class DryRunExecutor:
    """Renders the command an approved plan would run, and stops there."""

    def __init__(self, target: str = DEFAULT_TARGET) -> None:
        self.target = target

    def execute(self, plan: RemediationPlan, token: ApprovalToken | None) -> ExecutionResult:
        """Run an approved plan, or refuse and say why.

        Refusal is a normal outcome with a reason attached, except for a token that does
        not match the plan, which is an error rather than a decision.
        """
        if token is None:
            logger.info("execution refused: no approval token")
            return ExecutionResult(executed=False, kind="refused", detail="no approval was given")

        if not token.authorises(plan):
            raise NotApprovedError(
                "approval token does not match this plan; "
                f"approved {token.plan_hash[:12]}, asked to run {plan_hash(plan)[:12]}"
            )

        spec = match_action(plan.immediate_action)
        if spec is None:
            logger.warning("execution refused: unrecognised action %r", plan.immediate_action)
            return ExecutionResult(
                executed=False,
                kind="unrecognised",
                detail=f"no known action matches {plan.immediate_action!r}",
            )

        command = spec.template.format(target=self.target)
        logger.info("executing (dry run) %s for %s", spec.kind, token.principal)
        return ExecutionResult(
            executed=True, kind=spec.kind, command=command, detail=spec.description
        )
