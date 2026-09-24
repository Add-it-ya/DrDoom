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

**Only a named catalogue action runs.** The model writes its action as prose for the
human, and prose is not a command. What runs is read from ``RemediationPlan.action``, a
field that can only name an entry in a small catalogue or nothing. The prose is never
searched for keywords: a substring match reads "never roll back" as a rollback. A plan that
names no action is refused and reported as refused. A system arranged around not trusting
the model should not end by executing its own reading of whatever sentence it produced.

Nothing here touches a real cluster. Commands are rendered and returned, never run.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from drdoom.agents.schemas import RemediationPlan, RiskLevel

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
    """One action the system knows how to perform, and the least risk it can carry.

    ``risk_floor`` is policy, not a prediction. No rating from a model can take a plan
    that runs this action below it, so the floor is the one rating a poisoned document
    or a flattering self-assessment cannot move.
    """

    kind: str
    template: str
    description: str
    risk_floor: RiskLevel


CATALOGUE: tuple[ActionSpec, ...] = (
    ActionSpec(
        kind="rollout_restart",
        template="kubectl rollout restart deployment/{target}",
        description="Recreate the pods of a deployment without changing its image",
        # Every pod is replaced; a service with one replica or a slow start drops traffic.
        risk_floor="medium",
    ),
    ActionSpec(
        kind="rollout_undo",
        template="kubectl rollout undo deployment/{target}",
        description="Return a deployment to its previous revision",
        # The previous revision may not match today's schema, config or dependencies.
        risk_floor="medium",
    ),
    ActionSpec(
        kind="scale_out",
        template="kubectl scale deployment/{target} --replicas=6",
        description="Raise the replica count of a deployment",
        # Adds capacity and removes none; undone by scaling back.
        risk_floor="low",
    ),
    ActionSpec(
        kind="set_memory_limit",
        template="kubectl set resources deployment/{target} --limits=memory=2Gi",
        description="Apply a memory limit to a deployment",
        # A limit set too low turns a leak into a crash loop, and it restarts every pod.
        risk_floor="medium",
    ),
    ActionSpec(
        kind="cordon_node",
        template="kubectl drain node/{target} --ignore-daemonsets",
        description="Move workloads off a node for maintenance",
        # Evicts everything on the node at once, including workloads unrelated to the incident.
        risk_floor="high",
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


def action_spec(kind: str | None) -> ActionSpec | None:
    """The catalogue entry a plan names, if it names one."""
    return next((spec for spec in CATALOGUE if spec.kind == kind), None) if kind else None


def risk_floor(kind: str | None) -> RiskLevel:
    """The least risk a plan naming this action can be rated.

    A plan that names nothing runs nothing, so its floor is low; the other ratings still
    decide whether a human reads it.
    """
    spec = action_spec(kind)
    return spec.risk_floor if spec else "low"


class DryRunExecutor:
    """Renders the command an approved plan would run, and stops there."""

    def __init__(self, target: str = DEFAULT_TARGET) -> None:
        self.target = target

    def preview(self, plan: RemediationPlan) -> str | None:
        """The exact command approving this plan would run, or None if it runs nothing.

        Shown at the gate so a human approves a command, not a sentence. It is the same
        rendering ``execute`` uses, so the two cannot disagree.
        """
        spec = action_spec(plan.action)
        return spec.template.format(target=self.target) if spec else None

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

        spec = action_spec(plan.action)
        if spec is None:
            logger.warning("execution refused: plan names no catalogue action (%r)", plan.action)
            return ExecutionResult(
                executed=False,
                kind="unrecognised",
                detail="the plan names no catalogue action, so there is nothing to run",
            )

        command = self.preview(plan)
        logger.info("executing (dry run) %s for %s", spec.kind, token.principal)
        return ExecutionResult(
            executed=True, kind=spec.kind, command=command, detail=spec.description
        )
