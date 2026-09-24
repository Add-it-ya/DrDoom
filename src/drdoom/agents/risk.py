"""Risk assessment: a second opinion the plan's author cannot talk down.

The remediation agent writes a plan and rates its blast radius, and until now that rating
alone decided whether a human saw the plan. An author grading its own work is the weakest
reviewer available: models tend to rate their own output favourably, and one call that is
steered by a poisoned document steers the plan and its rating together.

So three ratings are taken, and the most cautious one stands:

- **The policy floor**, a fixed minimum per catalogue action, written in code. A node
  drain is high risk whatever anyone says about it. See ``ActionSpec.risk_floor``.
- **This assessor**, a separate call with a fresh context and an adversarial brief: find
  the way this action could hurt the service. It is shown the incident, the plan and the
  exact command, and it is not shown the author's rating, so it cannot anchor on it. Nor
  is it told which rating lets a plan through without a human.
- **The author's own rating**, kept, but only able to raise the result.

None of the three can lower what another decided. An assessor that cannot answer, or
answers with something unusable, counts as high: an unassessed plan is never treated as a
safe one.

The assessor reads passage titles rather than passage text. The plan and the command are
what it judges, and every retrieved paragraph handed to it is another place an injected
instruction could hide. The titles still tell it which runbooks the plan came from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from drdoom.agents.schemas import Citation, Diagnosis, RemediationPlan, RiskAssessment, RiskLevel
from drdoom.llm.base import (
    Completion,
    Failure,
    LLMInvalidOutputError,
    LLMProvider,
    LLMUnavailableError,
    Message,
    failure_kind,
    spent_on,
)
from drdoom.llm.structured import generate_structured

logger = logging.getLogger(__name__)

ORDER: dict[RiskLevel, int] = {"low": 0, "medium": 1, "high": 2}

SYSTEM = (
    "You review remediation plans for live production incidents before anything runs. "
    "Your job is to find how the proposed action could hurt the service: take it down, "
    "lose data, disturb live traffic, or make the incident worse if the diagnosis is "
    "wrong. Judge the command that would actually run, not the intent described around "
    "it. Rate by the worst realistic outcome: low means it cannot disturb live traffic "
    "even if the diagnosis is wrong; medium means it can disturb live traffic or is hard "
    "to reverse; high means it can cause an outage or lose data. When unsure between "
    "two levels, choose the higher one."
)


def highest(*levels: RiskLevel) -> RiskLevel:
    """The most cautious of the ratings given."""
    return max(levels, key=ORDER.__getitem__)


@dataclass(frozen=True)
class RiskResult:
    """What the assessor concluded, or why it could not conclude anything."""

    assessment: RiskAssessment | None
    completions: list[Completion] = field(default_factory=list)
    failure: Failure | None = None

    @property
    def level(self) -> RiskLevel:
        """The assessor's rating, with no answer read as high."""
        return self.assessment.risk_level if self.assessment else "high"

    @property
    def degraded(self) -> bool:
        return self.assessment is None


class RiskAssessor:
    """Rates a plan without being told how its author rated it."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def prompt(
        self,
        triage: dict[str, Any],
        diagnosis: Diagnosis,
        plan: RemediationPlan,
        command: str | None,
        citations: list[Citation],
    ) -> str:
        """Everything the plan rests on, minus the author's rating.

        Built field by field from the plan rather than from a dump of it, so that adding a
        field to the plan cannot quietly show the assessor the rating it must not see.
        """
        sources = "\n".join(f"- {citation.title}" for citation in citations) or "- none"
        return (
            "Incident:\n"
            f"- Detector score {triage.get('score')} against threshold "
            f"{triage.get('threshold')}\n"
            f"- Classified cause: {triage.get('root_cause') or 'not determined'}\n\n"
            "Diagnosis:\n"
            f"- Summary: {diagnosis.summary}\n"
            f"- Likely cause: {diagnosis.likely_cause} (confidence {diagnosis.confidence})\n\n"
            "Proposed remediation:\n"
            f"- Immediate action: {plan.immediate_action}\n"
            f"- Command that would run: {command or 'none; nothing would be executed'}\n"
            f"- Short-term fix: {plan.short_term_fix}\n"
            f"- Rollback: {plan.rollback}\n\n"
            f"Documentation the plan was written from:\n{sources}\n\n"
            "Assess the risk as JSON with keys risk_level (low, medium or high), worst_case "
            "(one sentence: the worst realistic outcome of running this) and reasons (up to "
            "three short strings)."
        )

    def run(
        self,
        triage: dict[str, Any],
        diagnosis: Diagnosis,
        plan: RemediationPlan,
        command: str | None,
        citations: list[Citation],
    ) -> RiskResult:
        prompt = self.prompt(triage, diagnosis, plan, command, citations)
        try:
            assessment, completions = generate_structured(
                self.provider,
                [Message(role="user", content=prompt)],
                RiskAssessment,
                system=SYSTEM,
                max_tokens=500,
            )
        except (LLMUnavailableError, LLMInvalidOutputError) as error:
            logger.warning("no usable risk assessment, treating the plan as high risk: %s", error)
            return RiskResult(None, completions=spent_on(error), failure=failure_kind(error))

        logger.info("independent assessment: %s", assessment.risk_level)
        return RiskResult(assessment, completions=completions)
