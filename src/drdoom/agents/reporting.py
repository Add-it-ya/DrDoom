"""Reporting: write the record of what happened and what was decided.

The postmortem is generated last, once a decision exists, so it can state what was
actually done rather than what was proposed. Rejections are recorded as rejections; a
report that reads the same whether or not a human approved the action would make the
approval gate decorative.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from drdoom.agents.schemas import Citation, Diagnosis, Postmortem, RemediationPlan
from drdoom.llm.base import Completion, LLMProvider, LLMUnavailableError, Message
from drdoom.llm.structured import generate_structured

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are writing a post-incident report for an engineering team. Be concise and "
    "factual. Describe only what the incident record states was done."
)


@dataclass(frozen=True)
class ReportResult:
    postmortem: Postmortem
    markdown: str
    degraded: bool = False
    completions: list[Completion] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(completion.total_tokens for completion in self.completions)


class ReportingAgent:
    """Turns the incident record into a written postmortem."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def run(
        self,
        diagnosis: Diagnosis,
        plan: RemediationPlan,
        decision: str,
        executed: str | None = None,
        citations: list[Citation] | None = None,
    ) -> ReportResult:
        prompt = (
            "Incident record:\n"
            f"- Diagnosis: {diagnosis.summary}\n"
            f"- Likely cause: {diagnosis.likely_cause} (confidence {diagnosis.confidence})\n"
            f"- Proposed action: {plan.immediate_action}\n"
            f"- Risk level: {plan.risk_level}\n"
            f"- Approval required: {plan.requires_approval}\n"
            f"- Decision: {decision}\n"
            f"- Executed: {executed or 'nothing was executed'}\n"
            f"- Short-term fix: {plan.short_term_fix}\n"
            f"- Long-term fix: {plan.long_term_fix}\n\n"
            "Write the postmortem as JSON with keys title, summary, what_happened, "
            "root_cause, action_taken and prevention."
        )

        try:
            postmortem, completions = generate_structured(
                self.provider,
                [Message(role="user", content=prompt)],
                Postmortem,
                system=SYSTEM,
                max_tokens=1200,
            )
        except LLMUnavailableError as error:
            logger.warning("provider unavailable, writing the record without prose: %s", error)
            postmortem = self._degraded(diagnosis, plan, decision, executed)
            return ReportResult(
                postmortem=postmortem, markdown=self._render(postmortem, citations), degraded=True
            )

        return ReportResult(
            postmortem=postmortem,
            markdown=self._render(postmortem, citations),
            completions=completions,
        )

    def _degraded(
        self, diagnosis: Diagnosis, plan: RemediationPlan, decision: str, executed: str | None
    ) -> Postmortem:
        """The facts, unembellished, when no model was available to write them up."""
        return Postmortem(
            title=f"Incident: {diagnosis.likely_cause}",
            summary="Generated without a model. The recorded facts follow.",
            what_happened=diagnosis.summary,
            root_cause=f"{diagnosis.likely_cause} (confidence {diagnosis.confidence})",
            action_taken=(
                f"Proposed: {plan.immediate_action}. Risk {plan.risk_level}. "
                f"Decision: {decision}. Executed: {executed or 'nothing'}."
            ),
            prevention=plan.long_term_fix,
        )

    def _render(self, postmortem: Postmortem, citations: list[Citation] | None) -> str:
        markdown = postmortem.to_markdown()
        if not citations:
            return markdown
        lines = ["## Sources", ""]
        lines += [
            f"- {citation.title}" + (f" ({citation.url})" if citation.url else "")
            for citation in citations
        ]
        return markdown + "\n" + "\n".join(lines) + "\n"
