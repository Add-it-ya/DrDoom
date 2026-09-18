"""Remediation: propose a fix, and classify how dangerous it is.

The plan is retrieved against remediation-shaped language so the passages are about
fixing rather than describing. The model proposes an action and rates its blast radius;
whether that rating demands a human is decided by the schema, not here and not by the
model. See ``RemediationPlan.requires_approval``.

Nothing in this module executes anything. Proposing and doing are separated so that the
approval gate has something real to stand between.

When the provider is unreachable the agent degrades rather than failing the whole
investigation, and the plan it falls back to is built to be harmless: it proposes
nothing, it is rated high risk so a human still has to see it, and its action matches
nothing the executor knows how to run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from drdoom.agents.diagnosis import SHORTLIST_DEPTH, format_passages, to_citations
from drdoom.agents.schemas import Citation, RemediationPlan
from drdoom.executor import CATALOGUE
from drdoom.llm.base import Completion, LLMProvider, LLMUnavailableError, Message
from drdoom.llm.structured import generate_structured
from drdoom.rag.index import Hit, Retriever
from drdoom.rag.rerank import NoReranker, Reranker

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are proposing a remediation for a live production incident. Ground every step "
    "in the documentation excerpts provided. Rate risk by blast radius: low is safe to "
    "apply unattended, medium can disturb live traffic, high can cause an outage or "
    "lose data."
)

CONTEXT_PASSAGES = 4
ACTION_MENU = "\n".join(f"- {spec.kind}: {spec.description}" for spec in CATALOGUE)


@dataclass(frozen=True)
class RemediationResult:
    plan: RemediationPlan
    citations: list[Citation] = field(default_factory=list)
    degraded: bool = False
    completions: list[Completion] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(completion.total_tokens for completion in self.completions)


class RemediationAgent:
    """Turns a diagnosis into a structured, risk-rated plan."""

    def __init__(
        self,
        retriever: Retriever,
        provider: LLMProvider,
        reranker: Reranker | None = None,
        passages: int = CONTEXT_PASSAGES,
    ) -> None:
        self.retriever = retriever
        self.provider = provider
        self.reranker = reranker or NoReranker()
        self.passages = passages

    def retrieve(self, root_cause: str | None, diagnosis_summary: str) -> list[Hit]:
        cause = (root_cause or "").replace("_", " ")
        query = f"how to fix remediate resolve {cause} {diagnosis_summary}".strip()
        shortlist = self.retriever.search(query, SHORTLIST_DEPTH)
        return self.reranker.rerank(query, shortlist, self.passages)

    def run(self, diagnosis_summary: str, root_cause: str | None = None) -> RemediationResult:
        hits = self.retrieve(root_cause, diagnosis_summary)
        prompt = (
            f"Diagnosis:\n{diagnosis_summary}\n\n"
            f"Detected cause: {root_cause or 'not determined'}\n\n"
            f"Documentation excerpts:\n{format_passages(hits)}\n\n"
            "Return a remediation plan as JSON with keys immediate_action, risk_level "
            "(low, medium or high), short_term_fix, long_term_fix, rollback and action.\n\n"
            "action is what would actually run once a human approves. It must be exactly one "
            "of these, and only if it is the step immediate_action describes; otherwise null:\n"
            f"{ACTION_MENU}"
        )

        try:
            plan, completions = generate_structured(
                self.provider,
                [Message(role="user", content=prompt)],
                RemediationPlan,
                system=SYSTEM,
                max_tokens=800,
            )
        except LLMUnavailableError as error:
            logger.warning("provider unavailable, holding an empty plan for a human: %s", error)
            return RemediationResult(
                plan=self._degraded(hits), citations=to_citations(hits), degraded=True
            )

        logger.info("plan rated %s, approval required: %s", plan.risk_level, plan.requires_approval)
        return RemediationResult(plan=plan, citations=to_citations(hits), completions=completions)

    def _degraded(self, hits: list[Hit]) -> RemediationPlan:
        """A plan that proposes nothing, rated so that a human has to look at it.

        Its risk is unknown, and unknown is treated as high rather than low: the costly
        mistake is letting an unassessed action through unattended. The action names nothing
        in the executor's catalogue, so approving the plan runs nothing either.
        """
        leading = hits[0].chunk.citation if hits else "no matching documentation"
        return RemediationPlan(
            immediate_action="None proposed: no model was available to write a plan",
            risk_level="high",
            short_term_fix=f"Work from the most relevant documentation retrieved: {leading}.",
            long_term_fix="Decide the fix manually once the cause is confirmed.",
            rollback="Nothing was applied, so there is nothing to undo.",
        )
