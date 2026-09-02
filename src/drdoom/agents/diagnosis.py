"""Diagnosis: explain the incident, grounded in documentation that exists.

The model is given retrieved passages and asked to reason within them. Retrieval is
driven by the symptoms and the predicted cause, not by a filter that pre-selects the
document, so the passages are found rather than looked up.

When the provider is unreachable the agent degrades instead of failing: the retrieved
passages are returned with a diagnosis that says plainly no model was consulted. An
on-call engineer with the right three documents and no summary is better served than one
with a stack trace, and the alternative -- inventing a summary -- is the failure mode this
whole project is arranged against.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from drdoom.agents.schemas import Citation, Diagnosis
from drdoom.llm.base import Completion, LLMProvider, LLMUnavailableError, Message
from drdoom.llm.structured import generate_structured
from drdoom.rag.index import Hit, Retriever
from drdoom.rag.rerank import NoReranker, Reranker

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are assisting an on-call engineer with a live production incident. "
    "Answer only from the documentation excerpts provided. If they do not explain the "
    "symptoms, say so in the summary rather than speculating."
)

SHORTLIST_DEPTH = 30
CONTEXT_PASSAGES = 5


@dataclass(frozen=True)
class DiagnosisResult:
    diagnosis: Diagnosis
    citations: list[Citation]
    degraded: bool = False
    completions: list[Completion] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(completion.total_tokens for completion in self.completions)


def build_query(root_cause: str | None, symptoms: str) -> str:
    """Compose the retrieval query from what triage established."""
    parts = [symptoms.strip()]
    if root_cause:
        parts.append(root_cause.replace("_", " "))
    return " ".join(part for part in parts if part)


def format_passages(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{position}] {hit.chunk.citation}\n{hit.chunk.text}"
        for position, hit in enumerate(hits, start=1)
    )


def to_citations(hits: list[Hit]) -> list[Citation]:
    return [
        Citation(
            chunk_id=hit.chunk.chunk_id,
            doc_id=hit.chunk.doc_id,
            title=hit.chunk.citation,
            url=hit.chunk.url,
            licence=hit.chunk.licence,
        )
        for hit in hits
    ]


class DiagnosisAgent:
    """Retrieves supporting documentation and asks the model to explain the incident."""

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

    def retrieve(self, query: str) -> list[Hit]:
        shortlist = self.retriever.search(query, SHORTLIST_DEPTH)
        return self.reranker.rerank(query, shortlist, self.passages)

    def run(self, symptoms: str, root_cause: str | None = None) -> DiagnosisResult:
        query = build_query(root_cause, symptoms)
        hits = self.retrieve(query)
        citations = to_citations(hits)

        prompt = (
            f"Observed symptoms:\n{symptoms}\n\n"
            f"Detected cause from the classifier: {root_cause or 'not determined'}\n\n"
            f"Documentation excerpts:\n{format_passages(hits)}\n\n"
            "Diagnose the incident as JSON with keys summary, likely_cause, confidence "
            "(low, medium or high) and next_action."
        )

        try:
            diagnosis, completions = generate_structured(
                self.provider,
                [Message(role="user", content=prompt)],
                Diagnosis,
                system=SYSTEM,
                max_tokens=800,
            )
        except LLMUnavailableError as error:
            logger.warning("provider unavailable, degrading to retrieval only: %s", error)
            return DiagnosisResult(
                diagnosis=self._degraded(hits, root_cause), citations=citations, degraded=True
            )

        return DiagnosisResult(
            diagnosis=diagnosis, citations=citations, degraded=False, completions=completions
        )

    def _degraded(self, hits: list[Hit], root_cause: str | None) -> Diagnosis:
        """A truthful placeholder: what was retrieved, and that nothing summarised it."""
        leading = hits[0].chunk.citation if hits else "no matching documentation"
        return Diagnosis(
            summary=(
                "No model was available, so this incident has not been summarised. "
                f"The most relevant documentation retrieved was: {leading}. "
                "Read the cited passages directly."
            ),
            likely_cause=root_cause or "not determined",
            confidence="low",
            next_action="Review the cited documentation and diagnose manually.",
        )
