"""The shapes the agents exchange, and the one rule that is never delegated.

``RemediationPlan.requires_approval`` is a computed field, not an input. The model is
never asked for it, and a value supplied anyway is ignored. This is the single most
important line in the project: a system whose safety argument is "a human approves risky
actions" cannot let the thing being supervised decide what counts as risky. A predecessor
project observed exactly that failure -- a plan returned with ``risk_level`` of high and
``requires_approval`` of false -- and patched it after parsing. Here the rule lives in the
type, so every construction path gets it: model output, direct construction,
deserialisation from disk.

Everything else is ordinary validation, which exists because syntactically valid JSON is
not a usable plan. A risk level has three possible values, not any sentence the model
feels like writing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

RiskLevel = Literal["low", "medium", "high"]
Confidence = Literal["low", "medium", "high"]

APPROVAL_REQUIRED_AT: frozenset[str] = frozenset({"medium", "high"})


class Citation(BaseModel):
    """Where a retrieved passage came from, so a claim can be traced."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    title: str
    url: str = ""
    licence: str = ""


class Diagnosis(BaseModel):
    """What the model believes happened, grounded in retrieved documentation."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    summary: str = Field(description="Two or three sentences an on-call engineer can act on")
    likely_cause: str = Field(description="The single most probable cause")
    confidence: Confidence = Field(description="How well the evidence supports the cause")
    next_action: str = Field(description="The immediate next step to take or verify")


class RemediationPlan(BaseModel):
    """A proposed fix. Whether it needs a human is decided here, not by the model."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    immediate_action: str = Field(description="One sentence, the single most urgent action")
    risk_level: RiskLevel = Field(description="Blast radius if this action goes wrong")
    short_term_fix: str = Field(description="What stabilises the service today")
    long_term_fix: str = Field(description="What stops it recurring")
    rollback: str = Field(description="How to undo the immediate action if it makes things worse")

    @computed_field
    @property
    def requires_approval(self) -> bool:
        """Derived from risk level, never supplied.

        Medium counts as requiring approval, not just high. The costly mistake is
        treating an under-classified action as safe, and the cost of an unnecessary
        confirmation is one click.
        """
        return self.risk_level in APPROVAL_REQUIRED_AT


class Postmortem(BaseModel):
    """The written record of an incident and what was decided about it."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    title: str
    summary: str
    what_happened: str
    root_cause: str
    action_taken: str
    prevention: str

    def to_markdown(self) -> str:
        sections = (
            ("Summary", self.summary),
            ("What happened", self.what_happened),
            ("Root cause", self.root_cause),
            ("Action taken", self.action_taken),
            ("Prevention", self.prevention),
        )
        body = "\n\n".join(f"## {heading}\n\n{text}" for heading, text in sections)
        return f"# {self.title}\n\n{body}\n"
