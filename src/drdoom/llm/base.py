"""The provider interface the agents talk to.

Agents depend on this and never on a vendor SDK, so swapping providers is a
configuration change rather than an edit to four agent files. The interface is
deliberately narrow: one method, one return type, no streaming, no tool loop. Anything a
particular vendor does better is either mapped into this shape or not used.

Failures are separated by whether retrying could help. ``LLMUnavailableError`` means the
provider could not be reached or refused the request, which the caller answers by
degrading to the retrieved documentation. ``LLMInvalidOutputError`` means the provider
answered but the answer did not fit the schema, which the caller answers by repairing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant"]

DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT_SECONDS = 45.0


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class Completion:
    """One model response, with the accounting needed to report cost per incident."""

    text: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMError(Exception):
    """Base for every failure the provider layer raises."""


class LLMUnavailableError(LLMError):
    """The provider could not be reached, timed out, or refused the request.

    Retrying the same call may or may not help; the caller should degrade rather than
    loop, because the fault is not in what was asked.
    """


class LLMInvalidOutputError(LLMError):
    """The provider answered, but the answer did not satisfy the schema."""

    def __init__(self, message: str, raw: str = "", attempts: int = 0) -> None:
        super().__init__(message)
        self.raw = raw
        self.attempts = attempts


class LLMProvider(Protocol):
    """Anything that can turn messages into text."""

    name: str
    model: str

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_schema: dict | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> Completion:
        """Return one completion, or raise ``LLMUnavailableError``."""
        ...


def user(content: str) -> Message:
    return Message(role="user", content=content)


def assistant(content: str) -> Message:
    return Message(role="assistant", content=content)
