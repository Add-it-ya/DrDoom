"""Choose a provider from configuration.

The provider name is the only place in the codebase that knows which vendor is in use.
Agents receive whatever this returns.
"""

from __future__ import annotations

import logging

from drdoom.config import load_env_file
from drdoom.llm.base import Completion, LLMProvider, LLMUnavailableError, Message

logger = logging.getLogger(__name__)

PROVIDERS = ("groq", "anthropic", "stub")


def build_provider(name: str = "groq", model: str | None = None) -> LLMProvider:
    """Return a provider by name, raising if it cannot be constructed."""
    load_env_file()
    if name == "groq":
        from drdoom.llm.groq import DEFAULT_MODEL, GroqProvider

        return GroqProvider(model=model or DEFAULT_MODEL)
    if name == "anthropic":
        from drdoom.llm.anthropic_backend import DEFAULT_MODEL, AnthropicProvider

        return AnthropicProvider(model=model or DEFAULT_MODEL)
    if name == "stub":
        from drdoom.llm.stub import StubProvider

        return StubProvider(model=model or "stub-1")
    raise LLMUnavailableError(f"unknown provider {name!r}; expected one of {PROVIDERS}")


class UnavailableProvider:
    """Stands in for a provider that could not be built, so the service still starts.

    Every call fails the way an unreachable provider fails, and each agent already answers
    that by degrading: retrieved passages instead of a summary, an empty plan held for a
    human, a report of the recorded facts. A missing key is then a configuration problem
    the health check reports, not a container that exits and restarts forever.
    """

    name = "unavailable"

    def __init__(self, reason: str, model: str = "none") -> None:
        self.model = model
        self.reason = reason

    def complete(self, messages: list[Message], **_: object) -> Completion:
        raise LLMUnavailableError(self.reason)


def build_provider_or_unavailable(name: str = "groq", model: str | None = None) -> LLMProvider:
    """The configured provider, or a stand-in that makes every agent degrade and says why."""
    try:
        return build_provider(name, model)
    except LLMUnavailableError as error:
        logger.error(
            "no language model (%s); every agent will degrade until one is configured", error
        )
        return UnavailableProvider(str(error))
