"""Choose a provider from configuration.

The provider name is the only place in the codebase that knows which vendor is in use.
Agents receive whatever this returns.
"""

from __future__ import annotations

import logging

from drdoom.config import load_env_file
from drdoom.llm.base import LLMProvider, LLMUnavailableError

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
