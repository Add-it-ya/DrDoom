"""Anthropic backend.

Optional: install with ``uv sync --extra anthropic``. Nothing in the pipeline requires
it, and the default provider is free to run, so this exists to keep the seam honest --
an abstraction with a single implementation has not been shown to abstract anything.

The module is deliberately not called ``anthropic.py``. A module of that name inside this
package would shadow the vendor package on import and fail in a way that reads like a
missing install.

Unlike the chat-completions backend, this one can constrain the response to a schema
server-side, which it does when one is supplied. Validation still happens downstream:
a provider guarantee is worth using and not worth trusting alone.
"""

from __future__ import annotations

import logging
import os
import time

from drdoom.llm.base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT_SECONDS,
    Completion,
    LLMUnavailableError,
    Message,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"
API_KEY_ENV = "ANTHROPIC_API_KEY"


class AnthropicProvider:
    """Messages API, with server-side schema constraint when a schema is given."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_retries: int = 2,
    ) -> None:
        try:
            import anthropic
        except ImportError as error:  # pragma: no cover - depends on install extras
            raise LLMUnavailableError(
                "anthropic backend not installed; run: uv sync --extra anthropic"
            ) from error

        key = api_key or os.environ.get(API_KEY_ENV)
        self.name = "anthropic"
        self.model = model
        # A bare client also resolves an `ant auth login` profile, so an unset key is
        # not proof that there are no credentials.
        self._client = (
            anthropic.Anthropic(api_key=key, max_retries=max_retries)
            if key
            else anthropic.Anthropic(max_retries=max_retries)
        )

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
        import anthropic

        payload = [
            {"role": message.role, "content": message.content}
            for message in messages
            if message.role != "system"
        ]
        system_parts = [system] if system else []
        system_parts += [message.content for message in messages if message.role == "system"]
        leading_system = "\n\n".join(system_parts)

        request: dict = {"model": self.model, "max_tokens": max_tokens, "messages": payload}
        if leading_system:
            request["system"] = leading_system
        if json_schema is not None:
            request["output_config"] = {"format": {"type": "json_schema", "schema": json_schema}}

        started = time.perf_counter()
        try:
            response = self._client.with_options(timeout=timeout).messages.create(**request)
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as error:
            raise LLMUnavailableError(f"anthropic unreachable: {error}") from error
        except anthropic.APIStatusError as error:
            raise LLMUnavailableError(f"anthropic returned {error.status_code}: {error}") from error

        latency_ms = (time.perf_counter() - started) * 1000

        # A safety classifier can decline with a 200, so the stop reason has to be
        # checked before the content is read.
        if getattr(response, "stop_reason", None) == "refusal":
            raise LLMUnavailableError("anthropic declined the request")

        text = "".join(block.text for block in response.content if block.type == "text")
        return Completion(
            text=text,
            model=self.model,
            provider=self.name,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
        )
