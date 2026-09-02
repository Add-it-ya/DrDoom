"""Groq backend, reached through its OpenAI-compatible endpoint.

This is the default because its free tier makes the whole system runnable, and a
demonstration nobody can afford to run is not a demonstration. The SDK is imported
lazily so that installing the project, importing the agents, and running the tests never
require it to be configured.

Every failure the vendor raises is translated into this project's two error types, so
callers never catch a vendor exception and the agents stay provider-agnostic.
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

BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
API_KEY_ENV = "GROQ_API_KEY"


class GroqProvider:
    """Chat completions against Groq."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        max_retries: int = 2,
    ) -> None:
        from openai import OpenAI

        key = api_key or os.environ.get(API_KEY_ENV)
        if not key:
            raise LLMUnavailableError(
                f"no Groq credentials; set {API_KEY_ENV} or pass api_key explicitly"
            )

        self.name = "groq"
        self.model = model
        self._client = OpenAI(api_key=key, base_url=base_url, max_retries=max_retries)

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
        import openai

        payload = [{"role": message.role, "content": message.content} for message in messages]
        if system:
            payload.insert(0, {"role": "system", "content": system})

        extra: dict = {}
        if json_schema is not None:
            # Schema support varies across models served here, so the reliable request is
            # plain json mode. The schema is still enforced downstream by validation,
            # which has to happen regardless of what the provider promises.
            extra["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=payload,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                **extra,
            )
        except openai.APIStatusError as error:
            raise LLMUnavailableError(f"groq returned {error.status_code}: {error}") from error
        except (openai.APIConnectionError, openai.APITimeoutError) as error:
            raise LLMUnavailableError(f"groq unreachable: {error}") from error

        latency_ms = (time.perf_counter() - started) * 1000
        usage = response.usage
        return Completion(
            text=response.choices[0].message.content or "",
            model=self.model,
            provider=self.name,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=latency_ms,
        )
