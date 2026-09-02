"""A deterministic provider for tests and offline runs.

Every agent test in this project runs against this, so the whole pipeline is exercised
with no API key, no network, and no per-run cost. Responses are matched by substring
against the outgoing prompt, which keeps a test's expectation next to the behaviour it
is checking rather than hidden in a fixture file.
"""

from __future__ import annotations

from collections.abc import Callable

from drdoom.llm.base import Completion, LLMUnavailableError, Message

Responder = Callable[[list[Message]], str]


class StubProvider:
    """Returns canned text, chosen by matching substrings in the prompt."""

    def __init__(
        self,
        default: str = "stub response",
        rules: list[tuple[str, str]] | None = None,
        responder: Responder | None = None,
        fail_with: Exception | None = None,
        model: str = "stub-1",
    ) -> None:
        self.name = "stub"
        self.model = model
        self.default = default
        self.rules = rules or []
        self.responder = responder
        self.fail_with = fail_with
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_schema: dict | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = 45.0,
    ) -> Completion:
        self.calls.append(list(messages))
        if self.fail_with is not None:
            raise self.fail_with

        prompt = "\n".join(message.content for message in messages)
        if self.responder is not None:
            text = self.responder(messages)
        else:
            text = next((reply for needle, reply in self.rules if needle in prompt), self.default)

        return Completion(
            text=text,
            model=self.model,
            provider=self.name,
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
            latency_ms=0.0,
        )


class SequenceProvider:
    """Returns prepared responses in order, so a repair retry can be driven exactly."""

    def __init__(self, responses: list[str], model: str = "stub-seq") -> None:
        if not responses:
            raise ValueError("SequenceProvider needs at least one response")
        self.name = "stub"
        self.model = model
        self.responses = responses
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_schema: dict | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = 45.0,
    ) -> Completion:
        self.calls.append(list(messages))
        if len(self.calls) > len(self.responses):
            raise LLMUnavailableError("SequenceProvider exhausted")
        return Completion(
            text=self.responses[len(self.calls) - 1],
            model=self.model,
            provider=self.name,
            input_tokens=1,
            output_tokens=1,
        )
