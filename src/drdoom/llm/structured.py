"""Get a validated object out of a model, or fail loudly.

Asking for JSON and calling ``json.loads`` on the answer is the shape that produces a
500 in production. Syntactically valid JSON is not a valid plan: a key can be missing, a
risk level can be a sentence, a field can arrive as a string where a list was meant. The
code that indexes ``plan["immediate_action"]`` three layers down is where that surfaces.

So every structured call goes through here. The response is parsed and validated against
a pydantic model, and when validation fails the error is handed back to the model with
one bounded chance to correct it. One retry, not a loop: a model that cannot satisfy the
schema on the second attempt is not usually going to satisfy it on the fifth, and an
unbounded repair loop turns a bad response into a bill.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from drdoom.llm.base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT_SECONDS,
    Completion,
    LLMInvalidOutputError,
    LLMProvider,
    Message,
)

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
DEFAULT_REPAIRS = 1


def extract_json(text: str) -> str:
    """Pull the JSON object out of a response that may be wrapped in prose or fences."""
    fenced = FENCE.search(text)
    if fenced:
        return fenced.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text.strip()


def _repair_prompt(raw: str, error: str) -> str:
    return (
        "The previous response could not be used. It failed validation with:\n"
        f"{error}\n\n"
        "Previous response:\n"
        f"{raw}\n\n"
        "Return only corrected JSON matching the requested structure. No commentary, "
        "no code fences."
    )


def generate_structured(
    provider: LLMProvider,
    messages: list[Message],
    schema: type[ModelT],
    *,
    system: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_repairs: int = DEFAULT_REPAIRS,
) -> tuple[ModelT, list[Completion]]:
    """Return a validated model instance and every completion it took to get there.

    Completions are returned rather than discarded so the caller can account for the
    tokens a repair actually cost.
    """
    json_schema = schema.model_json_schema()
    conversation = list(messages)
    completions: list[Completion] = []
    last_error = ""
    last_raw = ""

    for attempt in range(max_repairs + 1):
        completion = provider.complete(
            conversation,
            system=system,
            json_schema=json_schema,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        completions.append(completion)
        last_raw = completion.text

        try:
            return schema.model_validate_json(extract_json(completion.text)), completions
        except (ValidationError, json.JSONDecodeError, ValueError) as error:
            last_error = str(error)
            logger.warning(
                "structured output failed validation on attempt %d: %s", attempt + 1, last_error
            )
            if attempt == max_repairs:
                break
            conversation = [
                *conversation,
                Message(role="assistant", content=completion.text),
                Message(role="user", content=_repair_prompt(completion.text, last_error)),
            ]

    raise LLMInvalidOutputError(
        f"could not obtain a valid {schema.__name__} after {max_repairs + 1} attempts: "
        f"{last_error}",
        raw=last_raw,
        attempts=max_repairs + 1,
    )
