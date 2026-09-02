"""What an incident costs to investigate.

Token counts are the honest unit -- they are what the provider actually meters -- but
nobody budgets in tokens, so they are converted here for reporting. Rates change and vary
by provider, so they live in one table with a date on them rather than scattered through
the code, and an unknown model reports tokens with no cost rather than guessing a number.
"""

from __future__ import annotations

from dataclasses import dataclass

RATES_UPDATED = "2026-09"


@dataclass(frozen=True)
class Rate:
    """Cost per million tokens, in United States dollars."""

    input_per_million: float
    output_per_million: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_million + output_tokens * self.output_per_million
        ) / 1_000_000


RATES: dict[str, Rate] = {
    "llama-3.3-70b-versatile": Rate(0.59, 0.79),
    "claude-opus-5": Rate(5.00, 25.00),
    "claude-sonnet-5": Rate(2.00, 10.00),
    "claude-haiku-4-5": Rate(1.00, 5.00),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Cost in dollars, or ``None`` when the model's rate is not known here."""
    rate = RATES.get(model)
    if rate is None:
        return None
    return rate.cost(input_tokens, output_tokens)


def describe(model: str, input_tokens: int, output_tokens: int) -> dict:
    """A reportable accounting of one incident's model usage."""
    cost = estimate_cost(model, input_tokens, output_tokens)
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost_usd": round(cost, 6) if cost is not None else None,
        "rates_updated": RATES_UPDATED if cost is not None else None,
    }
