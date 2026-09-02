"""The provider seam, and getting a validated object out of a model.

Nothing here reaches the network or needs a key.
"""

import pytest
from pydantic import BaseModel

from drdoom.llm.base import Completion, LLMInvalidOutputError, LLMUnavailableError, Message, user
from drdoom.llm.factory import build_provider
from drdoom.llm.structured import extract_json, generate_structured
from drdoom.llm.stub import SequenceProvider, StubProvider


class Plan(BaseModel):
    action: str
    urgency: int


def test_completion_totals_its_tokens() -> None:
    completion = Completion(text="x", model="m", provider="p", input_tokens=10, output_tokens=4)

    assert completion.total_tokens == 14


def test_stub_matches_a_rule_by_substring() -> None:
    provider = StubProvider(default="fallback", rules=[("memory", "leak detected")])

    assert provider.complete([user("high memory usage")]).text == "leak detected"
    assert provider.complete([user("disk full")]).text == "fallback"


def test_stub_records_what_it_was_asked() -> None:
    provider = StubProvider()

    provider.complete([user("first")])
    provider.complete([user("second")])

    assert [call[0].content for call in provider.calls] == ["first", "second"]


def test_stub_can_simulate_an_outage() -> None:
    provider = StubProvider(fail_with=LLMUnavailableError("down"))

    with pytest.raises(LLMUnavailableError):
        provider.complete([user("anything")])


def test_sequence_provider_needs_responses() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SequenceProvider([])


def test_sequence_provider_runs_out() -> None:
    provider = SequenceProvider(["only one"])
    provider.complete([user("a")])

    with pytest.raises(LLMUnavailableError, match="exhausted"):
        provider.complete([user("b")])


def test_json_is_extracted_from_a_fenced_block() -> None:
    assert extract_json('here:\n```json\n{"a": 1}\n```\nthanks') == '{"a": 1}'


def test_json_is_extracted_from_surrounding_prose() -> None:
    assert extract_json('Sure! {"a": 1} Hope that helps.') == '{"a": 1}'


def test_extraction_leaves_bare_json_alone() -> None:
    assert extract_json('{"a": 1}') == '{"a": 1}'


def test_valid_output_needs_one_call() -> None:
    provider = StubProvider(default='{"action": "restart", "urgency": 3}')

    plan, completions = generate_structured(provider, [user("go")], Plan)

    assert plan.action == "restart"
    assert len(completions) == 1


def test_malformed_output_is_repaired_on_the_second_attempt() -> None:
    provider = SequenceProvider(["not json", '{"action": "scale", "urgency": 1}'])

    plan, completions = generate_structured(provider, [user("go")], Plan)

    assert plan.action == "scale"
    assert len(completions) == 2


def test_the_repair_prompt_carries_the_validation_error() -> None:
    provider = SequenceProvider(['{"action": "scale"}', '{"action": "scale", "urgency": 2}'])

    generate_structured(provider, [user("go")], Plan)

    repair = provider.calls[1][-1].content
    assert "failed validation" in repair
    assert "urgency" in repair


def test_a_wrong_field_type_is_treated_as_invalid() -> None:
    provider = SequenceProvider(
        ['{"action": "scale", "urgency": "very"}', '{"action": "scale", "urgency": 2}']
    )

    plan, _ = generate_structured(provider, [user("go")], Plan)

    assert plan.urgency == 2


def test_repairs_are_bounded() -> None:
    provider = StubProvider(default="still not json")

    with pytest.raises(LLMInvalidOutputError) as raised:
        generate_structured(provider, [user("go")], Plan, max_repairs=1)

    assert raised.value.attempts == 2
    assert len(provider.calls) == 2


def test_repair_budget_is_configurable() -> None:
    provider = StubProvider(default="nope")

    with pytest.raises(LLMInvalidOutputError):
        generate_structured(provider, [user("go")], Plan, max_repairs=3)

    assert len(provider.calls) == 4


def test_the_raw_response_is_kept_on_failure() -> None:
    provider = StubProvider(default="I cannot help with that")

    with pytest.raises(LLMInvalidOutputError) as raised:
        generate_structured(provider, [user("go")], Plan, max_repairs=0)

    assert raised.value.raw == "I cannot help with that"


def test_an_outage_is_not_caught_as_a_validation_failure() -> None:
    provider = StubProvider(fail_with=LLMUnavailableError("network down"))

    with pytest.raises(LLMUnavailableError):
        generate_structured(provider, [user("go")], Plan)


def test_the_schema_is_offered_to_the_provider() -> None:
    seen: dict = {}

    class Recording(StubProvider):
        def complete(self, messages, **kwargs):
            seen.update(kwargs)
            return super().complete(messages, **kwargs)

    provider = Recording(default='{"action": "a", "urgency": 1}')
    generate_structured(provider, [user("go")], Plan)

    assert "action" in seen["json_schema"]["properties"]


def test_factory_builds_the_stub_without_credentials() -> None:
    provider = build_provider("stub")

    assert provider.name == "stub"


def test_factory_rejects_an_unknown_provider() -> None:
    with pytest.raises(LLMUnavailableError, match="unknown provider"):
        build_provider("telepathy")


def test_groq_without_a_key_fails_clearly(monkeypatch) -> None:
    # A developer .env would otherwise supply the key this test is about the absence of.
    monkeypatch.setattr("drdoom.llm.factory.load_env_file", lambda *a, **k: 0)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(LLMUnavailableError, match="GROQ_API_KEY"):
        build_provider("groq")


def test_messages_carry_their_role() -> None:
    message = Message(role="assistant", content="hi")

    assert message.role == "assistant"
    assert user("hello").role == "user"
