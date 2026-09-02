"""Scoring the generated half, and the snapshots that make it free to run."""

import json

import pytest

from drdoom.evals.groundedness import (
    FILLER,
    JUDGE_RUBRIC,
    content_words,
    judge_prompt,
    score_text,
    split_sentences,
)
from drdoom.evals.run import THRESHOLDS, CaseResult, check, render_markdown, summarise
from drdoom.llm.base import Completion, LLMUnavailableError, user
from drdoom.llm.recording import (
    RecordingProvider,
    ReplayProvider,
    SnapshotStore,
    request_key,
)
from drdoom.llm.stub import StubProvider

CONTEXT = (
    "Set a memory limit on the container so the kubelet restarts it when the limit is "
    "exceeded. A deployment can be restarted with a rollout restart, which recreates "
    "its pods without changing the image."
)


# --- groundedness ------------------------------------------------------------------


def test_an_answer_drawn_from_the_context_scores_high() -> None:
    answer = "The container exceeded its memory limit. Restart the deployment pods."

    assert score_text(answer, CONTEXT).score > 0.9


def test_an_answer_the_context_does_not_mention_scores_low() -> None:
    answer = "A corrupted routing table on the edge appliance poisoned the anycast announcement."

    assert score_text(answer, CONTEXT).score < 0.3


def test_a_mixed_answer_lands_between() -> None:
    answer = (
        "The container exceeded its memory limit. "
        "A corrupted routing table poisoned the anycast announcement."
    )

    report = score_text(answer, CONTEXT)

    assert 0.3 < report.score < 0.9
    assert len(report.unsupported) == 1


def test_the_unsupported_sentence_is_named() -> None:
    answer = "Restart the deployment pods. Quantum decoherence in the hypervisor scheduler ring."

    unsupported = score_text(answer, CONTEXT).unsupported

    assert len(unsupported) == 1
    assert "decoherence" in unsupported[0].sentence


def test_filler_words_do_not_count_as_evidence() -> None:
    """Otherwise any sentence made of connectives would score as fully grounded."""
    assert "the" in FILLER
    assert content_words("the and for that with from") == set()


def test_short_sentences_are_not_scored() -> None:
    """A two-word fragment carries no claim, and scoring it only adds noise."""
    report = score_text("Yes. Restart the deployment pods now.", CONTEXT)

    assert len(report.sentences) == 1


def test_an_empty_answer_is_vacuously_grounded() -> None:
    report = score_text("", CONTEXT)

    assert report.score == 1.0
    assert report.supported_fraction == 1.0


def test_sentences_split_on_terminators() -> None:
    assert len(split_sentences("One thing. Two things! Three things?")) == 3


def test_the_report_serialises() -> None:
    payload = score_text("The container exceeded its memory limit today.", CONTEXT).as_dict()

    assert set(payload) == {"score", "supported_fraction", "sentences", "unsupported"}


def test_the_judge_rubric_defines_its_scale_and_asks_for_json() -> None:
    assert "0.0" in JUDGE_RUBRIC and "1.0" in JUDGE_RUBRIC
    assert "not correctness" in JUDGE_RUBRIC
    assert "strict JSON" in JUDGE_RUBRIC
    assert "EXCERPTS" in judge_prompt("d", "c")


# --- snapshots ---------------------------------------------------------------------


def test_the_key_is_stable_for_the_same_request() -> None:
    first = request_key("m", [user("hello")], "sys", {"a": 1})
    second = request_key("m", [user("hello")], "sys", {"a": 1})

    assert first == second


@pytest.mark.parametrize(
    "changed",
    [
        {"model": "other"},
        {"messages": [user("different")]},
        {"system": "other system"},
        {"json_schema": {"b": 2}},
    ],
)
def test_changing_any_part_of_the_request_changes_the_key(changed: dict) -> None:
    """A prompt edit must miss the cache, not replay an answer to a different question."""
    base = {"model": "m", "messages": [user("hello")], "system": "sys", "json_schema": {"a": 1}}

    assert request_key(**base) != request_key(**{**base, **changed})


def test_a_snapshot_round_trips(tmp_path) -> None:
    store = SnapshotStore(tmp_path)
    completion = Completion(text="hello", model="m", provider="p", input_tokens=3, output_tokens=1)

    store.write("abc", completion, note="test")
    restored = store.read("abc")

    assert restored.text == "hello"
    assert restored.input_tokens == 3
    assert len(store) == 1


def test_a_missing_snapshot_reads_as_nothing(tmp_path) -> None:
    assert SnapshotStore(tmp_path).read("nope") is None
    assert len(SnapshotStore(tmp_path / "absent")) == 0


def test_recording_saves_what_the_provider_said(tmp_path) -> None:
    store = SnapshotStore(tmp_path)
    inner = StubProvider(default="recorded answer")

    RecordingProvider(inner, store, note="eval").complete([user("question")])

    assert len(store) == 1
    saved = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert saved["text"] == "recorded answer"
    assert saved["note"] == "eval"


def test_replay_serves_what_was_recorded(tmp_path) -> None:
    store = SnapshotStore(tmp_path)
    RecordingProvider(StubProvider(default="the answer"), store).complete([user("question")])

    replayed = ReplayProvider(store, model="stub-1").complete([user("question")])

    assert replayed.text == "the answer"


def test_replay_refuses_to_invent_an_answer(tmp_path) -> None:
    provider = ReplayProvider(SnapshotStore(tmp_path))

    with pytest.raises(LLMUnavailableError, match="no recorded response"):
        provider.complete([user("never asked before")])

    assert len(provider.misses) == 1


def test_replay_is_deterministic(tmp_path) -> None:
    store = SnapshotStore(tmp_path)
    RecordingProvider(StubProvider(default="fixed"), store).complete([user("q")])
    provider = ReplayProvider(store, model="stub-1")

    assert provider.complete([user("q")]).text == provider.complete([user("q")]).text


# --- thresholds and reporting ------------------------------------------------------


def sample_summary(**overrides) -> dict:
    base = {
        "cases": 15,
        "retrieval_hit_at_5": 0.85,
        "retrieval_mrr": 0.62,
        "retrieval_queries": 40,
        "diagnosis_retrieval_hit": 0.60,
        "groundedness": 0.59,
        "supported_fraction": 0.49,
        "expected_terms_present": 0.66,
        "parse_success": 1.0,
        "degraded": 0,
        "total_tokens": 4200,
    }
    return {**base, **overrides}


def test_a_healthy_run_fails_nothing() -> None:
    assert check(sample_summary()) == []


def test_a_groundedness_regression_is_caught() -> None:
    failures = check(sample_summary(groundedness=0.20))

    assert len(failures) == 1
    assert "groundedness" in failures[0]


def test_a_parse_failure_is_caught() -> None:
    """Any unparseable diagnosis fails the build; the floor is one."""
    assert check(sample_summary(parse_success=0.95))


def test_several_regressions_are_all_reported() -> None:
    assert len(check(sample_summary(groundedness=0.1, retrieval_hit_at_5=0.1))) == 2


def test_the_floors_sit_below_the_measured_baseline() -> None:
    """A floor above the measured value makes the suite permanently red and ignored."""
    measured = sample_summary()

    for name, floor in THRESHOLDS.items():
        assert floor <= measured[name], f"{name} floor {floor} exceeds baseline {measured[name]}"


def test_the_summary_averages_over_cases() -> None:
    results = [
        CaseResult("a", True, 1.0, 1.0, True, True, False, 100),
        CaseResult("b", False, 0.0, 0.0, False, True, False, 50),
    ]

    summary = summarise(results, {"hit_at_5": 0.7, "mrr": 0.5, "queries": 40})

    assert summary["groundedness"] == 0.5
    assert summary["diagnosis_retrieval_hit"] == 0.5
    assert summary["total_tokens"] == 150


def test_the_report_states_what_the_measure_is_not() -> None:
    results = [CaseResult("a", True, 0.9, 1.0, True, True, False, 10)]

    text = render_markdown(sample_summary(), results, [])

    assert "proxy for entailment, not entailment" in text
    assert "| Measure | Score | Floor |" in text
    assert "| a |" in text


def test_failures_are_listed_in_the_report() -> None:
    text = render_markdown(sample_summary(), [], ["groundedness: 0.400 below floor 0.65"])

    assert "## Failures" in text
    assert "below floor" in text


def test_replay_finds_snapshots_recorded_under_another_model_name(tmp_path) -> None:
    """The model is part of the key, so replay has to use the one that was recorded.

    Guessing it wrong misses every lookup and degrades the pipeline quietly, which reads
    as a quality regression rather than a configuration mistake.
    """
    store = SnapshotStore(tmp_path)
    RecordingProvider(StubProvider(default="recorded", model="vendor/big-model"), store).complete(
        [user("question")]
    )

    replayed = ReplayProvider(store).complete([user("question")])

    assert store.recorded_model() == "vendor/big-model"
    assert replayed.text == "recorded"


def test_the_manifest_is_not_counted_as_a_snapshot(tmp_path) -> None:
    store = SnapshotStore(tmp_path)
    store.set_recorded_model("m")

    assert len(store) == 0


def test_replay_without_a_manifest_still_reports_a_miss(tmp_path) -> None:
    provider = ReplayProvider(SnapshotStore(tmp_path))

    with pytest.raises(LLMUnavailableError):
        provider.complete([user("never recorded")])
