"""The independent risk rating: blind to the author, unable to lower anything."""

import json

import pytest

from drdoom.agents import remediation
from drdoom.agents.risk import SYSTEM, RiskAssessor, highest
from drdoom.agents.schemas import Citation, Diagnosis, RemediationPlan
from drdoom.audit import AuditLog, compute_entry_hash
from drdoom.executor import CATALOGUE, risk_floor
from drdoom.llm.base import LLMUnavailableError
from drdoom.llm.stub import StubProvider

TRIAGE = {"is_anomaly": True, "score": 9.1, "threshold": 5.0, "root_cause": "memory_leak"}
DIAGNOSIS = Diagnosis(
    summary="Memory grew until the container was killed.",
    likely_cause="memory leak",
    confidence="high",
    next_action="Restart the pods.",
)
CITATIONS = [Citation(chunk_id="c1", doc_id="k8s:memory", title="Assign Memory Resources")]
ASSESSMENT = json.dumps(
    {"risk_level": "medium", "worst_case": "Pods restart at once.", "reasons": ["restart"]}
)


def plan(risk: str = "low", action: str | None = "rollout_restart") -> RemediationPlan:
    return RemediationPlan(
        immediate_action="Rolling restart of the api pods",
        risk_level=risk,
        short_term_fix="Set a memory limit",
        long_term_fix="Fix the cache",
        rollback="Scale the previous replica set back up",
        action=action,
    )


def prompt_for(subject: RemediationPlan) -> str:
    return RiskAssessor(StubProvider()).prompt(
        TRIAGE, DIAGNOSIS, subject, "kubectl rollout restart deployment/x", CITATIONS
    )


# --- combining -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("levels", "expected"),
    [
        (("low", "low", "low"), "low"),
        (("low", "medium", "low"), "medium"),
        (("high", "low", "low"), "high"),
        (("medium", "low", "high"), "high"),
    ],
)
def test_the_most_cautious_rating_stands(levels, expected) -> None:
    assert highest(*levels) == expected


# --- the floor -----------------------------------------------------------------------


def test_every_catalogue_action_has_a_floor() -> None:
    for spec in CATALOGUE:
        assert spec.risk_floor in {"low", "medium", "high"}


def test_draining_a_node_is_high_risk_whatever_anyone_says() -> None:
    assert risk_floor("cordon_node") == "high"


def test_only_adding_capacity_has_a_low_floor() -> None:
    assert [spec.kind for spec in CATALOGUE if spec.risk_floor == "low"] == ["scale_out"]


def test_a_plan_that_runs_nothing_has_a_low_floor() -> None:
    assert risk_floor(None) == "low"


# --- what the assessor is and is not told --------------------------------------------


def test_the_assessor_never_sees_the_authors_rating() -> None:
    """Two plans that differ only in their author's rating produce the same prompt."""
    assert prompt_for(plan(risk="low")) == prompt_for(plan(risk="medium"))


def test_the_assessor_is_not_told_which_rating_skips_the_human() -> None:
    for text in (SYSTEM, prompt_for(plan())):
        lowered = text.lower()
        assert "approv" not in lowered
        assert "unattended" not in lowered
        assert "human" not in lowered


def test_the_assessor_judges_the_command_that_would_run() -> None:
    prompt = prompt_for(plan())

    assert "kubectl rollout restart deployment/x" in prompt
    assert "Assign Memory Resources" in prompt
    assert "Memory grew until the container was killed." in prompt


def test_the_plan_author_is_no_longer_told_what_low_allows() -> None:
    assert "unattended" not in remediation.SYSTEM.lower()


# --- when it cannot answer ------------------------------------------------------------


def test_an_answer_is_used() -> None:
    result = RiskAssessor(StubProvider(default=ASSESSMENT)).run(
        TRIAGE, DIAGNOSIS, plan(), None, CITATIONS
    )

    assert result.level == "medium"
    assert result.degraded is False
    assert result.assessment.worst_case == "Pods restart at once."


def test_no_answer_counts_as_high() -> None:
    provider = StubProvider(fail_with=LLMUnavailableError("down"))

    result = RiskAssessor(provider).run(TRIAGE, DIAGNOSIS, plan(), None, CITATIONS)

    assert result.level == "high"
    assert result.degraded is True
    assert result.failure == "unavailable"


def test_an_unusable_answer_counts_as_high_and_is_billed() -> None:
    result = RiskAssessor(StubProvider(default="looks fine to me")).run(
        TRIAGE, DIAGNOSIS, plan(), None, CITATIONS
    )

    assert result.level == "high"
    assert result.failure == "invalid_output"
    assert len(result.completions) == 2


# --- the record ------------------------------------------------------------------------


def test_the_audit_entry_records_how_the_risk_was_decided(tmp_path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    risk = {"author": "low", "floor": "medium", "assessor": "low", "final": "medium"}

    entry = log.record(
        incident_id="i",
        principal="p",
        decision="approved_by_human",
        risk_level="medium",
        immediate_action="restart",
        plan_hash="h",
        executed=True,
        execution="ran",
        risk=risk,
    )

    assert log.entries()[0].risk == risk
    assert entry.entry_hash == compute_entry_hash(entry.payload())
    assert log.verify() == (True, "chain intact")


def test_a_log_written_before_risk_was_recorded_still_verifies(tmp_path) -> None:
    """An older entry has no risk field, and its hash must come out exactly as before."""
    path = tmp_path / "audit.jsonl"
    payload = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "incident_id": "old",
        "principal": "p",
        "decision": "approved_by_human",
        "risk_level": "high",
        "immediate_action": "restart",
        "plan_hash": "h",
        "executed": True,
        "execution": "ran",
        "previous_hash": "0" * 64,
    }
    old_line = payload | {"entry_hash": compute_entry_hash(payload)}
    path.write_text(json.dumps(old_line, sort_keys=True) + "\n", encoding="utf-8")
    log = AuditLog(path)

    log.record(
        incident_id="new",
        principal="p",
        decision="approved_by_human",
        risk_level="medium",
        immediate_action="restart",
        plan_hash="h2",
        executed=True,
        execution="ran",
        risk={"final": "medium"},
    )

    assert log.verify() == (True, "chain intact")


def test_rewriting_the_recorded_risk_breaks_the_chain(tmp_path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.record(
        incident_id="i",
        principal="p",
        decision="approved_by_human",
        risk_level="medium",
        immediate_action="restart",
        plan_hash="h",
        executed=True,
        execution="ran",
        risk={"author": "low", "final": "medium"},
    )

    path.write_text(path.read_text(encoding="utf-8").replace('"low"', '"high"'), "utf-8")

    assert log.verify()[0] is False


# --- the labelled cases ------------------------------------------------------------------


def test_every_labelled_case_is_a_valid_plan_with_a_valid_label() -> None:
    from drdoom.evals.risk import load_cases

    cases = load_cases()

    assert len({case["id"] for case in cases}) == len(cases) >= 15
    for case in cases:
        RemediationPlan.model_validate(case["plan"])
        Diagnosis.model_validate(case["diagnosis"])
        assert case["expected"] in {"low", "medium", "high"}


def test_the_set_holds_under_ratings_only_an_assessor_can_catch() -> None:
    """Otherwise it would measure the floor twice and the assessor not at all."""
    from drdoom.agents.risk import ORDER
    from drdoom.evals.risk import load_cases

    beyond_floor = [
        case
        for case in load_cases()
        if ORDER[highest(case["plan"]["risk_level"], risk_floor(case["plan"]["action"]))]
        < ORDER[case["expected"]]
    ]

    assert len(beyond_floor) >= 3


def test_the_rules_are_scored_against_the_label() -> None:
    from drdoom.evals.risk import CaseResult, summarise

    results = [
        CaseResult("a", expected="high", author="low", floor="medium", assessor="high"),
        CaseResult("b", expected="low", author="medium", floor="low", assessor="low"),
    ]

    summary = summarise(results)

    assert (summary["author"]["under"], summary["author"]["over"]) == (1, 1)
    assert summary["author_and_floor"]["under_ids"] == ["a"]
    assert summary["all_three"]["under"] == 0


def test_without_recorded_answers_the_assessor_row_is_not_measured() -> None:
    from drdoom.evals.risk import CaseResult, render, summarise

    results = [CaseResult("a", expected="high", author="low", floor="medium", assessor=None)]

    page = render(summarise(results), results)

    assert "| Author, floor and assessor (now) | not measured |" in page


def test_a_case_is_run_through_the_assessor() -> None:
    from drdoom.evals.risk import load_cases, run_case

    result = run_case(RiskAssessor(StubProvider(default=ASSESSMENT)), load_cases()[0])

    assert result.assessor == "medium"
    assert result.floor == risk_floor(load_cases()[0]["plan"]["action"])


def test_the_published_risk_page_is_what_the_suite_writes(tmp_path) -> None:
    from pathlib import Path

    from drdoom.evals.risk import main

    main(["--out", str(tmp_path)])
    published = Path(__file__).resolve().parents[1] / "docs" / "risk-results.md"

    assert (tmp_path / "risk-results.md").read_text(encoding="utf-8") == published.read_text(
        encoding="utf-8"
    )
