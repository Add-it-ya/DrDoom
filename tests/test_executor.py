"""The gate, and the record it leaves.

Two claims are under test: nothing runs without an approval naming the exact plan, and
the log can tell you if it has been edited since.
"""

import json

import pytest

from drdoom.agents.schemas import RemediationPlan
from drdoom.audit import GENESIS, AuditLog, compute_entry_hash
from drdoom.executor import (
    CATALOGUE,
    ApprovalToken,
    DryRunExecutor,
    NotApprovedError,
    match_action,
    plan_hash,
)


def make_plan(action: str = "Rolling restart of the affected pods", risk: str = "high"):
    return RemediationPlan(
        immediate_action=action,
        risk_level=risk,
        short_term_fix="Set a memory limit",
        long_term_fix="Fix the cache eviction policy",
        rollback="Scale the previous replica set back up",
    )


# --- the gate ----------------------------------------------------------------------


def test_nothing_runs_without_an_approval() -> None:
    result = DryRunExecutor().execute(make_plan(), None)

    assert result.executed is False
    assert result.kind == "refused"
    assert "no approval" in result.detail


def test_an_approved_plan_renders_its_command() -> None:
    plan = make_plan()

    result = DryRunExecutor(target="api").execute(plan, ApprovalToken.issue(plan, "aditya"))

    assert result.executed is True
    assert result.command == "kubectl rollout restart deployment/api"
    assert result.dry_run is True


def test_a_token_for_a_different_plan_is_refused() -> None:
    """Approving a restart must not authorise deleting a volume."""
    approved = make_plan()
    substituted = make_plan(action="Delete the persistent volume claim")
    token = ApprovalToken.issue(approved, "aditya")

    with pytest.raises(NotApprovedError, match="does not match this plan"):
        DryRunExecutor().execute(substituted, token)


def test_editing_any_field_invalidates_the_approval() -> None:
    approved = make_plan(risk="medium")
    token = ApprovalToken.issue(approved, "aditya")
    escalated = make_plan(risk="high")

    with pytest.raises(NotApprovedError):
        DryRunExecutor().execute(escalated, token)


def test_a_token_authorises_the_plan_it_was_issued_for() -> None:
    plan = make_plan()

    assert ApprovalToken.issue(plan, "aditya").authorises(plan) is True


def test_an_unrecognised_action_is_refused_rather_than_guessed() -> None:
    plan = make_plan(action="Politely ask the database to behave")

    result = DryRunExecutor().execute(plan, ApprovalToken.issue(plan, "aditya"))

    assert result.executed is False
    assert result.kind == "unrecognised"


@pytest.mark.parametrize(
    ("action", "kind"),
    [
        ("Perform a rolling restart of the pods", "rollout_restart"),
        ("Roll back to the previous version", "rollout_undo"),
        ("Scale up the deployment to absorb the load", "scale_out"),
        ("Apply a memory limit to the container", "set_memory_limit"),
        ("Drain the node before maintenance", "cordon_node"),
    ],
)
def test_known_actions_are_recognised(action: str, kind: str) -> None:
    spec = match_action(action)

    assert spec is not None
    assert spec.kind == kind


def test_every_catalogue_entry_renders_a_command() -> None:
    for spec in CATALOGUE:
        assert "{target}" in spec.template
        assert spec.template.format(target="x")


def test_plan_hash_is_stable_and_specific() -> None:
    assert plan_hash(make_plan()) == plan_hash(make_plan())
    assert plan_hash(make_plan()) != plan_hash(make_plan(risk="low"))


def test_plan_hash_covers_the_derived_approval_flag() -> None:
    """A low-risk plan and a high-risk one differ by more than the label."""
    low, high = make_plan(risk="low"), make_plan(risk="high")

    assert low.requires_approval != high.requires_approval
    assert plan_hash(low) != plan_hash(high)


# --- the audit log -----------------------------------------------------------------


def record(log: AuditLog, incident: str = "i1", decision: str = "approved_by_human", **kwargs):
    plan = kwargs.pop("plan", make_plan())
    return log.record(
        incident_id=incident,
        principal=kwargs.pop("principal", "aditya"),
        decision=decision,
        risk_level=plan.risk_level,
        immediate_action=plan.immediate_action,
        plan_hash=plan_hash(plan),
        executed=kwargs.pop("executed", True),
        execution=kwargs.pop("execution", "kubectl rollout restart deployment/api"),
    )


def test_an_empty_log_is_valid(tmp_path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")

    assert log.entries() == []
    assert log.verify() == (True, "chain intact")


def test_the_first_entry_follows_the_genesis_hash(tmp_path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")

    entry = record(log)

    assert entry.previous_hash == GENESIS


def test_entries_chain_to_the_one_before(tmp_path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")

    first = record(log, incident="i1")
    second = record(log, incident="i2")

    assert second.previous_hash == first.entry_hash
    assert log.verify()[0] is True


def test_the_recorded_hash_matches_the_plan_that_was_approved(tmp_path) -> None:
    """A review must be able to confirm what ran is what was shown."""
    log = AuditLog(tmp_path / "audit.jsonl")
    plan = make_plan()

    record(log, plan=plan)

    assert log.entries()[0].plan_hash == plan_hash(plan)


def test_editing_an_entry_is_detected(tmp_path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    record(log, incident="i1")
    record(log, incident="i2")

    lines = path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["decision"] = "approved_by_human"
    tampered["principal"] = "someone-else"
    path.write_text(json.dumps(tampered, sort_keys=True) + "\n" + lines[1] + "\n", encoding="utf-8")

    valid, reason = log.verify()
    assert valid is False
    assert "altered" in reason


def test_removing_an_entry_is_detected(tmp_path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    record(log, incident="i1")
    record(log, incident="i2")

    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(lines[1] + "\n", encoding="utf-8")

    valid, reason = log.verify()
    assert valid is False
    assert "follow" in reason


def test_recording_only_ever_appends(tmp_path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    record(log, incident="i1")
    first_line = path.read_text(encoding="utf-8").splitlines()[0]

    record(log, incident="i2")

    assert path.read_text(encoding="utf-8").splitlines()[0] == first_line


def test_rejections_are_recorded_too(tmp_path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")

    record(log, decision="rejected_by_human", executed=False, execution="nothing was executed")

    entry = log.entries()[0]
    assert entry.decision == "rejected_by_human"
    assert entry.executed is False


def test_entries_can_be_read_back_per_incident(tmp_path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    record(log, incident="i1")
    record(log, incident="i2")
    record(log, incident="i1")

    assert len(log.for_incident("i1")) == 2


def test_the_entry_hash_covers_every_field_but_itself(tmp_path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    entry = record(log)

    assert compute_entry_hash(entry.payload()) == entry.entry_hash
    assert "entry_hash" not in entry.payload()
