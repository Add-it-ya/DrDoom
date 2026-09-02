"""Structured logs, incident tagging, and stage timing."""

import json
import logging

from drdoom.observability import (
    Counters,
    JsonFormatter,
    Timings,
    configure_logging,
    current_incident,
    incident_context,
    timed,
)


def render(record: logging.LogRecord) -> dict:
    return json.loads(JsonFormatter().format(record))


def make_record(message: str = "something happened", **extra) -> logging.LogRecord:
    record = logging.LogRecord("drdoom.test", logging.INFO, __file__, 1, message, None, None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_a_log_line_is_one_json_object() -> None:
    payload = render(make_record())

    assert payload["message"] == "something happened"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "drdoom.test"


def test_every_line_carries_the_incident() -> None:
    """Without this, two concurrent runs interleave and neither can be reconstructed."""
    with incident_context("abc123"):
        payload = render(make_record())

    assert payload["incident"] == "abc123"


def test_outside_an_incident_the_field_is_still_present() -> None:
    assert render(make_record())["incident"] == "-"


def test_the_incident_is_restored_afterwards() -> None:
    with incident_context("outer"), incident_context("inner"):
        assert current_incident.get() == "inner"

    assert current_incident.get() == "-"


def test_extra_fields_reach_the_output() -> None:
    payload = render(make_record(stage="triage", duration_ms=12.5))

    assert payload["stage"] == "triage"
    assert payload["duration_ms"] == 12.5


def test_logging_internals_are_not_leaked_into_the_output() -> None:
    payload = render(make_record())

    assert "msecs" not in payload
    assert "processName" not in payload


def test_an_exception_is_captured() -> None:
    try:
        raise ValueError("broken")
    except ValueError:
        import sys

        record = make_record("failed")
        record.exc_info = sys.exc_info()

    assert "ValueError: broken" in render(record)["error"]


def test_plain_formatting_stays_available_for_a_terminal() -> None:
    configure_logging("INFO", structured=False)
    handler = logging.getLogger().handlers[0]

    assert not isinstance(handler.formatter, JsonFormatter)

    configure_logging("INFO", structured=True)
    assert isinstance(logging.getLogger().handlers[0].formatter, JsonFormatter)


def test_timing_records_a_stage() -> None:
    timings = Timings()

    with timed("triage", timings):
        pass

    assert "triage" in timings.stages
    assert timings.total_ms >= 0


def test_timings_sum_across_stages() -> None:
    timings = Timings()
    timings.record("a", 10.0)
    timings.record("b", 5.0)

    assert timings.total_ms == 15.0
    assert timings.as_dict()["stages_ms"] == {"a": 10.0, "b": 5.0}


def test_timing_survives_a_failure_inside_the_block() -> None:
    """A stage that raises is still the stage that took the time."""
    timings = Timings()

    try:
        with timed("diagnose", timings):
            raise RuntimeError("provider down")
    except RuntimeError:
        pass

    assert "diagnose" in timings.stages


def test_counters_track_events() -> None:
    counters = Counters()
    counters.increment("investigate")
    counters.increment("investigate")

    assert counters.snapshot()["events"]["investigate"] == 2


def test_counters_report_stage_percentiles() -> None:
    counters = Counters()
    for value in range(1, 101):
        counters.observe("triage", float(value))

    stage = counters.snapshot()["stages"]["triage"]

    assert stage["count"] == 100
    assert stage["p50_ms"] <= stage["p95_ms"]
    assert stage["p95_ms"] >= 90


def test_a_stage_with_one_sample_reports_it() -> None:
    counters = Counters()
    counters.observe("report", 42.0)

    assert counters.snapshot()["stages"]["report"]["p50_ms"] == 42.0


def test_an_empty_snapshot_has_no_stages() -> None:
    assert Counters().snapshot() == {"events": {}, "stages": {}}
