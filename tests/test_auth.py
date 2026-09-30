"""Keys, the names they stand for, and when they stop working."""

from datetime import UTC, datetime

import pytest

from drdoom.api.auth import KEYS_ENV, KeyRing

NEW_YEAR = datetime(2030, 1, 1, tzinfo=UTC)
JUST_BEFORE = datetime(2029, 12, 31, 23, 59, 59, tzinfo=UTC)


def ring_from(monkeypatch, value: str) -> KeyRing:
    monkeypatch.setenv(KEYS_ENV, value)
    return KeyRing.from_environment()


def test_a_key_without_a_date_does_not_expire(monkeypatch) -> None:
    ring = ring_from(monkeypatch, "ops:abc123")

    assert ring.resolve("abc123", now=datetime(2100, 1, 1, tzinfo=UTC)).name == "ops"
    assert ring.usable() == 1


def test_a_key_stops_working_at_the_start_of_its_expiry_date(monkeypatch) -> None:
    ring = ring_from(monkeypatch, "ops:abc123:2030-01-01")

    assert ring.resolve("abc123", now=JUST_BEFORE).name == "ops"
    assert ring.resolve("abc123", now=NEW_YEAR) is None
    assert ring.usable(JUST_BEFORE) == 1
    assert ring.usable(NEW_YEAR) == 0


def test_keys_with_and_without_dates_mix(monkeypatch) -> None:
    ring = ring_from(monkeypatch, "ops:abc123:2030-01-01, oncall:def456")

    assert ring.usable(NEW_YEAR) == 1
    assert ring.resolve("def456", now=NEW_YEAR).name == "oncall"


def test_a_key_may_contain_a_colon(monkeypatch) -> None:
    ring = ring_from(monkeypatch, "ops:ab:cd")

    assert ring.resolve("ab:cd").name == "ops"


def test_an_impossible_date_stops_start_up(monkeypatch) -> None:
    """A typo must not quietly turn into part of the key and a key that never expires."""
    with pytest.raises(ValueError, match="not a date"):
        ring_from(monkeypatch, "ops:abc123:2030-13-01")


def test_a_wrong_key_is_refused_whatever_its_date(monkeypatch) -> None:
    ring = ring_from(monkeypatch, "ops:abc123:2030-01-01")

    assert ring.resolve("abc12", now=JUST_BEFORE) is None
    assert ring.resolve(None) is None
