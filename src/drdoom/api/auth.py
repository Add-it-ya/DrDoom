"""Who is allowed to approve, and under what name.

A predecessor project put no authentication on its approval endpoint: the only thing
between the open internet and approving a high-risk production action was guessing an
eight-character identifier. For a project whose whole argument is the approval gate, that
was the contradiction worth fixing first.

Keys map to a named principal rather than to a boolean, because the audit log needs to
record *who* decided, and "someone with a valid key" is not an answer a review accepts.

Keys are compared with a constant-time comparison. The timing signal on a short string is
tiny, but writing the comparison correctly costs nothing and writing it wrongly is the
kind of detail that ends up in a security review.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

HEADER_NAME = "X-API-Key"
KEYS_ENV = "DRDOOM_API_KEYS"

api_key_header = APIKeyHeader(name=HEADER_NAME, auto_error=False)


@dataclass(frozen=True)
class Principal:
    """An authenticated caller, named for the audit log."""

    name: str


class KeyRing:
    """The keys this deployment accepts, mapped to the names they authenticate as.

    A key may carry an expiry date, after which it is refused like any wrong key. A static
    key that never expires stays valid for as long as it sits in someone's shell history.
    """

    def __init__(
        self, keys: dict[str, str] | None = None, expires: dict[str, datetime] | None = None
    ) -> None:
        self.keys = dict(keys or {})
        self.expires = dict(expires or {})

    @classmethod
    def from_environment(cls) -> KeyRing:
        """Read ``name:key`` or ``name:key:YYYY-MM-DD`` entries from the environment.

        Entries are comma separated. A date is when the key stops working, at the start of
        that day in UTC. Something shaped like a date that is not one stops start-up
        rather than quietly becoming part of the key.

        An empty key ring means no caller can approve anything. That is the correct
        default: a deployment that forgot to configure credentials should refuse
        approvals, not accept them from anyone.
        """
        raw = os.environ.get(KEYS_ENV, "").strip()
        if not raw:
            logger.warning("%s is unset, so no caller can approve a remediation", KEYS_ENV)
            return cls({})

        keys: dict[str, str] = {}
        expires: dict[str, datetime] = {}
        for entry in raw.split(","):
            name, _, rest = entry.partition(":")
            name, rest = name.strip(), rest.strip()
            key, expiry = rest, None
            head, _, tail = rest.rpartition(":")
            if DATE.fullmatch(tail.strip()):
                key, expiry = head.strip(), _parse_expiry(tail.strip(), name)
            if name and key:
                keys[key] = name
                if expiry is not None:
                    expires[key] = expiry
        ring = cls(keys, expires)
        now = datetime.now(UTC)
        for key, when in expires.items():
            state = "expired" if when <= now else "expires"
            logger.warning("the key for %s %s on %s", keys[key], state, when.date())
        logger.info("loaded %d approval credential(s), %d usable now", len(keys), ring.usable(now))
        return ring

    def resolve(self, presented: str | None, now: datetime | None = None) -> Principal | None:
        if not presented:
            return None
        for key, name in self.keys.items():
            if secrets.compare_digest(key, presented):
                expiry = self.expires.get(key)
                if expiry is not None and expiry <= (now or datetime.now(UTC)):
                    logger.warning("refused the expired key for %s", name)
                    return None
                return Principal(name=name)
        return None

    def usable(self, now: datetime | None = None) -> int:
        """How many keys would be accepted at ``now``."""
        moment = now or datetime.now(UTC)
        return sum(1 for key in self.keys if key not in self.expires or self.expires[key] > moment)

    def __len__(self) -> int:
        return len(self.keys)


DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _parse_expiry(text: str, name: str) -> datetime:
    try:
        return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError(f"the key for {name} has an expiry that is not a date: {text}") from error


_keyring = KeyRing()


def configure(keyring: KeyRing) -> None:
    """Install the key ring the application will authenticate against."""
    global _keyring
    _keyring = keyring


def current_keyring() -> KeyRing:
    return _keyring


PresentedKey = Annotated[str | None, Security(api_key_header)]


def require_principal(presented: PresentedKey) -> Principal:
    """Authenticate a request for a protected endpoint, or refuse it."""
    principal = _keyring.resolve(presented)
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"a valid {HEADER_NAME} is required",
            headers={"WWW-Authenticate": HEADER_NAME},
        )
    return principal
