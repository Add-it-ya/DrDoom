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
import secrets
from dataclasses import dataclass
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
    """The keys this deployment accepts, mapped to the names they authenticate as."""

    def __init__(self, keys: dict[str, str] | None = None) -> None:
        self.keys = dict(keys or {})

    @classmethod
    def from_environment(cls) -> KeyRing:
        """Read ``name:key`` pairs from the environment, comma separated.

        An empty key ring means no caller can approve anything. That is the correct
        default: a deployment that forgot to configure credentials should refuse
        approvals, not accept them from anyone.
        """
        raw = os.environ.get(KEYS_ENV, "").strip()
        if not raw:
            logger.warning("%s is unset, so no caller can approve a remediation", KEYS_ENV)
            return cls({})

        keys: dict[str, str] = {}
        for pair in raw.split(","):
            name, _, key = pair.partition(":")
            if name.strip() and key.strip():
                keys[key.strip()] = name.strip()
        logger.info("loaded %d approval credential(s)", len(keys))
        return cls(keys)

    def resolve(self, presented: str | None) -> Principal | None:
        if not presented:
            return None
        for key, name in self.keys.items():
            if secrets.compare_digest(key, presented):
                return Principal(name=name)
        return None

    def __len__(self) -> int:
        return len(self.keys)


_keyring = KeyRing()


def configure(keyring: KeyRing) -> None:
    """Install the key ring the application will authenticate against."""
    global _keyring
    _keyring = keyring


def current_keyring() -> KeyRing:
    return _keyring


PresentedKey = Annotated[str | None, Security(api_key_header)]


def require_principal(presented: PresentedKey) -> Principal:
    """Authenticate an approval request, or refuse it."""
    principal = _keyring.resolve(presented)
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"a valid {HEADER_NAME} is required to approve a remediation",
            headers={"WWW-Authenticate": HEADER_NAME},
        )
    return principal
