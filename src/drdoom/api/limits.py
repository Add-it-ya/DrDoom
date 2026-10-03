"""How many investigations may be started, by whom, per minute.

Starting an investigation is deliberately open, so the demo can be clicked, and each one
costs several thousand model tokens. Without a limit anyone who found the service could
spend its whole provider quota. Two sliding windows guard it:

- **per caller**, where a caller is its API key when it sends a valid one and otherwise
  its network address, so one client cannot crowd out the rest;
- **in total**, so that many addresses together still cannot drain the quota.

A refused request is not recorded, so the callers tracked at any moment are at most the
requests admitted in the last window. Memory is bounded by the total limit, not by how
many addresses an attacker has.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable


class RateLimiter:
    """Sliding-window limits on one kind of request, per caller and overall."""

    def __init__(
        self,
        per_caller: int,
        total: int,
        window: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.per_caller = per_caller
        self.total = total
        self.window = window
        self.clock = clock
        self._callers: dict[str, deque[float]] = {}
        self._all: deque[float] = deque()
        self._lock = threading.Lock()

    def admit(self, caller: str) -> float | None:
        """Record one request and return None, or return the seconds until one is allowed.

        A limit of zero is no limit.
        """
        with self._lock:
            now = self.clock()
            self._expire(now)
            mine = self._callers.get(caller, deque())
            waits = []
            if self.per_caller and len(mine) >= self.per_caller:
                waits.append(mine[0] + self.window - now)
            if self.total and len(self._all) >= self.total:
                waits.append(self._all[0] + self.window - now)
            if waits:
                return max(waits)
            self._callers.setdefault(caller, mine).append(now)
            self._all.append(now)
            return None

    def tracked(self) -> int:
        """How many callers have requests inside the current window."""
        with self._lock:
            self._expire(self.clock())
            return len(self._callers)

    def _expire(self, now: float) -> None:
        cutoff = now - self.window
        while self._all and self._all[0] <= cutoff:
            self._all.popleft()
        for caller in list(self._callers):
            times = self._callers[caller]
            while times and times[0] <= cutoff:
                times.popleft()
            if not times:
                del self._callers[caller]
