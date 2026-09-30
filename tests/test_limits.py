"""How many investigations may be started, by whom, per minute."""

from concurrent.futures import ThreadPoolExecutor

from drdoom.api.limits import RateLimiter


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_a_caller_is_admitted_up_to_its_limit_then_told_how_long_to_wait() -> None:
    clock = Clock()
    limiter = RateLimiter(per_caller=2, total=100, clock=clock)

    assert limiter.admit("a") is None
    clock.now += 10
    assert limiter.admit("a") is None
    clock.now += 5

    assert limiter.admit("a") == 45.0


def test_the_window_slides() -> None:
    clock = Clock()
    limiter = RateLimiter(per_caller=1, total=100, clock=clock)
    limiter.admit("a")

    clock.now += 60

    assert limiter.admit("a") is None


def test_callers_have_separate_allowances() -> None:
    limiter = RateLimiter(per_caller=1, total=100, clock=Clock())

    assert limiter.admit("a") is None
    assert limiter.admit("b") is None
    assert limiter.admit("a") is not None


def test_the_total_caps_every_caller_together() -> None:
    """Many addresses at once must not add up to an unlimited allowance."""
    limiter = RateLimiter(per_caller=5, total=3, clock=Clock())

    admitted = [limiter.admit(caller) is None for caller in "abcd"]

    assert admitted == [True, True, True, False]


def test_a_refused_request_does_not_extend_the_wait() -> None:
    clock = Clock()
    limiter = RateLimiter(per_caller=1, total=100, clock=clock)
    limiter.admit("a")

    clock.now += 30
    first = limiter.admit("a")
    clock.now += 10
    second = limiter.admit("a")

    assert (first, second) == (30.0, 20.0)


def test_callers_tracked_stay_within_the_total_however_many_there_are() -> None:
    clock = Clock()
    limiter = RateLimiter(per_caller=1, total=10, clock=clock)

    for n in range(1_000):
        limiter.admit(f"address-{n}")
        clock.now += 0.5

    assert limiter.tracked() <= 10


def test_zero_lifts_a_limit() -> None:
    limiter = RateLimiter(per_caller=0, total=0, clock=Clock())

    assert all(limiter.admit("a") is None for _ in range(1_000))


def test_callers_racing_are_admitted_exactly_up_to_the_limit() -> None:
    limiter = RateLimiter(per_caller=100, total=1_000, clock=Clock())

    def attempt(_: int) -> int:
        return sum(limiter.admit("a") is None for _ in range(50))

    with ThreadPoolExecutor(8) as pool:
        admitted = sum(pool.map(attempt, range(8)))

    assert admitted == 100
