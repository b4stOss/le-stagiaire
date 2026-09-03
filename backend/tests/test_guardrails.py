from app.guardrails import DemoGuard


class Clock:
    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def _guard(clock: Clock, **overrides) -> DemoGuard:
    params = {"daily_budget": 5, "hourly_per_ip": 2, "max_concurrent": 1}
    params.update(overrides)
    return DemoGuard(clock=clock, **params)


def test_one_slot_at_a_time():
    guard = _guard(Clock())
    assert guard.acquire("a") is None
    assert "busy" in (guard.acquire("b") or "")
    guard.release()
    assert guard.acquire("b") is None


def test_hourly_limit_per_ip_slides_with_time():
    clock = Clock()
    guard = _guard(clock, max_concurrent=10)
    assert guard.acquire("a") is None
    assert guard.acquire("a") is None
    assert "within an hour" in (guard.acquire("a") or "")
    assert guard.acquire("b") is None  # another address is not affected
    clock.now += 3601
    assert guard.acquire("a") is None


def test_daily_budget_is_shared_and_resets_the_next_day():
    clock = Clock()
    guard = _guard(clock, max_concurrent=10, hourly_per_ip=100)
    for ip in ("a", "b", "c", "d", "e"):
        assert guard.acquire(ip) is None
    assert "budget" in (guard.acquire("f") or "")
    clock.now += 86_400
    assert guard.acquire("f") is None


def test_a_refused_question_gives_its_slot_back():
    guard = _guard(Clock(), daily_budget=0)
    assert "budget" in (guard.acquire("a") or "")
    assert "budget" in (guard.acquire("a") or "")  # not "busy": the slot was released
