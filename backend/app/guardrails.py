"""Spending guardrails for the public demo: daily budget, per-IP hourly rate, concurrency."""

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import UTC, datetime


class DemoGuard:
    def __init__(
        self,
        daily_budget: int,
        hourly_per_ip: int,
        max_concurrent: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.daily_budget = daily_budget
        self.hourly_per_ip = hourly_per_ip
        self._clock = clock
        self._lock = threading.Lock()
        self._slots = threading.Semaphore(max_concurrent)
        self._budget_day = ""
        self._budget_used = 0
        self._ip_hits: dict[str, deque[float]] = defaultdict(deque)

    def acquire(self, ip: str) -> str | None:
        """Reserve a slot and one question. Returns None when allowed, else why it was refused."""
        if not self._slots.acquire(blocking=False):
            return "The demo is busy answering other questions. Give it a few seconds."
        refusal = self._claim_question(ip)
        if refusal:
            self._slots.release()
        return refusal

    def release(self) -> None:
        self._slots.release()

    def _claim_question(self, ip: str) -> str | None:
        now = self._clock()
        today = datetime.fromtimestamp(now, tz=UTC).strftime("%Y-%m-%d")
        with self._lock:
            if today != self._budget_day:
                self._budget_day, self._budget_used = today, 0
                self._ip_hits.clear()
            if self._budget_used >= self.daily_budget:
                return (
                    "This demo has used up its question budget for today. It answers on a "
                    "personal Mistral API key, so daily spending is capped. Try again tomorrow."
                )
            hits = self._ip_hits[ip]
            while hits and now - hits[0] > 3600:
                hits.popleft()
            if len(hits) >= self.hourly_per_ip:
                return (
                    f"That is {self.hourly_per_ip} questions within an hour from this address, "
                    "which is the demo limit. Try again a little later."
                )
            hits.append(now)
            self._budget_used += 1
            return None
