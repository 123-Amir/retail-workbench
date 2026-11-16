"""Retry with exponential backoff and a circuit breaker. `sleep` and `clock` are injectable so tests never wait."""
import time

from app.llm.base import LLMError, LLMRateLimit


def call_with_retry(fn, *, retries: int = 2, base_delay: float = 0.5, max_delay: float = 5.0, sleep=time.sleep, on_retry=None):
    attempt = 0
    while True:
        try:
            return fn()
        except LLMError as e:
            if not e.retryable or attempt >= retries:
                raise
            delay = min(max_delay, e.retry_after if isinstance(e, LLMRateLimit) and e.retry_after else base_delay * 2 ** attempt)
            if on_retry:
                on_retry(attempt + 1, e, delay)
            sleep(delay)
            attempt += 1


class CircuitBreaker:
    """closed -> (threshold consecutive failures) -> open for `cooldown` seconds -> half-open (one trial call)."""

    def __init__(self, threshold: int = 3, cooldown: float = 30.0, clock=time.monotonic):
        self.threshold, self.cooldown, self.clock = threshold, cooldown, clock
        self.failures, self.opened_at = 0, None

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "half_open" if self.clock() - self.opened_at >= self.cooldown else "open"

    def allow(self) -> bool:
        return self.state != "open"

    def success(self):
        self.failures, self.opened_at = 0, None

    def failure(self):
        self.failures += 1
        if self.failures >= self.threshold or self.state == "half_open":
            self.opened_at = self.clock()
