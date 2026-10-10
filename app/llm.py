"""Resilient LLM client: global rate-limit wait, retries w/ jitter, circuit breaker, token budget, metrics."""
import logging
import time

import anthropic
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from .config import Settings
from .metrics import LLM_CALLS, LLM_LAT, LLM_TOKENS
from .rate_limiter import RateLimiter

log = logging.getLogger(__name__)


class BudgetExceeded(Exception):
    pass


class CircuitOpen(Exception):
    pass


class RunBudget:
    """Hard cap on tokens spent by a single workflow run (stops runaway agent loops)."""

    def __init__(self, limit: int):
        self.limit, self.used = limit, 0

    def check(self) -> None:
        if self.used >= self.limit:
            raise BudgetExceeded(f"token budget {self.limit} exhausted")

    def add(self, n: int) -> None:
        self.used += n


class CircuitBreaker:
    def __init__(self, threshold: int, cooldown_s: float):
        self.threshold, self.cooldown_s = threshold, cooldown_s
        self.failures, self.opened_at = 0, 0.0

    def allow(self) -> bool:
        if self.failures < self.threshold:
            return True
        if time.monotonic() - self.opened_at >= self.cooldown_s:
            return True  # half-open: let one probe through
        return False

    def success(self) -> None:
        self.failures = 0

    def failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = time.monotonic()


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APITimeoutError)):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code >= 500 or exc.status_code == 429
    return False


class LLM:
    def __init__(self, settings: Settings, limiter: RateLimiter, client: anthropic.AsyncAnthropic | None = None):
        self.s, self.limiter = settings, limiter
        # max_retries=0: tenacity owns retries so every attempt goes through the global limiter
        self.client = client or anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key, timeout=settings.llm_timeout_s, max_retries=0
        )
        self.breaker = CircuitBreaker(settings.breaker_threshold, settings.breaker_cooldown_s)

    async def complete(
        self, *, system: str, user: str, model: str, agent: str,
        max_tokens: int = 1024, budget: RunBudget | None = None,
    ) -> str:
        if budget:
            budget.check()
        if not self.breaker.allow():
            raise CircuitOpen("LLM circuit open")

        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception(_retryable),
                stop=stop_after_attempt(self.s.llm_max_retries),
                wait=wait_random_exponential(multiplier=1, max=20),
                reraise=True,
            ):
                with attempt:
                    await self.limiter.wait("rl:global:llm", self.s.global_llm_rpm, self.s.global_llm_burst)
                    t0 = time.monotonic()
                    resp = await self.client.messages.create(
                        model=model, max_tokens=max_tokens, system=system,
                        messages=[{"role": "user", "content": user}],
                    )
                    LLM_LAT.labels(agent).observe(time.monotonic() - t0)
        except Exception:
            self.breaker.failure()
            LLM_CALLS.labels(agent, model, "error").inc()
            raise

        self.breaker.success()
        LLM_CALLS.labels(agent, model, "ok").inc()
        tin, tout = resp.usage.input_tokens, resp.usage.output_tokens
        LLM_TOKENS.labels(model, "in").inc(tin)
        LLM_TOKENS.labels(model, "out").inc(tout)
        if budget:
            budget.add(tin + tout)
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
