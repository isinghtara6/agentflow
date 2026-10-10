"""Distributed token-bucket rate limiter (atomic Lua on Redis) + daily token quota.

One bucket per key. Used for:
  * per-user request limits at the API edge
  * a GLOBAL bucket that every LLM call waits on, so N pods never exceed your Anthropic limits
"""
import asyncio
import logging
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from redis.asyncio import Redis
from redis.exceptions import RedisError

log = logging.getLogger(__name__)

_LUA = """
local key      = KEYS[1]
local rate     = tonumber(ARGV[1])   -- tokens per second
local capacity = tonumber(ARGV[2])
local cost     = tonumber(ARGV[3])
local t   = redis.call('TIME')
local now = t[1] * 1000 + math.floor(t[2] / 1000)
local d = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(d[1])
local ts = tonumber(d[2])
if tokens == nil then tokens = capacity; ts = now end
tokens = math.min(capacity, tokens + (math.max(0, now - ts) / 1000) * rate)
local allowed = 0
local retry_ms = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  retry_ms = math.ceil((cost - tokens) / rate * 1000)
end
redis.call('HSET', key, 'tokens', tostring(tokens), 'ts', now)
redis.call('PEXPIRE', key, math.ceil(capacity / rate * 1000) * 2 + 1000)
return {allowed, tostring(tokens), retry_ms}
"""


class RateLimited(Exception):
    def __init__(self, retry_after_s: float, scope: str = "user"):
        super().__init__(f"rate limited ({scope}); retry in {retry_after_s:.1f}s")
        self.retry_after_s = retry_after_s
        self.scope = scope


@dataclass
class RateResult:
    allowed: bool
    remaining: float
    retry_after_s: float
    limit: int


class RateLimiter:
    def __init__(self, redis: Redis, fail_open: bool = True):
        self.redis = redis
        self.fail_open = fail_open
        self._script = redis.register_script(_LUA)

    async def check(self, key: str, rpm: int, burst: int, cost: int = 1) -> RateResult:
        try:
            allowed, tokens, retry_ms = await self._script(keys=[key], args=[rpm / 60.0, burst, cost])
            return RateResult(bool(int(allowed)), float(tokens), int(retry_ms) / 1000.0, burst)
        except RedisError as e:
            log.error("rate limiter redis error: %s", e)
            if self.fail_open:
                return RateResult(True, float(burst), 0.0, burst)
            return RateResult(False, 0.0, 1.0, burst)

    async def wait(self, key: str, rpm: int, burst: int, max_wait_s: float = 15.0) -> None:
        """Block until a token is available (smooths bursts instead of failing)."""
        deadline = time.monotonic() + max_wait_s
        while True:
            r = await self.check(key, rpm, burst)
            if r.allowed:
                return
            if time.monotonic() + r.retry_after_s > deadline:
                raise RateLimited(r.retry_after_s, scope="global")
            await asyncio.sleep(r.retry_after_s + random.uniform(0, 0.05))


class DailyQuota:
    """Per-user token quota per UTC day."""

    def __init__(self, redis: Redis, limit: int):
        self.redis, self.limit = redis, limit

    def _key(self, user: str) -> str:
        return f"quota:{user}:{datetime.now(UTC):%Y%m%d}"

    async def remaining(self, user: str) -> int:
        try:
            used = int(await self.redis.get(self._key(user)) or 0)
        except RedisError:
            return self.limit
        return max(0, self.limit - used)

    async def add(self, user: str, tokens: int) -> None:
        try:
            k = self._key(user)
            pipe = self.redis.pipeline()
            pipe.incrby(k, tokens)
            pipe.expire(k, 172_800)
            await pipe.execute()
        except RedisError as e:
            log.error("quota add failed: %s", e)
