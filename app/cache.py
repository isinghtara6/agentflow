"""Redis cache with single-flight: identical concurrent requests trigger ONE workflow run."""
import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from .metrics import CACHE

log = logging.getLogger(__name__)

_UNLOCK = "if redis.call('get',KEYS[1])==ARGV[1] then return redis.call('del',KEYS[1]) else return 0 end"


class Cache:
    def __init__(self, redis: Redis, ttl_s: int):
        self.redis, self.ttl_s = redis, ttl_s
        self._unlock = redis.register_script(_UNLOCK)

    @staticmethod
    def make_key(namespace: str, payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return f"cache:{namespace}:{hashlib.sha256(blob.encode()).hexdigest()}"

    async def get(self, key: str) -> Any | None:
        try:
            raw = await self.redis.get(key)
            return json.loads(raw) if raw else None
        except (RedisError, ValueError) as e:
            log.error("cache get failed: %s", e)
            return None

    async def set(self, key: str, value: Any) -> None:
        try:
            await self.redis.set(key, json.dumps(value), ex=self.ttl_s)
        except RedisError as e:
            log.error("cache set failed: %s", e)

    async def _try_lock(self, lock_key: str, token: str, ttl: int) -> bool:
        try:
            return bool(await self.redis.set(lock_key, token, nx=True, ex=ttl))
        except RedisError:
            return True  # degrade: compute without coordination

    async def get_or_compute(
        self, key: str, compute: Callable[[], Awaitable[Any]], lock_ttl_s: int = 170
    ) -> tuple[Any, bool]:
        """Returns (value, cache_hit)."""
        v = await self.get(key)
        if v is not None:
            CACHE.labels("hit").inc()
            return v, True

        lock_key, token = f"lock:{key}", uuid.uuid4().hex
        if await self._try_lock(lock_key, token, lock_ttl_s):
            CACHE.labels("miss").inc()
            try:
                value = await compute()
                await self.set(key, value)
                return value, False
            finally:
                try:
                    await self._unlock(keys=[lock_key], args=[token])
                except RedisError:
                    pass

        # follower: wait for the leader to populate the cache
        deadline = time.monotonic() + lock_ttl_s
        while time.monotonic() < deadline:
            await asyncio.sleep(0.25)
            v = await self.get(key)
            if v is not None:
                CACHE.labels("coalesced").inc()
                return v, True
            try:
                if not await self.redis.exists(lock_key):
                    break  # leader failed; take over
            except RedisError:
                break
        CACHE.labels("miss").inc()
        value = await compute()
        await self.set(key, value)
        return value, False
