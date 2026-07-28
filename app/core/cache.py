"""
Layered caching: an in-process TTL cache (`cachetools.TTLCache`) that always
works with zero external dependencies, plus an optional Redis-backed layer
used automatically when `CACHE_BACKEND=redis` and a reachable `REDIS_URL` is
configured. If Redis is configured but unreachable, the cache silently and
permanently falls back to the in-process cache for the rest of the process
lifetime (logged once), so the application never crashes because of a cache
backend outage.

Cache keys MUST include the owning user's id (or "system" for admin/shared
data) so that no cached value can ever leak across users.
"""
import json
import hashlib
from functools import lru_cache
from typing import Any, Optional

from cachetools import TTLCache

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def make_cache_key(namespace: str, user_id: Optional[str], **params: Any) -> str:
    """Builds a deterministic, user-scoped cache key from a namespace and parameters."""
    owner = user_id or "system"
    serialized = json.dumps(params, sort_keys=True, default=str)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]
    return f"{namespace}:{owner}:{digest}"


class CacheManager:
    def __init__(self):
        settings = get_settings()
        self.ttl_seconds = settings.CACHE_TTL_SECONDS
        self._memory_cache: TTLCache = TTLCache(maxsize=2048, ttl=self.ttl_seconds)
        self._redis_client = None
        self._redis_attempted = False
        self._backend_mode = "memory"

        if settings.CACHE_BACKEND == "redis":
            self._try_connect_redis(settings.REDIS_URL)

    def _try_connect_redis(self, redis_url: str) -> None:
        self._redis_attempted = True
        try:
            import redis
            client = redis.from_url(redis_url, socket_connect_timeout=1, socket_timeout=1)
            client.ping()
            self._redis_client = client
            self._backend_mode = "redis"
            logger.info("Cache backend: Redis (%s)", redis_url)
        except Exception as exc:
            logger.warning("Redis unreachable at %s (%s); falling back to in-memory cache.", redis_url, exc)
            self._redis_client = None
            self._backend_mode = "memory"

    @property
    def backend_mode(self) -> str:
        return self._backend_mode

    def get(self, key: str) -> Optional[Any]:
        if self._redis_client is not None:
            try:
                raw = self._redis_client.get(key)
                return json.loads(raw) if raw is not None else None
            except Exception:
                logger.warning("Redis GET failed; falling back to memory cache for this key.", exc_info=True)
        return self._memory_cache.get(key)

    def set(self, key: str, value: Any) -> None:
        if self._redis_client is not None:
            try:
                self._redis_client.setex(key, self.ttl_seconds, json.dumps(value, default=str))
                return
            except Exception:
                logger.warning("Redis SET failed; falling back to memory cache for this key.", exc_info=True)
        self._memory_cache[key] = value

    def delete_prefix(self, prefix: str) -> int:
        """Invalidates every cache entry whose key starts with the given prefix (e.g. 'summary:<user_id>')."""
        count = 0
        # In-process cache
        for key in list(self._memory_cache.keys()):
            if key.startswith(prefix):
                del self._memory_cache[key]
                count += 1
        # Redis (best-effort scan; skipped gracefully if unavailable)
        if self._redis_client is not None:
            try:
                for key in self._redis_client.scan_iter(match=f"{prefix}*"):
                    self._redis_client.delete(key)
                    count += 1
            except Exception:
                logger.warning("Redis prefix invalidation failed.", exc_info=True)
        return count

    def diagnostics(self) -> dict:
        return {
            "backend_mode": self._backend_mode,
            "ttl_seconds": self.ttl_seconds,
            "memory_cache_size": len(self._memory_cache),
            "memory_cache_maxsize": self._memory_cache.maxsize,
        }


@lru_cache
def get_cache_manager() -> CacheManager:
    return CacheManager()
