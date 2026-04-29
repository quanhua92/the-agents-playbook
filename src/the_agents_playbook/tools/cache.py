"""ToolResultCache — dict-based TTL cache for tool results."""

import hashlib
import json
import logging
from time import monotonic
from typing import Any

from .protocol import ToolResult

logger = logging.getLogger(__name__)


class ToolResultCache:
    """Cache tool results with TTL eviction.

    Uses a dict keyed by (tool_name, arguments_hash) with monotonic
    timestamps for TTL tracking. No external dependencies.

    Usage:
        cache = ToolResultCache(default_ttl=60.0)
        cache.set("web_search", {"query": "python"}, result)
        cached = cache.get("web_search", {"query": "python"})
    """

    def __init__(self, default_ttl: float = 60.0) -> None:
        self._store: dict[str, tuple[ToolResult, float]] = {}
        self._default_ttl = default_ttl

    def _make_key(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Create a cache key from tool name and argument hash."""
        args_str = json.dumps(arguments, sort_keys=True)
        args_hash = hashlib.sha256(args_str.encode()).hexdigest()[:16]
        return f"{tool_name}:{args_hash}"

    def get(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult | None:
        """Retrieve a cached result. Returns None on cache miss or expired entry."""
        key = self._make_key(tool_name, arguments)
        entry = self._store.get(key)

        if entry is None:
            return None

        result, timestamp = entry
        age = monotonic() - timestamp

        if age > self._default_ttl:
            del self._store[key]
            logger.debug("Cache expired for %s (age=%.1fs)", key, age)
            return None

        logger.debug("Cache hit for %s", key)
        return result

    def set(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: ToolResult,
        ttl: float | None = None,
    ) -> None:
        """Store a result in the cache with an optional per-entry TTL override."""
        key = self._make_key(tool_name, arguments)
        self._store[key] = (result, monotonic())
        logger.debug(
            "Cached result for %s (ttl=%.1fs)", key, ttl or self._default_ttl
        )

    def evict_expired(self) -> int:
        """Remove all expired entries. Returns the number evicted."""
        now = monotonic()
        expired_keys = [
            key
            for key, (_, timestamp) in self._store.items()
            if now - timestamp > self._default_ttl
        ]
        for key in expired_keys:
            del self._store[key]
        if expired_keys:
            logger.debug("Evicted %d expired cache entries", len(expired_keys))
        return len(expired_keys)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._store.clear()
        logger.debug("Cache cleared")

    @property
    def size(self) -> int:
        """Current number of entries (including possibly expired ones)."""
        return len(self._store)
