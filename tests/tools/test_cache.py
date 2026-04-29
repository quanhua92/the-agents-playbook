"""Tests for tools/cache.py — ToolResultCache."""

import time

from the_agents_playbook.tools.cache import ToolResultCache
from the_agents_playbook.tools.protocol import ToolResult


def test_cache_hit_and_miss():
    cache = ToolResultCache(default_ttl=10.0)
    result = ToolResult(output="cached value")

    assert cache.get("tool", {"key": "a"}) is None

    cache.set("tool", {"key": "a"}, result)
    hit = cache.get("tool", {"key": "a"})

    assert hit is not None
    assert hit.output == "cached value"


def test_cache_miss_different_args():
    cache = ToolResultCache(default_ttl=10.0)
    cache.set("tool", {"key": "a"}, ToolResult(output="a"))

    assert cache.get("tool", {"key": "b"}) is None


def test_cache_args_order_invariant():
    """Same arguments in different order should produce the same cache key."""
    cache = ToolResultCache(default_ttl=10.0)
    cache.set("tool", {"a": 1, "b": 2}, ToolResult(output="value"))

    hit = cache.get("tool", {"b": 2, "a": 1})
    assert hit is not None
    assert hit.output == "value"


def test_cache_ttl_expiry():
    cache = ToolResultCache(default_ttl=0.1)
    cache.set("tool", {"key": "x"}, ToolResult(output="expires"))

    assert cache.get("tool", {"key": "x"}) is not None

    time.sleep(0.15)

    assert cache.get("tool", {"key": "x"}) is None


def test_cache_per_entry_ttl_override():
    cache = ToolResultCache(default_ttl=10.0)
    cache.set("tool", {"key": "short"}, ToolResult(output="short"), ttl=0.1)

    assert cache.get("tool", {"key": "short"}) is not None
    time.sleep(0.15)
    assert cache.get("tool", {"key": "short"}) is None


def test_evict_expired():
    cache = ToolResultCache(default_ttl=0.1)
    cache.set("tool", {"key": "a"}, ToolResult(output="a"))
    cache.set("tool", {"key": "b"}, ToolResult(output="b"))

    time.sleep(0.15)

    evicted = cache.evict_expired()
    assert evicted == 2
    assert cache.size == 0


def test_evict_expired_nothing_expired():
    cache = ToolResultCache(default_ttl=10.0)
    cache.set("tool", {"key": "a"}, ToolResult(output="a"))

    evicted = cache.evict_expired()
    assert evicted == 0
    assert cache.size == 1


def test_clear():
    cache = ToolResultCache(default_ttl=10.0)
    cache.set("tool", {"key": "a"}, ToolResult(output="a"))
    cache.set("tool", {"key": "b"}, ToolResult(output="b"))
    assert cache.size == 2

    cache.clear()
    assert cache.size == 0


def test_size():
    cache = ToolResultCache()
    assert cache.size == 0

    cache.set("tool", {"key": "a"}, ToolResult(output="a"))
    assert cache.size == 1

    cache.set("tool", {"key": "b"}, ToolResult(output="b"))
    assert cache.size == 2


def test_different_tools_same_args_dont_collide():
    """Same args on different tools should be separate cache entries."""
    cache = ToolResultCache(default_ttl=10.0)
    cache.set("tool_a", {"key": "x"}, ToolResult(output="from_a"))
    cache.set("tool_b", {"key": "x"}, ToolResult(output="from_b"))

    assert cache.get("tool_a", {"key": "x"}).output == "from_a"
    assert cache.get("tool_b", {"key": "x"}).output == "from_b"
