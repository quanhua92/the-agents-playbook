"""05-tool-caching.py — Cache tool results with TTL to avoid redundant work.

The ToolResultCache stores (tool_name, arguments_hash) → ToolResult with a
configurable TTL. Expired entries are evicted on access. This is useful for
tools that return the same result for the same input (e.g., web search,
file reads).
"""

import asyncio
from typing import Any

from the_agents_playbook.tools import Tool, ToolResult, ToolRegistry
from the_agents_playbook.tools.cache import ToolResultCache


class SlowTool(Tool):
    """A tool that simulates a slow operation (e.g., API call)."""

    @property
    def name(self) -> str:
        return "slow_lookup"

    @property
    def description(self) -> str:
        return "Look up data with an artificial delay."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "The lookup key"},
            },
            "required": ["key"],
            "additionalProperties": False,
        }

    async def execute(self, key: str, **kwargs: Any) -> ToolResult:
        await asyncio.sleep(0.5)  # Simulate slow operation
        return ToolResult(output=f"Result for '{key}': value-{key}")


async def main():
    cache = ToolResultCache(default_ttl=2.0)
    registry = ToolRegistry()
    registry.register(SlowTool())

    key = "test-key"

    # --- First call: cache miss ---
    import time
    start = time.monotonic()
    result = await registry.dispatch("slow_lookup", {"key": key})
    elapsed = time.monotonic() - start
    print(f"Call 1 (cache miss): {result.output} ({elapsed:.2f}s)")

    # Store in cache
    cache.set("slow_lookup", {"key": key}, result)
    print(f"Cache size: {cache.size}\n")

    # --- Second call: cache hit ---
    cached = cache.get("slow_lookup", {"key": key})
    if cached:
        print(f"Call 2 (cache hit):  {cached.output} (0.00s)")
    else:
        print("Call 2: cache miss (unexpected!)")

    # --- Different arguments: cache miss ---
    cached = cache.get("slow_lookup", {"key": "other-key"})
    print(f"Call 3 (diff args):   {cached} (None = miss)\n")

    # --- TTL expiry ---
    print("Waiting for TTL expiry (2.5s)...")
    await asyncio.sleep(2.5)

    cached = cache.get("slow_lookup", {"key": key})
    print(f"Call 4 (expired):     {cached} (None = evicted)")
    print(f"Cache size after eviction: {cache.size}\n")

    # --- Bulk eviction ---
    cache.set("slow_lookup", {"key": "a"}, ToolResult(output="a"))
    cache.set("slow_lookup", {"key": "b"}, ToolResult(output="b"))
    print(f"Cache size before eviction: {cache.size}")
    evicted = cache.evict_expired()
    print(f"Evicted: {evicted}")
    print(f"Cache size after eviction:  {cache.size}")

    # --- Clear all ---
    cache.set("slow_lookup", {"key": "x"}, ToolResult(output="x"))
    cache.clear()
    print(f"Cache size after clear: {cache.size}")


asyncio.run(main())
