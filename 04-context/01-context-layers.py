"""01-context-layers.py — Create layers, set priorities, sort.

Demonstrates the ContextLayer dataclass and LayerPriority enum.
Layers sort by priority first (STATIC → SEMI_STABLE → DYNAMIC),
then by order within each priority.
"""

import asyncio

from the_agents_playbook.context import ContextLayer, LayerPriority


async def main():
    # Create layers at different priorities
    system_rules = ContextLayer(
        name="system_rules",
        content="You are a helpful coding assistant. Always explain your reasoning.",
        priority=LayerPriority.STATIC,
        order=0,
    )

    tool_definitions = ContextLayer(
        name="tool_definitions",
        content="Available tools: shell, file_read, file_write, web_search",
        priority=LayerPriority.STATIC,
        order=1,
    )

    memory_summary = ContextLayer(
        name="memory_summary",
        content="User prefers Python. Project uses FastAPI.",
        priority=LayerPriority.SEMI_STABLE,
        order=0,
    )

    user_preferences = ContextLayer(
        name="user_preferences",
        content="Response language: English. Detail level: verbose.",
        priority=LayerPriority.SEMI_STABLE,
        order=1,
    )

    git_status = ContextLayer(
        name="git_status",
        content="Branch: main. Working tree: 3 files modified.",
        priority=LayerPriority.DYNAMIC,
        order=0,
    )

    current_date = ContextLayer(
        name="current_date",
        content="Date: 2025-01-15. Time: 14:30 UTC.",
        priority=LayerPriority.DYNAMIC,
        order=1,
    )

    layers = [git_status, system_rules, current_date, tool_definitions,
              memory_summary, user_preferences]

    # Sorting puts them in the correct assembly order
    print("=== Before sorting ===")
    for layer in layers:
        print(f"  {layer.priority.name:12s} order={layer.order} {layer.name}")

    layers.sort()

    print("\n=== After sorting (prompt assembly order) ===")
    for layer in layers:
        print(f"  {layer.priority.name:12s} order={layer.order} {layer.name}")

    # Comparison operators work between layers
    assert system_rules < memory_summary  # STATIC < SEMI_STABLE
    assert tool_definitions < git_status   # STATIC < DYNAMIC
    assert memory_summary < current_date   # SEMI_STABLE < DYNAMIC

    # Within same priority, order breaks ties
    same_priority = ContextLayer(name="a", content="a", priority=LayerPriority.STATIC, order=2)
    assert tool_definitions < same_priority  # order=1 < order=2

    print("\n✓ Layer sorting works correctly")


if __name__ == "__main__":
    asyncio.run(main())
