"""02-graceful-degradation.py — Fallback behavior when tools/LLM fail.

DegradationManager provides graceful fallbacks for common failure
modes so the agent can continue operating in limited capacity.
"""

import asyncio

from the_agents_playbook.claw.degradation import DegradationManager


async def main():
    mgr = DegradationManager()

    # --- Tool failure ---

    print("=== Tool Failure (default) ===")
    result = await mgr.handle_tool_failure("shell", RuntimeError("permission denied"))
    print(f"  Handled:  {result.handled}")
    print(f"  Strategy: {result.strategy}")
    print(f"  Output:   {result.output}")
    print()

    # --- Registered fallback ---

    print("=== Registered Fallback ===")
    mgr.register_tool_fallback("deploy", "Rolling back to previous version.")
    result = await mgr.handle_tool_failure("deploy", RuntimeError("500"))
    print(f"  Output: {result.output}")
    print()

    # --- LLM failure ---

    print("=== LLM Failure ===")
    result = await mgr.handle_llm_failure(ConnectionError("connection refused"))
    print(f"  Strategy: {result.strategy}")
    print(f"  Output:   {result.output}")
    print()

    # --- Context overflow ---

    print("=== Context Overflow ===")
    result = await mgr.handle_context_overflow(max_tokens=4096)
    print(f"  Strategy: {result.strategy}")
    print(f"  Output:   {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
