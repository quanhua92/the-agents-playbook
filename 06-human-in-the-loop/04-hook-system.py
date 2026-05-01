"""04-hook-system.py — Register hooks, emit events, observe agent decisions.

HookSystem fires at key points in the agent lifecycle, enabling logging,
monitoring, and audit trails without modifying the agent core.
"""

import asyncio

from the_agents_playbook.guardrails import (
    ON_TOOL_CALL,
    ON_TOOL_RESULT,
    ON_TURN_END,
    ON_TURN_START,
    HookSystem,
)


async def main():
    hooks = HookSystem()

    # --- Collect hook calls for inspection ---

    log: list[dict] = []

    async def log_handler(event_name: str, **kwargs):
        log.append({"event": event_name, **kwargs})

    # --- Register hooks on standard events ---

    hooks.on(ON_TURN_START, lambda **kw: log_handler("turn_start", **kw))
    hooks.on(ON_TOOL_CALL, lambda **kw: log_handler("tool_call", **kw))
    hooks.on(ON_TOOL_RESULT, lambda **kw: log_handler("tool_result", **kw))
    hooks.on(ON_TURN_END, lambda **kw: log_handler("turn_end", **kw))
    print(
        f"Registered {sum(len(hooks.handlers(e)) for e in [ON_TURN_START, ON_TOOL_CALL, ON_TOOL_RESULT, ON_TURN_END])} hooks"
    )
    print()

    # --- Simulate an agent turn ---

    print("=== Simulated Agent Turn ===")

    await hooks.emit(ON_TURN_START, prompt="Fix the bug in auth.py")
    print("  Turn started")

    await hooks.emit(
        ON_TOOL_CALL, tool_name="shell", arguments={"command": "grep -n 'bug' auth.py"}
    )
    print("  Tool called: shell")

    await hooks.emit(
        ON_TOOL_RESULT,
        tool_name="shell",
        output="auth.py:42: # bug here",
        error=False,
        latency_ms=15.2,
    )
    print("  Tool result: OK (15.2ms)")

    await hooks.emit(
        ON_TURN_END, response="Found the bug at line 42", total_tokens=450, tool_calls=1
    )
    print("  Turn ended")
    print()

    # --- Inspect logged events ---

    print("=== Hook Log ===")
    for entry in log:
        print(f"  [{entry['event']:12s}] ", end="")
        for k, v in entry.items():
            if k != "event":
                print(f"{k}={v} ", end="")
        print()

    # --- Multiple handlers per event ---

    print("\n=== Multiple Handlers ===")
    count = 0

    async def counter(**kw):
        nonlocal count
        count += 1

    hooks.on("custom_event", counter)
    hooks.on("custom_event", counter)
    hooks.on("custom_event", counter)
    await hooks.emit("custom_event")
    print(f"  3 handlers on 'custom_event' → counter = {count}")

    # --- Handler errors don't stop others ---

    print("\n=== Error Isolation ===")
    good_called = False

    async def bad_handler(**kw):
        raise RuntimeError("oops")

    async def good_handler(**kw):
        nonlocal good_called
        good_called = True

    hooks.on("error_test", bad_handler)
    hooks.on("error_test", good_handler)
    await hooks.emit("error_test")
    print(f"  Bad handler raised, good handler still called: {good_called}")

    # --- Clean up ---

    hooks.clear()
    print(f"\n  After clear(): {len(hooks.handlers(ON_TOOL_CALL))} handlers")


if __name__ == "__main__":
    asyncio.run(main())
