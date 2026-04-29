"""01-self-repair.py — Agent encounters error, diagnoses, retries.

RepairLoop wraps tool dispatch with automatic retry on failure.
Errors are collected for post-mortem analysis.
"""

import asyncio
from unittest.mock import AsyncMock

from the_agents_playbook.claw.repair import RepairLoop, RepairResult
from the_agents_playbook.tools.protocol import ToolResult


class StubRegistry:
    def __init__(self, results=None):
        self._results = results or []
        self._index = 0

    async def dispatch(self, tool_name, arguments):
        if self._index >= len(self._results):
            raise RuntimeError("no more results")
        result = self._results[self._index]
        self._index += 1
        return result


async def main():
    # --- Successful first try ---

    print("=== Successful First Try ===")
    registry = StubRegistry([ToolResult(output="file contents")])
    loop = RepairLoop(registry, max_retries=3)
    result = await loop.repair("read_file", {"path": "main.py"})
    print(f"  Success:   {result.success}")
    print(f"  Attempts: {result.attempts}")
    print(f"  Output:    {result.final_output}")
    print()

    # --- Fails then recovers ---

    print("=== Recovers After Retry ===")
    registry = StubRegistry([
        ToolResult(output="connection refused", error=True),
        ToolResult(output="connection refused", error=True),
        ToolResult(output="ok, connected"),
    ])
    loop = RepairLoop(registry, max_retries=3)
    result = await loop.repair("shell", {"command": "ls"})
    print(f"  Success:   {result.success}")
    print(f"  Attempts: {result.attempts}")
    print(f"  Errors:    {result.error_history}")
    print()

    # --- Exhausts retries ---

    print("=== Exhausts Retries ===")
    registry = StubRegistry([
        ToolResult(output="fail", error=True),
        ToolResult(output="fail", error=True),
        ToolResult(output="fail", error=True),
    ])
    loop = RepairLoop(registry, max_retries=3)
    result = await loop.repair("tool", {})
    print(f"  Success:   {result.success}")
    print(f"  Attempts: {result.attempts}")
    print(f"  Errors:    {result.error_history}")
    print()

    # --- Diagnose ---

    print("=== Diagnose ===")
    diagnosis = await loop.diagnose("deploy", "HTTP 500 Internal Server Error")
    print(f"  {diagnosis}")


if __name__ == "__main__":
    asyncio.run(main())
