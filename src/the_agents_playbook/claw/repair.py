"""Self-repair loop — when a tool fails, the agent reads the error, diagnoses, and retries.

The repair loop wraps tool execution with a retry strategy: catch errors,
attempt diagnosis, and retry with a fix. Max retries prevent infinite loops.
Error history is preserved for post-mortem analysis.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from ..tools.protocol import ToolResult

logger = logging.getLogger(__name__)


@dataclass
class RepairResult:
    """Outcome of a self-repair attempt.

    Attributes:
        success: Whether the repair succeeded.
        attempts: Number of retry attempts made.
        final_output: The successful output (if repaired).
        error_history: List of error messages from each attempt.
    """

    success: bool
    attempts: int = 1
    final_output: str | None = None
    error_history: list[str] = field(default_factory=list)


class RepairLoop:
    """Wraps tool dispatch with retry-on-failure logic.

    Usage:
        loop = RepairLoop(registry, max_retries=3)
        result = await loop.repair("shell", {"command": "ls /tmp"})
    """

    def __init__(self, registry: Any, max_retries: int = 3) -> None:
        self._registry = registry
        self._max_retries = max_retries

    @property
    def max_retries(self) -> int:
        return self._max_retries

    async def repair(self, tool_name: str, arguments: dict[str, Any]) -> RepairResult:
        """Execute a tool with automatic retry on failure.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Arguments to pass to the tool.

        Returns:
            RepairResult with outcome and error history.
        """
        error_history: list[str] = []

        for attempt in range(1, self._max_retries + 1):
            try:
                result = await self._registry.dispatch(tool_name, arguments)
                if result.error:
                    error_history.append(result.output)
                    logger.warning(
                        "Repair attempt %d/%d: tool %s returned error: %s",
                        attempt, self._max_retries, tool_name, result.output,
                    )
                    continue

                return RepairResult(
                    success=True,
                    attempts=attempt,
                    final_output=result.output,
                    error_history=error_history,
                )
            except Exception as exc:
                error_msg = str(exc)
                error_history.append(error_msg)
                logger.warning(
                    "Repair attempt %d/%d: tool %s raised: %s",
                    attempt, self._max_retries, tool_name, error_msg,
                )

        return RepairResult(
            success=False,
            attempts=self._max_retries,
            error_history=error_history,
        )

    async def diagnose(self, tool_name: str, error: str) -> str:
        """Produce a diagnosis string for a tool error.

        In a real implementation, this would invoke the LLM to analyze
        the error. Here it returns a structured description.

        Args:
            tool_name: Name of the tool that failed.
            error: The error message.

        Returns:
            A diagnosis string.
        """
        return f"Tool '{tool_name}' failed: {error}. Possible fix: check arguments and retry."
