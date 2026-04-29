"""Graceful degradation — fallback behavior when tools fail, LLM is unreachable, or context overflows.

DegradationManager provides handlers for common failure modes so the agent
can continue operating (in a limited capacity) rather than crashing.
"""

import logging
from dataclasses import dataclass
from typing import Any

from ..tools.protocol import ToolResult

logger = logging.getLogger(__name__)


@dataclass
class FallbackResult:
    """Result of a degradation fallback.

    Attributes:
        handled: Whether the fallback was applied successfully.
        output: The fallback output text.
        strategy: Which fallback strategy was used.
    """

    handled: bool
    output: str
    strategy: str


class DegradationManager:
    """Manages graceful fallback when parts of the system fail.

    Usage:
        mgr = DegradationManager()
        result = await mgr.handle_tool_failure("shell", RuntimeError("timeout"))
    """

    def __init__(self) -> None:
        self._tool_fallbacks: dict[str, str] = {}
        self._default_tool_fallback = "Return partial result and explain limitation."

    def register_tool_fallback(self, tool_name: str, fallback_message: str) -> None:
        """Register a specific fallback message for a tool."""
        self._tool_fallbacks[tool_name] = fallback_message

    async def handle_tool_failure(self, tool_name: str, error: Exception) -> FallbackResult:
        """Handle a tool failure with a graceful fallback.

        Args:
            tool_name: Name of the tool that failed.
            error: The exception that was raised.

        Returns:
            FallbackResult with a user-facing message.
        """
        fallback_msg = self._tool_fallbacks.get(
            tool_name, self._default_tool_fallback
        )
        output = f"Tool '{tool_name}' is unavailable: {error}. {fallback_msg}"

        logger.warning("Tool degradation: %s → %s", tool_name, error)
        return FallbackResult(
            handled=True,
            output=output,
            strategy="tool_fallback",
        )

    async def handle_llm_failure(self, error: Exception) -> FallbackResult:
        """Handle an LLM provider failure.

        Args:
            error: The exception from the provider.

        Returns:
            FallbackResult explaining the situation.
        """
        output = f"LLM is temporarily unavailable ({error}). Please retry."
        logger.error("LLM degradation: %s", error)
        return FallbackResult(
            handled=True,
            output=output,
            strategy="llm_fallback",
        )

    async def handle_context_overflow(self, max_tokens: int = 8192) -> FallbackResult:
        """Handle context window overflow.

        Suggests compacting context or reducing input.

        Args:
            max_tokens: The context window limit.

        Returns:
            FallbackResult with remediation advice.
        """
        output = f"Context exceeds {max_tokens} tokens. Try compacting history or reducing input."
        logger.warning("Context overflow: limit=%d tokens", max_tokens)
        return FallbackResult(
            handled=True,
            output=output,
            strategy="context_compaction",
        )
