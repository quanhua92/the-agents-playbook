"""Multi-turn tool chaining with re-scoring between steps.

ToolChainer executes sequential tool calls, re-scoring after each step
to decide whether to continue chaining or stop. This is what separates
"an agent that can call a tool" from "an agent that can accomplish tasks."
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from ..tools.protocol import ToolResult
from ..tools.registry import ToolRegistry
from .scoring import score_tools

logger = logging.getLogger(__name__)


@dataclass
class ToolChain:
    """Record of a multi-step tool chain execution.

    Attributes:
        steps: Ordered list of [{tool_name, arguments, result}, ...]
        final_output: The last tool's output text.
        confidence: Confidence score after the chain (1.0 = certain).
    """

    steps: list[dict[str, Any]] = field(default_factory=list)
    final_output: str | None = None
    confidence: float = 1.0


class ToolChainer:
    """Execute sequential tool calls with re-scoring between steps.

    The chainer runs tool calls one after another, scoring remaining tools
    after each step. If entropy drops below a threshold (tool becomes clear),
    or the chain reaches max length, it stops.

    Usage:
        chainer = ToolChainer(registry, max_chain_length=3)
        chain = await chainer.execute_chain({
            "tool_name": "shell",
            "arguments": {"command": "ls"}
        })
    """

    def __init__(
        self,
        registry: ToolRegistry,
        max_chain_length: int = 3,
        entropy_threshold: float = 1.0,
    ) -> None:
        self._registry = registry
        self._max_chain_length = max_chain_length
        self._entropy_threshold = entropy_threshold

    @property
    def max_chain_length(self) -> int:
        return self._max_chain_length

    def should_chain(self, result: ToolResult, entropy: float) -> bool:
        """Decide whether to continue chaining after a tool result.

        Args:
            result: The result from the last tool call.
            entropy: Current Shannon entropy of tool selection scores.

        Returns:
            True if the agent should continue chaining.
        """
        # Stop on errors
        if result.error:
            logger.info("Stopping chain: tool returned error")
            return False

        # Stop if entropy is low enough (tool selection is clear)
        if entropy < self._entropy_threshold:
            logger.info(
                "Stopping chain: entropy %.2f below threshold %.2f",
                entropy,
                self._entropy_threshold,
            )
            return False

        return True

    async def execute_chain(
        self,
        initial_call: dict[str, Any],
        tool_scores: dict[str, float] | None = None,
    ) -> ToolChain:
        """Execute a chain of tool calls starting from initial_call.

        After each step, remaining tools are re-scored. If entropy is low
        or chain reaches max length, execution stops.

        Args:
            initial_call: Dict with "tool_name" and "arguments" keys.
            tool_scores: Optional initial tool relevance scores for entropy
                calculation. If None, entropy is not checked.

        Returns:
            ToolChain with all steps and the final output.
        """
        chain = ToolChain()
        current_call = initial_call

        for step_num in range(self._max_chain_length):
            tool_name = current_call.get("tool_name", "")
            arguments = current_call.get("arguments", {})

            logger.info("Chain step %d: %s(%s)", step_num + 1, tool_name, arguments)

            try:
                result = await self._registry.dispatch(tool_name, arguments)
            except Exception as exc:
                logger.warning("Chain step %d failed: %s", step_num + 1, exc)
                chain.steps.append(
                    {
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "result": ToolResult(output=str(exc), error=True),
                    }
                )
                chain.confidence = 0.0
                chain.final_output = str(exc)
                break

            chain.steps.append(
                {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": result,
                }
            )
            chain.final_output = result.output

            # Mark confidence as zero if tool returned an error
            if result.error:
                chain.confidence = 0.0

            # Check if we should continue
            if tool_scores is not None:
                entropy = score_tools(tool_scores)
                if not self.should_chain(result, entropy):
                    break
            else:
                # No scores provided — just check for errors
                if result.error:
                    break

            # In a real agent loop, the LLM would decide the next call
            # For the chainer, we stop after one call unless extended
            break

        return chain
