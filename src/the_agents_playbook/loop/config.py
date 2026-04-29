"""Agent configuration — controls the ReAct loop behavior."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class AgentConfig:
    """Configuration for the agent loop.

    Attributes:
        max_tool_iterations: Safety limit on ReAct loop cycles.
        on_error: Controls error propagation in the loop.
        entropy_threshold: Shannon entropy threshold — ask user when
            tool selection uncertainty exceeds this (in bits).
        max_chain_length: Maximum sequential tool calls before re-scoring.
    """

    max_tool_iterations: int = 25
    on_error: Literal["raise", "yield_and_continue", "abort"] = "abort"
    entropy_threshold: float = 1.5
    max_chain_length: int = 3
