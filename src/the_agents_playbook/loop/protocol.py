"""Agent event types — the streaming contract for the agent loop."""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class AgentEvent:
    """A single event yielded by the agent loop during execution.

    The agent yields events as it works through the ReAct loop, allowing
    callers to stream progress, tool calls, and results in real time.
    """

    type: Literal["text", "tool_call", "tool_result", "status", "error"]
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type == "text":
            self.data.setdefault("text", "")
        elif self.type == "tool_call":
            self.data.setdefault("tool_name", "")
            self.data.setdefault("arguments", {})
        elif self.type == "tool_result":
            self.data.setdefault("output", "")
            self.data.setdefault("error", False)
        elif self.type == "status":
            self.data.setdefault("message", "")
        elif self.type == "error":
            self.data.setdefault("message", "")


@dataclass
class TurnResult:
    """Summary of a single ReAct turn (one LLM call + tool executions).

    A turn may include zero or more tool calls. The loop collects these
    into a TurnResult for diagnostics and history tracking.
    """

    events: list[AgentEvent] = field(default_factory=list)
    tool_calls_made: int = 0
    final_response: str | None = None
    error: str | None = None
