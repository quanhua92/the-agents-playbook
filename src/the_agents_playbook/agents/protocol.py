"""Agent protocol — the contract for any agent in a multi-agent system.

In a multi-agent architecture, agents are not monolithic. Each agent has
a specific role (researcher, writer, coder, reviewer) with a scoped set
of tools. A dispatcher routes tasks to the appropriate worker.

Key principle: the dispatcher decides, the worker does. Separation of
concerns prevents runaway agent behavior. Scoped tool sets prevent a
research agent from accidentally sending emails.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any


class AgentEvent:
    """Base event yielded by agents during execution."""

    def __init__(self, type: str, data: dict[str, Any] | None = None, source: str = ""):
        self.type = type
        self.data = data or {}
        self.source = source

    def __repr__(self) -> str:
        return f"AgentEvent(type={self.type!r}, source={self.source!r})"


class BaseAgent(ABC):
    """Abstract base class for all agents in a multi-agent system.

    Every agent must define:
    - name: unique identifier used for routing
    - description: what this agent does (used by dispatcher for routing)
    - tools: the scoped set of tools this agent can use
    - run(): execute a prompt and yield events

    Usage:
        class MyAgent(BaseAgent):
            @property
            def name(self) -> str: return "my-agent"
            @property
            def description(self) -> str: return "Does X"
            @property
            def tools(self) -> list[Tool]: return [tool_a, tool_b]
            async def run(self, prompt: str) -> AsyncGenerator[AgentEvent, None]:
                ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this agent."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this agent does."""
        ...

    @property
    @abstractmethod
    def tools(self) -> list[Any]:
        """Scoped tool set for this agent."""
        ...

    @abstractmethod
    async def run(self, prompt: str) -> AsyncGenerator[AgentEvent, None]:
        """Execute the agent on a prompt and yield events.

        Args:
            prompt: The task or instruction.

        Yields:
            AgentEvent objects as the agent works.
        """
        ...
