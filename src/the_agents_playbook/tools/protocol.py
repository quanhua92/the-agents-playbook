"""Tool protocol — the contract every tool must fulfill."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """The outcome of a tool execution."""

    output: str
    error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    """Base class for all tools.

    Every tool must define:
    - name: a unique string identifier
    - description: what the tool does (sent to the LLM)
    - parameters: a JSON Schema dict describing accepted arguments
    - execute(): the actual implementation
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """Return a JSON Schema dict for this tool's arguments."""
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool with validated arguments."""
        ...
