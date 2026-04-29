"""ToolRegistry — register tools, look them up, generate ToolSpecs for the LLM."""

import logging
from typing import Any

from the_agents_playbook.providers.types import ToolSpec
from .protocol import Tool, ToolResult

logger = logging.getLogger(__name__)


class ToolNotFoundError(Exception):
    """Raised when dispatching to an unregistered tool."""

    def __init__(self, name: str):
        super().__init__(f"Tool '{name}' not registered")
        self.name = name


class ToolRegistry:
    """Register tools, look them up by name, and produce ToolSpecs for the LLM.

    Usage:
        registry = ToolRegistry()
        registry.register(MyTool())
        specs = registry.get_specs()  # list[ToolSpec] for MessageRequest.tools
        result = await registry.dispatch("my_tool", {"arg": "value"})
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool instance by its name. Overwrites if already exists."""
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def get(self, name: str) -> Tool:
        """Look up a registered tool by name. Raises ToolNotFoundError."""
        if name not in self._tools:
            raise ToolNotFoundError(name)
        return self._tools[name]

    def list_tools(self) -> list[str]:
        """Return names of all registered tools."""
        return list(self._tools.keys())

    def get_specs(self) -> list[ToolSpec]:
        """Return ToolSpecs for all registered tools, ready for MessageRequest.tools."""
        return [
            ToolSpec(name=t.name, description=t.description, parameters=t.parameters)
            for t in self._tools.values()
        ]

    async def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Look up a tool by name and execute it with the given arguments."""
        tool = self.get(tool_name)
        logger.info("Dispatching tool %s with args: %s", tool_name, arguments)
        return await tool.execute(**arguments)
