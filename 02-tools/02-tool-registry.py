"""02-tool-registry.py — Register tools, get specs, look up by name.

The ToolRegistry is the central place where all tools are registered.
Its key feature: get_specs() returns list[ToolSpec] which feeds directly
into MessageRequest.tools — no adapter layer needed.
"""

import asyncio
from typing import Any

from the_agents_playbook.tools import Tool, ToolResult, ToolRegistry


class EchoTool(Tool):
    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Repeat back the input text."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to echo back"},
            },
            "required": ["text"],
            "additionalProperties": False,
        }

    async def execute(self, text: str, **kwargs: Any) -> ToolResult:
        return ToolResult(output=text)


class AddTool(Tool):
    @property
    def name(self) -> str:
        return "add"

    @property
    def description(self) -> str:
        return "Add two numbers together."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        }

    async def execute(self, a: float, b: float, **kwargs: Any) -> ToolResult:
        return ToolResult(output=str(a + b))


async def main():
    registry = ToolRegistry()

    # Register tools
    registry.register(EchoTool())
    registry.register(AddTool())

    # List all registered tools
    print(f"Registered tools: {registry.list_tools()}")
    print()

    # get_specs() returns list[ToolSpec] — the exact type MessageRequest.tools accepts
    specs = registry.get_specs()
    print("ToolSpecs for the LLM:")
    for spec in specs:
        print(f"  - {spec.name}: {spec.description}")
        print(f"    Parameters: {spec.parameters}")
    print()

    # Look up and dispatch tools by name
    result = await registry.dispatch("echo", {"text": "hello from the registry"})
    print(f"echo: {result.output}")

    result = await registry.dispatch("add", {"a": 42, "b": 58})
    print(f"add:  {result.output}")

    # Looking up an unregistered tool raises ToolNotFoundError
    try:
        registry.get("nonexistent")
    except Exception as e:
        print(f"\nLookup error: {e}")


asyncio.run(main())
