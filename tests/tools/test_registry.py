"""Tests for tools/registry.py — ToolRegistry."""

from typing import Any

import pytest

from the_agents_playbook.tools import Tool, ToolResult, ToolRegistry
from the_agents_playbook.tools.registry import ToolNotFoundError


class AlphaTool(Tool):
    @property
    def name(self) -> str:
        return "alpha"

    @property
    def description(self) -> str:
        return "Alpha tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"val": {"type": "string"}},
            "required": ["val"],
            "additionalProperties": False,
        }

    async def execute(self, val: str = "", **kwargs: Any) -> ToolResult:
        return ToolResult(output=f"alpha:{val}")


class BetaTool(Tool):
    @property
    def name(self) -> str:
        return "beta"

    @property
    def description(self) -> str:
        return "Beta tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(output="beta:ok")


def test_register_and_list():
    registry = ToolRegistry()
    registry.register(AlphaTool())
    assert registry.list_tools() == ["alpha"]


def test_register_multiple():
    registry = ToolRegistry()
    registry.register(AlphaTool())
    registry.register(BetaTool())
    assert set(registry.list_tools()) == {"alpha", "beta"}


def test_register_overwrites():
    registry = ToolRegistry()
    registry.register(AlphaTool())
    registry.register(AlphaTool())
    assert registry.list_tools() == ["alpha"]


def test_get_existing():
    registry = ToolRegistry()
    tool = AlphaTool()
    registry.register(tool)
    assert registry.get("alpha") is tool


def test_get_missing_raises():
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError) as exc_info:
        registry.get("nonexistent")
    assert "nonexistent" in str(exc_info.value)


def test_get_specs_returns_tool_specs():
    from the_agents_playbook.providers.types import ToolSpec

    registry = ToolRegistry()
    registry.register(AlphaTool())
    registry.register(BetaTool())

    specs = registry.get_specs()
    assert len(specs) == 2
    assert all(isinstance(s, ToolSpec) for s in specs)
    assert specs[0].name == "alpha"
    assert specs[1].name == "beta"


async def test_dispatch_calls_tool():
    registry = ToolRegistry()
    registry.register(AlphaTool())
    result = await registry.dispatch("alpha", {"val": "hello"})
    assert result.output == "alpha:hello"
    assert result.error is False


async def test_dispatch_missing_tool_raises():
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        await registry.dispatch("nonexistent", {})
