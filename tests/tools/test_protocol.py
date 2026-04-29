"""Tests for tools/protocol.py — Tool ABC and ToolResult."""

from typing import Any

from the_agents_playbook.tools.protocol import Tool, ToolResult


class FakeTool(Tool):
    @property
    def name(self) -> str:
        return "fake"

    @property
    def description(self) -> str:
        return "A fake tool for testing"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
            "additionalProperties": False,
        }

    async def execute(self, x: str = "", **kwargs: Any) -> ToolResult:
        return ToolResult(output=f"got {x}")


def test_tool_result_default_fields():
    result = ToolResult(output="hello")
    assert result.output == "hello"
    assert result.error is False
    assert result.metadata == {}


def test_tool_result_error():
    result = ToolResult(output="failed", error=True, metadata={"code": 42})
    assert result.error is True
    assert result.metadata["code"] == 42


def test_tool_result_is_dataclass():
    """ToolResult should be a dataclass for easy serialization."""
    import dataclasses
    assert dataclasses.is_dataclass(ToolResult)


def test_tool_protocol_requires_subclass():
    """Cannot instantiate Tool directly — must subclass."""
    import pytest
    with pytest.raises(TypeError):
        Tool()


def test_tool_subclass_has_contract():
    """A valid subclass must implement all abstract properties and methods."""
    tool = FakeTool()
    assert tool.name == "fake"
    assert tool.description == "A fake tool for testing"
    assert tool.parameters["required"] == ["x"]


async def test_tool_execute_returns_tool_result():
    tool = FakeTool()
    result = await tool.execute(x="test")
    assert isinstance(result, ToolResult)
    assert result.output == "got test"
    assert result.error is False
