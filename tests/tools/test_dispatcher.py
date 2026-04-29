"""Tests for tools/dispatcher.py — ToolDispatcher."""

from typing import Any

import pytest

from the_agents_playbook.tools import Tool, ToolResult, ToolRegistry
from the_agents_playbook.tools.dispatcher import (
    ToolArgumentError,
    ToolDispatcher,
)


class StrictTool(Tool):
    """Tool with required parameters and additionalProperties=False."""

    @property
    def name(self) -> str:
        return "strict"

    @property
    def description(self) -> str:
        return "Requires 'name' (str) and 'count' (int)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["name", "count"],
            "additionalProperties": False,
        }

    async def execute(self, name: str = "", count: int = 0, **kwargs: Any) -> ToolResult:
        return ToolResult(output=f"{name} x{count}")


class FailTool(Tool):
    """Tool that raises an exception to test error capture."""

    @property
    def name(self) -> str:
        return "fail"

    @property
    def description(self) -> str:
        return "Always fails"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        raise RuntimeError("intentional failure")


@pytest.fixture
def dispatcher() -> ToolDispatcher:
    registry = ToolRegistry()
    registry.register(StrictTool())
    registry.register(FailTool())
    return ToolDispatcher(registry)


# --- parse_arguments ---


def test_parse_valid_json(dispatcher: ToolDispatcher):
    args = dispatcher.parse_arguments('{"name": "test", "count": 3}')
    assert args == {"name": "test", "count": 3}


def test_parse_invalid_json(dispatcher: ToolDispatcher):
    with pytest.raises(ToolArgumentError):
        dispatcher.parse_arguments("not json{{{")


def test_parse_non_object(dispatcher: ToolDispatcher):
    with pytest.raises(ToolArgumentError) as exc_info:
        dispatcher.parse_arguments('"just a string"')
    assert "Expected JSON object" in str(exc_info.value)


# --- validate_arguments ---


def test_validate_passes(dispatcher: ToolDispatcher):
    dispatcher.validate_arguments("strict", {"name": "a", "count": 1})


def test_validate_missing_required(dispatcher: ToolDispatcher):
    with pytest.raises(ToolArgumentError) as exc_info:
        dispatcher.validate_arguments("strict", {"name": "a"})
    assert "Missing required" in str(exc_info.value)
    assert "count" in str(exc_info.value)


def test_validate_extra_properties(dispatcher: ToolDispatcher):
    with pytest.raises(ToolArgumentError) as exc_info:
        dispatcher.validate_arguments("strict", {"name": "a", "count": 1, "extra": True})
    assert "Unknown arguments" in str(exc_info.value)


# --- dispatch_one ---


async def test_dispatch_one_success(dispatcher: ToolDispatcher):
    call_id, result = await dispatcher.dispatch_one(
        "strict", '{"name": "test", "count": 5}', "call-1"
    )
    assert call_id == "call-1"
    assert result.output == "test x5"
    assert result.error is False


async def test_dispatch_one_bad_json(dispatcher: ToolDispatcher):
    call_id, result = await dispatcher.dispatch_one(
        "strict", "bad json", "call-2"
    )
    assert result.error is True
    assert "Invalid JSON" in result.output


async def test_dispatch_one_missing_arg(dispatcher: ToolDispatcher):
    call_id, result = await dispatcher.dispatch_one(
        "strict", '{"name": "a"}', "call-3"
    )
    assert result.error is True
    assert "Missing required" in result.output


async def test_dispatch_one_unknown_tool(dispatcher: ToolDispatcher):
    call_id, result = await dispatcher.dispatch_one(
        "nonexistent", '{}', "call-4"
    )
    assert result.error is True
    assert "not found" in result.output


async def test_dispatch_one_execution_error(dispatcher: ToolDispatcher):
    call_id, result = await dispatcher.dispatch_one(
        "fail", '{}', "call-5"
    )
    assert result.error is True
    assert "failed" in result.output


# --- dispatch_all ---


async def test_dispatch_all_multiple(dispatcher: ToolDispatcher):
    tool_calls = [
        {"id": "c1", "function": {"name": "strict", "arguments": '{"name": "a", "count": 1}'}},
        {"id": "c2", "function": {"name": "fail", "arguments": "{}"}},
    ]
    results = await dispatcher.dispatch_all(tool_calls)
    assert len(results) == 2
    assert results[0][0] == "c1"
    assert results[0][1].output == "a x1"
    assert results[0][1].error is False
    assert results[1][0] == "c2"
    assert results[1][1].error is True
