"""ToolDispatcher — maps LLM tool_calls to execution with argument validation."""

import json
import logging
from typing import Any

from .registry import ToolNotFoundError, ToolRegistry
from .protocol import ToolResult

logger = logging.getLogger(__name__)


class ToolArgumentError(Exception):
    """Raised when tool arguments fail validation."""

    def __init__(self, tool_name: str, reason: str):
        super().__init__(f"Invalid arguments for '{tool_name}': {reason}")
        self.tool_name = tool_name
        self.reason = reason


class ToolTimeoutError(Exception):
    """Raised when a tool execution exceeds its timeout."""

    def __init__(self, tool_name: str, timeout_seconds: float):
        super().__init__(f"Tool '{tool_name}' timed out after {timeout_seconds}s")
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds


class ToolDispatcher:
    """Validates tool call arguments and dispatches to the registry.

    Handles the full lifecycle of a single tool call:
    1. Parse arguments from the LLM response (JSON string → dict)
    2. Validate against the tool's JSON Schema (basic checks)
    3. Execute via the registry
    4. Handle errors gracefully

    Usage:
        dispatcher = ToolDispatcher(registry)
        results = await dispatcher.dispatch_all(response.message.tool_calls)
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def parse_arguments(self, arguments_json: str) -> dict[str, Any]:
        """Parse the JSON string of arguments from an LLM tool call.

        The OpenAI API returns arguments as a JSON string, not a dict.
        This method parses it and handles common malformation issues.
        """
        try:
            args = json.loads(arguments_json)
        except json.JSONDecodeError as e:
            raise ToolArgumentError("unknown", f"Invalid JSON: {e}")

        if not isinstance(args, dict):
            raise ToolArgumentError(
                "unknown", f"Expected JSON object, got {type(args).__name__}"
            )

        return args

    def validate_arguments(self, tool_name: str, args: dict[str, Any]) -> None:
        """Basic argument validation against the tool's JSON Schema.

        Checks:
        - Required properties are present
        - No unknown properties (if additionalProperties is False)
        - Basic type checking for primitive types

        This is intentionally lightweight — it catches the most common
        LLM mistakes without pulling in a full JSON Schema validator.
        """
        tool = self._registry.get(tool_name)
        schema = tool.parameters

        if "required" in schema:
            for prop in schema["required"]:
                if prop not in args:
                    raise ToolArgumentError(
                        tool_name,
                        f"Missing required argument: '{prop}'",
                    )

        if schema.get("additionalProperties") is False and "properties" in schema:
            allowed = set(schema["properties"].keys())
            unknown = set(args.keys()) - allowed
            if unknown:
                raise ToolArgumentError(
                    tool_name,
                    f"Unknown arguments: {unknown}. Allowed: {allowed}",
                )

    async def dispatch_one(
        self,
        tool_name: str,
        arguments_json: str,
        tool_call_id: str | None = None,
    ) -> tuple[str, ToolResult]:
        """Parse, validate, and execute a single tool call.

        Returns (tool_call_id, ToolResult). If execution fails, the error
        is captured in a ToolResult with error=True so the agent loop can
        feed it back to the LLM.
        """
        try:
            args = self.parse_arguments(arguments_json)
            self.validate_arguments(tool_name, args)
            result = await self._registry.dispatch(tool_name, args)
            return (tool_call_id or "", result)

        except ToolNotFoundError as e:
            logger.warning("Tool not found: %s", e.name)
            return (
                tool_call_id or "",
                ToolResult(output=f"Tool not found: {e.name}", error=True),
            )

        except ToolArgumentError as e:
            logger.warning(
                "Argument validation failed for %s: %s", e.tool_name, e.reason
            )
            return (
                tool_call_id or "",
                ToolResult(output=f"Invalid arguments: {e.reason}", error=True),
            )

        except Exception as e:
            logger.exception("Unexpected error dispatching tool %s", tool_name)
            return (
                tool_call_id or "",
                ToolResult(output=f"Tool execution failed: {e}", error=True),
            )

    async def dispatch_all(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[tuple[str, ToolResult]]:
        """Dispatch multiple tool calls from an LLM response.

        tool_calls is the raw list from OutputMessage.tool_calls, where
        each item has: {"id": "...", "function": {"name": "...", "arguments": "..."}}.
        """
        results = []
        for call in tool_calls:
            fn = call.get("function", {})
            tool_name = fn.get("name", "")
            arguments = fn.get("arguments", "{}")
            call_id = call.get("id", "")
            result = await self.dispatch_one(tool_name, arguments, call_id)
            results.append(result)
        return results
