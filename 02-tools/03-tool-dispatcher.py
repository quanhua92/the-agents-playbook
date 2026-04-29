"""03-tool-dispatcher.py — Full loop: LLM requests tool call → dispatcher validates → executes.

The ToolDispatcher bridges the gap between the LLM response and tool execution.
It parses the JSON arguments from the LLM, validates them against the tool's
JSON Schema, and dispatches to the registry. Errors are captured in ToolResult
so the agent loop can feed them back to the LLM.
"""

import asyncio
import json
from typing import Any

from the_agents_playbook import settings
from the_agents_playbook.providers import OpenAIProvider, MessageRequest, InputMessage, ToolChoice
from the_agents_playbook.tools import Tool, ToolResult, ToolRegistry
from the_agents_playbook.tools.dispatcher import ToolDispatcher


class ReverseTool(Tool):
    """A toy tool that reverses a string — simple enough to verify the full loop."""

    @property
    def name(self) -> str:
        return "reverse_string"

    @property
    def description(self) -> str:
        return "Reverse the given string."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The string to reverse",
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        }

    async def execute(self, text: str, **kwargs: Any) -> ToolResult:
        return ToolResult(output=text[::-1])


async def main():
    # Set up registry and dispatcher
    registry = ToolRegistry()
    registry.register(ReverseTool())
    dispatcher = ToolDispatcher(registry)

    provider = OpenAIProvider()

    # Step 1: Send a message with tools attached
    print("=== Sending request with tool ===")
    response = await provider.send_message(
        MessageRequest(
            model=settings.openai_model,
            system="You have access to a reverse_string tool. Use it when the user asks you to reverse text.",
            messages=[
                InputMessage(role="user", content="Please reverse the string 'hello world'"),
            ],
            tools=registry.get_specs(),  # list[ToolSpec] feeds directly into provider
            tool_choice=ToolChoice(type="auto"),
        )
    )

    print(f"Stop reason: {response.stop_reason}")
    print(f"Content:     {response.message.content}")
    print(f"Tool calls:  {response.message.tool_calls}")
    print()

    # Step 2: Dispatch tool calls from the response
    if response.message.tool_calls:
        print("=== Dispatching tool calls ===")
        results = await dispatcher.dispatch_all(response.message.tool_calls)

        for call_id, result in results:
            print(f"  [{call_id}] output={result.output!r} error={result.error}")
    else:
        print("No tool calls in response — model answered directly.")

    # Step 3: Show dispatcher error handling
    print("\n=== Error handling demos ===")

    # Bad JSON arguments
    result = await dispatcher.dispatch_one("reverse_string", "not valid json{{{", "test-id")
    print(f"Bad JSON:    {result.output} (error={result.error})")

    # Missing required argument
    result = await dispatcher.dispatch_one("reverse_string", "{}", "test-id")
    print(f"Missing arg: {result.output} (error={result.error})")

    # Unknown tool
    result = await dispatcher.dispatch_one("nonexistent", '{"text":"x"}', "test-id")
    print(f"Unknown tool: {result.output} (error={result.error})")

    await provider.close()


asyncio.run(main())
