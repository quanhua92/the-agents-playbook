"""04-react-agent.py — Full ReAct loop: user message → tools → response.

This example shows the Agent in action with a real LLM provider.
The agent receives a prompt, optionally calls tools, and produces
a final response. Events are streamed in real time.

Set MOCK_ONLY=true in .env to run without an API key (uses mock responses).
"""

import asyncio

from the_agents_playbook import settings
from the_agents_playbook.loop import Agent, AgentConfig
from the_agents_playbook.tools import Tool, ToolResult, ToolRegistry


class EchoTool(Tool):
    """A simple tool that echoes its input."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echoes back the input message."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        }

    async def execute(self, message: str, **kwargs) -> ToolResult:
        return ToolResult(output=f"Echo: {message}")


def make_mock_provider():
    """Create a mock provider that simulates a tool call then text response."""
    from unittest.mock import AsyncMock

    from the_agents_playbook.providers.types import (
        MessageResponse,
        OutputMessage,
    )

    provider = AsyncMock()
    provider.close = AsyncMock()

    import json

    # First call: tool call, second call: text response
    provider.send_message = AsyncMock(
        side_effect=[
            MessageResponse(
                message=OutputMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": json.dumps(
                                    {"message": "hello from agent"}
                                ),
                            },
                        }
                    ],
                ),
                stop_reason="tool_calls",
            ),
            MessageResponse(
                message=OutputMessage(
                    role="assistant",
                    content="I echoed your message and got back the result!",
                ),
                stop_reason="stop",
            ),
        ]
    )

    return provider


def make_provider():
    if settings.mock_only:
        print("NOTE: Running in MOCK_ONLY mode. Set MOCK_ONLY=false to use real LLM.")
        return make_mock_provider()
    else:
        from the_agents_playbook.providers import OpenAIProvider

        return OpenAIProvider()


async def main():
    # --- Set up registry with tools ---

    registry = ToolRegistry()
    registry.register(EchoTool())
    print(f"Registered tools: {registry.list_tools()}")
    print()

    # --- Create agent ---

    provider = make_provider()
    agent = Agent(
        provider=provider,
        registry=registry,
        config=AgentConfig(max_tool_iterations=5),
    )

    # --- Run the agent ---

    print("=== Agent Events ===")
    async for event in agent.run("Say hello"):
        if event.type == "status":
            print(f"  [STATUS] {event.data['message']}")
        elif event.type == "tool_call":
            print(f"  [TOOL]   {event.data['tool_name']}({event.data['arguments']})")
        elif event.type == "tool_result":
            status = "ERROR" if event.data["error"] else "OK"
            print(f"  [RESULT] [{status}] {event.data['output']}")
        elif event.type == "text":
            print(f"  [TEXT]   {event.data['text']}")
        elif event.type == "error":
            print(f"  [ERROR]  {event.data['message']}")

    print()

    # --- Collect into TurnResult ---

    provider2 = make_provider()
    agent2 = Agent(provider=provider2, registry=registry)
    turn = await agent2.run_turn("Say hello")

    print("=== TurnResult ===")
    print(f"  Events:        {len(turn.events)}")
    print(f"  Tool calls:    {turn.tool_calls_made}")
    print(f"  Final response: {turn.final_response}")
    print(f"  Error:         {turn.error}")

    await agent.close()
    await provider2.close()


if __name__ == "__main__":
    asyncio.run(main())
