"""Tests for loop.agent — Agent ReAct loop."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from the_agents_playbook.loop.agent import Agent
from the_agents_playbook.loop.config import AgentConfig
from the_agents_playbook.loop.protocol import AgentEvent, TurnResult
from the_agents_playbook.providers.types import (
    InputMessage,
    MessageRequest,
    MessageResponse,
    OutputMessage,
)


def _text_response(content: str = "Done!") -> MessageResponse:
    """Create a mock text-only response."""
    return MessageResponse(
        message=OutputMessage(role="assistant", content=content),
        stop_reason="stop",
    )


def _tool_call_response(tool_name: str, arguments: dict) -> MessageResponse:
    """Create a mock response with a single tool call."""
    return MessageResponse(
        message=OutputMessage(
            role="assistant",
            content=None,
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": __import__("json").dumps(arguments),
                },
            }],
        ),
        stop_reason="tool_calls",
    )


@pytest.fixture
def mock_provider():
    provider = AsyncMock()
    provider.send_message = AsyncMock(return_value=_text_response("Hello!"))
    provider.close = AsyncMock()
    return provider


@pytest.fixture
def mock_registry():
    registry = MagicMock()
    registry.get_specs = MagicMock(return_value=[])
    registry.list_tools = MagicMock(return_value=[])
    return registry


@pytest.fixture
def agent(mock_provider, mock_registry):
    return Agent(
        provider=mock_provider,
        registry=mock_registry,
        config=AgentConfig(max_tool_iterations=5),
    )


class TestAgentInit:
    def test_basic_init(self, mock_provider, mock_registry):
        agent = Agent(provider=mock_provider, registry=mock_registry)
        assert agent._provider is mock_provider
        assert agent._registry is mock_registry
        assert agent._memory is None
        assert agent._context_builder is None

    def test_init_with_memory(self, mock_provider, mock_registry):
        memory = MagicMock()
        agent = Agent(provider=mock_provider, registry=mock_registry, memory=memory)
        assert agent._memory is memory

    def test_init_with_config(self, mock_provider, mock_registry):
        config = AgentConfig(max_tool_iterations=10, on_error="raise")
        agent = Agent(provider=mock_provider, registry=mock_registry, config=config)
        assert agent._config.max_tool_iterations == 10
        assert agent._config.on_error == "raise"


class TestAgentRun:
    async def test_single_text_response(self, agent, mock_provider):
        """Agent gets a text response immediately — no tool calls."""
        mock_provider.send_message.return_value = _text_response("Final answer")

        events = []
        async for event in agent.run("Hello"):
            events.append(event)

        types = [e.type for e in events]
        assert "status" in types
        assert "text" in types
        text_event = next(e for e in events if e.type == "text")
        assert text_event.data["text"] == "Final answer"

    async def test_tool_call_then_text(self, agent, mock_provider, mock_registry):
        """Agent calls a tool, gets result, then responds with text."""
        tool_results = [
            _tool_call_response("echo", {"message": "hi"}),
            _text_response("Got: hi"),
        ]
        mock_provider.send_message = AsyncMock(side_effect=tool_results)
        mock_registry.dispatch = AsyncMock(
            return_value=MagicMock(output="hi", error=False)
        )

        events = []
        async for event in agent.run("Say hi"):
            events.append(event)

        types = [e.type for e in events]
        assert "tool_call" in types
        assert "tool_result" in types
        assert "text" in types

    async def test_max_iterations(self, agent, mock_provider):
        """Agent stops after max_tool_iterations."""
        mock_provider.send_message.return_value = _tool_call_response(
            "echo", {"message": "loop"}
        )

        events = []
        async for event in agent.run("Keep going"):
            events.append(event)

        status_events = [e for e in events if e.type == "status"]
        # Should have max_iterations status events + final max-iter notice
        assert len([e for e in status_events if "max" in e.data.get("message", "")]) == 1

    async def test_on_error_raise(self, mock_provider, mock_registry):
        """Agent raises on provider error when on_error='raise'."""
        mock_provider.send_message = AsyncMock(side_effect=RuntimeError("API down"))

        agent = Agent(
            provider=mock_provider,
            registry=mock_registry,
            config=AgentConfig(on_error="raise"),
        )

        with pytest.raises(RuntimeError, match="API down"):
            events = []
            async for event in agent.run("test"):
                events.append(event)

    async def test_on_error_abort(self, mock_provider, mock_registry):
        """Agent yields error event and stops when on_error='abort'."""
        mock_provider.send_message = AsyncMock(side_effect=RuntimeError("API down"))

        agent = Agent(
            provider=mock_provider,
            registry=mock_registry,
            config=AgentConfig(on_error="abort"),
        )

        events = []
        async for event in agent.run("test"):
            events.append(event)

        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) == 1
        assert "API down" in error_events[0].data["message"]

    async def test_on_error_yield_and_continue(self, mock_provider, mock_registry):
        """Agent yields error event and returns when on_error='yield_and_continue'."""
        mock_provider.send_message = AsyncMock(side_effect=RuntimeError("timeout"))

        agent = Agent(
            provider=mock_provider,
            registry=mock_registry,
            config=AgentConfig(on_error="yield_and_continue"),
        )

        events = []
        async for event in agent.run("test"):
            events.append(event)

        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) == 1


class TestAgentRunTurn:
    async def test_turn_returns_result(self, agent, mock_provider):
        mock_provider.send_message.return_value = _text_response("Done")

        result = await agent.run_turn("Hello")
        assert isinstance(result, TurnResult)
        assert result.final_response == "Done"
        assert result.error is None

    async def test_turn_with_error(self, mock_provider, mock_registry):
        mock_provider.send_message = AsyncMock(side_effect=RuntimeError("fail"))

        agent = Agent(
            provider=mock_provider,
            registry=mock_registry,
            config=AgentConfig(on_error="abort"),
        )

        result = await agent.run_turn("test")
        assert result.error == "fail"


class TestAgentWithMemory:
    async def test_stores_user_message(self, mock_provider, mock_registry):
        memory = AsyncMock()
        mock_provider.send_message.return_value = _text_response("ok")

        agent = Agent(provider=mock_provider, registry=mock_registry, memory=memory)
        async for _ in agent.run("remember this"):
            pass

        # Memory should have been called to store
        assert memory.store.called

    async def test_recalls_before_llm_call(self, mock_provider, mock_registry):
        memory = AsyncMock()
        memory.recall = AsyncMock(return_value=[])
        mock_provider.send_message.return_value = _text_response("ok")

        agent = Agent(provider=mock_provider, registry=mock_registry, memory=memory)
        async for _ in agent.run("test"):
            pass

        memory.recall.assert_called_once()

    async def test_stores_assistant_response(self, mock_provider, mock_registry):
        memory = AsyncMock()
        mock_provider.send_message.return_value = _text_response("stored!")

        agent = Agent(provider=mock_provider, registry=mock_registry, memory=memory)
        async for _ in agent.run("test"):
            pass

        # Should store both user message and assistant response
        assert memory.store.call_count >= 2


class TestAgentClose:
    async def test_close_calls_provider_close(self, mock_provider, mock_registry):
        agent = Agent(provider=mock_provider, registry=mock_registry)
        await agent.close()
        mock_provider.close.assert_called_once()
