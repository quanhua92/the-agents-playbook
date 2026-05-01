"""Tests for multi-agent dispatcher/worker pattern."""

import pytest

from the_agents_playbook.agents.protocol import AgentEvent, BaseAgent
from the_agents_playbook.agents.registry import AgentNotFoundError, AgentRegistry
from the_agents_playbook.agents.dispatcher import AgentDispatcher


class DummyAgent(BaseAgent):
    """Minimal agent for testing."""

    def __init__(self, name: str, description: str, tools=None):
        self._name = name
        self._description = description
        self._tools = tools or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def tools(self) -> list:
        return self._tools

    async def run(self, prompt: str):
        yield AgentEvent(type="status", data={"message": "ok"}, source=self._name)
        yield AgentEvent(
            type="text", data={"text": f"Response to: {prompt}"}, source=self._name
        )


class TestAgentEvent:
    def test_basic(self):
        event = AgentEvent(type="text", data={"text": "hello"})
        assert event.type == "text"
        assert event.data["text"] == "hello"
        assert event.source == ""

    def test_with_source(self):
        event = AgentEvent(
            type="tool_call", data={"tool_name": "x"}, source="researcher"
        )
        assert event.source == "researcher"

    def test_default_data(self):
        event = AgentEvent(type="status")
        assert event.data == {}


class TestAgentRegistry:
    def test_register_and_get(self):
        registry = AgentRegistry()
        agent = DummyAgent("test", "A test agent")
        registry.register(agent)
        assert registry.get("test") is agent

    def test_get_missing_raises(self):
        registry = AgentRegistry()
        with pytest.raises(AgentNotFoundError):
            registry.get("nonexistent")

    def test_list_agents(self):
        registry = AgentRegistry()
        a1 = DummyAgent("a", "first")
        a2 = DummyAgent("b", "second")
        registry.register(a1)
        registry.register(a2)
        assert len(registry.list_agents()) == 2

    def test_list_names(self):
        registry = AgentRegistry()
        registry.register(DummyAgent("alpha", "first"))
        registry.register(DummyAgent("beta", "second"))
        assert registry.list_names() == ["alpha", "beta"]

    def test_len(self):
        registry = AgentRegistry()
        assert len(registry) == 0
        registry.register(DummyAgent("x", "test"))
        assert len(registry) == 1

    def test_contains(self):
        registry = AgentRegistry()
        registry.register(DummyAgent("x", "test"))
        assert "x" in registry
        assert "y" not in registry

    def test_overwrite(self):
        registry = AgentRegistry()
        a1 = DummyAgent("x", "first")
        a2 = DummyAgent("x", "second")
        registry.register(a1)
        registry.register(a2)
        assert len(registry) == 1
        assert registry.get("x").description == "second"


class TestAgentDispatcher:
    def test_route_single_agent(self):
        registry = AgentRegistry()
        agent = DummyAgent("only", "Does everything")
        registry.register(agent)
        dispatcher = AgentDispatcher(registry)
        assert dispatcher.route("any task") is agent

    def test_route_by_keywords(self):
        registry = AgentRegistry()
        registry.register(DummyAgent("researcher", "Search and research information"))
        registry.register(DummyAgent("writer", "Write prose and documentation"))
        registry.register(DummyAgent("calculator", "Calculate math expressions"))
        dispatcher = AgentDispatcher(registry)

        result = dispatcher.route("search for papers about AI")
        assert result.name == "researcher"

        result = dispatcher.route("calculate 2 + 2")
        assert result.name == "calculator"

        result = dispatcher.route("write an email")
        assert result.name == "writer"

    def test_route_empty_registry(self):
        dispatcher = AgentDispatcher(AgentRegistry())
        assert dispatcher.route("any task") is None

    def test_default_agent(self):
        registry = AgentRegistry()
        registry.register(DummyAgent("fallback", "Default agent"))
        dispatcher = AgentDispatcher(registry, default_agent="fallback")

        result = dispatcher.route("completely unrelated task xyz")
        assert result.name == "fallback"

    def test_describe(self):
        registry = AgentRegistry()
        registry.register(DummyAgent("a", "Agent A"))
        registry.register(DummyAgent("b", "Agent B"))
        dispatcher = AgentDispatcher(registry)

        descs = dispatcher.describe()
        assert len(descs) == 2
        assert descs[0]["name"] == "a"

    async def test_dispatch(self):
        registry = AgentRegistry()
        registry.register(DummyAgent("worker", "A worker"))
        dispatcher = AgentDispatcher(registry)

        events = await dispatcher.dispatch("do something")
        assert len(events) == 2  # status + text


class TestWorkerAgent:
    def test_name_and_description(self):
        agent = DummyAgent("test", "desc")
        # WorkerAgent wraps a BaseAgent-like structure
        # Can't test without a real provider, but test the interface
        assert agent.name == "test"
        assert agent.description == "desc"

    async def test_base_agent_run(self):
        agent = DummyAgent("test", "desc")
        events = []
        async for event in agent.run("hello"):
            events.append(event)
        assert len(events) == 2
        assert events[0].type == "status"
        assert events[1].type == "text"
