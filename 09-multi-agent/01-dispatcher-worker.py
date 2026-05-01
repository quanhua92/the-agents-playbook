"""01-dispatcher-worker.py — Dispatcher routes tasks to specialized workers.

The core multi-agent pattern: a dispatcher classifies a task and routes
it to the appropriate worker. Each worker has a scoped tool set.

Key principle: the dispatcher decides, the worker does. A research agent
has search tools. A writer agent has no tools at all. Separation of
concerns prevents a researcher from accidentally sending emails.

This is a self-contained demo using simple keyword-based dispatching.
No SDK imports needed — we build from the agents protocol classes.
"""

import asyncio

from the_agents_playbook.agents import (
    AgentDispatcher,
    AgentRegistry,
    BaseAgent,
    AgentEvent,
)


class SimpleTool:
    """Minimal tool implementation for demo purposes."""

    def __init__(self, name: str, description: str, handler=None):
        self._name = name
        self._description = description
        self._handler = handler

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> object:
        if self._handler:
            return self._handler(**kwargs)
        return type("TR", (), {"output": f"{self._name} executed", "error": False})()


class ResearcherAgent(BaseAgent):
    """A researcher agent with search and lookup tools."""

    def __init__(self):
        self._tools = [
            SimpleTool("web_search", "Search the web for information"),
            SimpleTool("lookup_fact", "Look up facts in a knowledge base"),
        ]

    @property
    def name(self) -> str:
        return "researcher"

    @property
    def description(self) -> str:
        return "Search and research using web search and fact lookup tools"

    @property
    def tools(self) -> list:
        return self._tools

    async def run(self, prompt: str):
        yield AgentEvent(
            type="status",
            data={"message": f"[researcher] Starting research: {prompt[:50]}"},
            source=self.name,
        )
        yield AgentEvent(
            type="tool_call",
            data={"tool_name": "web_search", "arguments": {"query": prompt}},
            source=self.name,
        )
        yield AgentEvent(
            type="tool_result",
            data={"output": "Found 3 relevant results for: " + prompt[:50]},
            source=self.name,
        )
        yield AgentEvent(
            type="text",
            data={
                "text": f"Based on my research: {prompt} — here are the key findings..."
            },
            source=self.name,
        )


class WriterAgent(BaseAgent):
    """A writer agent with no tools — only generates text."""

    @property
    def name(self) -> str:
        return "writer"

    @property
    def description(self) -> str:
        return "Write and edit text content, prose, emails, and documentation"

    @property
    def tools(self) -> list:
        return []

    async def run(self, prompt: str):
        yield AgentEvent(
            type="status",
            data={"message": f"[writer] Starting composition: {prompt[:50]}"},
            source=self.name,
        )
        yield AgentEvent(
            type="text",
            data={"text": f"Here is the written output for: {prompt}\n\n..."},
            source=self.name,
        )


class CalculatorAgent(BaseAgent):
    """A calculator agent with math tools."""

    def __init__(self):
        self._tools = [
            SimpleTool("calculate", "Evaluate mathematical expressions"),
        ]

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Calculate mathematical expressions and numbers"

    @property
    def tools(self) -> list:
        return self._tools

    async def run(self, prompt: str):
        yield AgentEvent(
            type="status",
            data={"message": f"[calculator] Computing: {prompt[:50]}"},
            source=self.name,
        )
        yield AgentEvent(
            type="tool_call",
            data={"tool_name": "calculate", "arguments": {"expression": prompt}},
            source=self.name,
        )
        yield AgentEvent(
            type="tool_result",
            data={"output": "Result: 42"},
            source=self.name,
        )
        yield AgentEvent(
            type="text",
            data={"text": "The answer is 42."},
            source=self.name,
        )


async def main():
    # --- Build the registry ---
    print("=== Multi-Agent Dispatcher/Worker Pattern ===\n")

    registry = AgentRegistry()
    registry.register(ResearcherAgent())
    registry.register(WriterAgent())
    registry.register(CalculatorAgent())

    print("Registered agents:")
    for name in registry.list_names():
        agent = registry.get(name)
        tools_str = ", ".join(t.name for t in agent.tools) if agent.tools else "(none)"
        print(f"  {name:15s} tools=[{tools_str}]  — {agent.description}")
    print()

    # --- Create dispatcher ---
    dispatcher = AgentDispatcher(registry)

    # --- Route tasks ---
    tasks = [
        "Search the web for recent AI research papers",
        "Write a professional email to the team",
        "Calculate the fibonacci sequence up to 100",
        "What is 15 * 23 + 7?",
        "Find information about climate change",
    ]

    for task in tasks:
        print(f"--- Task: {task} ---")
        agent = dispatcher.route(task)
        if agent:
            print(f"  Routed to: {agent.name}")
            print(f"  Available tools: {[t.name for t in agent.tools]}")
            print("  Events:")
            async for event in agent.run(task):
                if event.type == "text":
                    print(f"    [text] {event.data.get('text', '')[:60]}")
                elif event.type == "tool_call":
                    print(f"    [tool] {event.data.get('tool_name')}")
                elif event.type == "status":
                    print(f"    [status] {event.data.get('message')[:60]}")
        else:
            print("  No agent available!")
        print()

    # --- Show dispatch descriptions ---
    print("=== Agent Descriptions ===\n")
    for desc in dispatcher.describe():
        print(f"  {desc['name']:15s} — {desc['description']}")


asyncio.run(main())
