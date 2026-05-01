"""Worker agent — wraps provider, tools, and context into a single agent.

A WorkerAgent is essentially the existing Agent class refactored to
implement the BaseAgent protocol. Each worker has a scoped tool set,
preventing it from accessing tools meant for other agents.

Usage:
    worker = WorkerAgent(
        name="researcher",
        description="Finds information using search and lookup tools",
        provider=provider,
        tools=[search_tool, lookup_tool],
    )
    async for event in worker.run("Find papers about transformers"):
        print(event)
"""

from collections.abc import AsyncGenerator

from ..loop.agent import Agent, AgentConfig
from ..providers.base import BaseProvider
from ..tools.protocol import Tool
from ..tools.registry import ToolRegistry
from .protocol import AgentEvent, BaseAgent


class WorkerAgent(BaseAgent):
    """A worker agent with a scoped tool set.

    Wraps the core Agent class and implements BaseAgent for use in
    multi-agent dispatch systems.

    Attributes:
        _name: Agent identifier.
        _description: What this agent does.
        _agent: The underlying Agent instance.
        _tools: This agent's scoped tools.
    """

    def __init__(
        self,
        name: str,
        description: str,
        provider: BaseProvider,
        tools: list[Tool],
        config: AgentConfig | None = None,
    ) -> None:
        self._name = name
        self._description = description
        self._tools = tools
        self._registry = ToolRegistry()
        for tool in tools:
            self._registry.register(tool)
        self._agent = Agent(
            provider=provider,
            registry=self._registry,
            config=config,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools)

    @property
    def agent(self) -> Agent:
        """Access the underlying Agent instance."""
        return self._agent

    async def run(self, prompt: str) -> AsyncGenerator[AgentEvent, None]:
        """Run the agent and adapt AgentEvents to multi-agent AgentEvents."""
        async for event in self._agent.run(prompt):
            yield AgentEvent(
                type=event.type,
                data=event.data,
                source=self._name,
            )

    async def close(self) -> None:
        """Clean up provider resources."""
        await self._agent.close()
