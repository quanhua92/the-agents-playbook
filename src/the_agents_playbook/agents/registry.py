"""Agent registry — maps agent names to BaseAgent instances.

The registry is the central directory for all agents in a multi-agent
system. The dispatcher queries it to find available agents.

Usage:
    registry = AgentRegistry()
    registry.register(my_research_agent)
    registry.register(my_writer_agent)

    researcher = registry.get("researcher")
    all_agents = registry.list_agents()
"""

import logging
from typing import Any

from .protocol import BaseAgent

logger = logging.getLogger(__name__)


class AgentNotFoundError(Exception):
    """Raised when looking up an agent that isn't registered."""

    def __init__(self, name: str):
        super().__init__(f"Agent '{name}' not found in registry")
        self.name = name


class AgentRegistry:
    """Registry mapping agent names to BaseAgent instances.

    Usage:
        registry = AgentRegistry()
        registry.register(agent)
        agent = registry.get("my-agent")
        all = registry.list_agents()
    """

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        """Register an agent by its name. Overwrites if already exists."""
        self._agents[agent.name] = agent
        logger.debug("Registered agent: %s", agent.name)

    def get(self, name: str) -> BaseAgent:
        """Look up a registered agent by name. Raises AgentNotFoundError."""
        if name not in self._agents:
            raise AgentNotFoundError(name)
        return self._agents[name]

    def list_agents(self) -> list[BaseAgent]:
        """Return all registered agents."""
        return list(self._agents.values())

    def list_names(self) -> list[str]:
        """Return names of all registered agents."""
        return list(self._agents.keys())

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        return name in self._agents
