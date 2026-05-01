"""Agent dispatcher — classifies tasks and routes to the appropriate worker.

The dispatcher is the single most important production pattern for
multi-agent systems. It decides which worker should handle a given
task based on the task description and available agents.

Key principle: the dispatcher decides, the worker does. The dispatcher
has NO tools itself (no search, no write, no email) — only memory
and dispatch capabilities.

Usage:
    registry = AgentRegistry()
    registry.register(WorkerAgent(name="researcher", ...))
    registry.register(WorkerAgent(name="writer", ...))

    dispatcher = AgentDispatcher(registry)
    worker = dispatcher.route("Search for papers about transformers")
    # Returns the researcher agent
"""

import logging
from typing import Any

from .protocol import BaseAgent
from .registry import AgentRegistry

logger = logging.getLogger(__name__)


class AgentDispatcher:
    """Routes tasks to the appropriate worker agent.

    The dispatcher examines the task description and matches it against
    agent descriptions to find the best fit. In a production system,
    this could use an LLM for classification. Here we use keyword
    matching for transparency and testability.

    Usage:
        dispatcher = AgentDispatcher(registry)
        agent = dispatcher.route("Calculate 2+2")
        async for event in agent.run("Calculate 2+2"):
            print(event)
    """

    def __init__(
        self,
        registry: AgentRegistry,
        default_agent: str | None = None,
    ) -> None:
        self._registry = registry
        self._default_agent = default_agent

    @property
    def registry(self) -> AgentRegistry:
        return self._registry

    def route(self, task: str) -> BaseAgent | None:
        """Route a task to the best matching agent.

        Scores each agent based on keyword overlap between the task
        and the agent's description. Returns the highest-scoring agent.

        Args:
            task: The task description to route.

        Returns:
            The best matching BaseAgent, or None if no agents registered.
        """
        agents = self._registry.list_agents()
        if not agents:
            return None

        if len(agents) == 1:
            return agents[0]

        # Score each agent by keyword overlap
        task_words = set(task.lower().split())
        best_agent = None
        best_score = -1

        for agent in agents:
            desc_words = set(agent.description.lower().split())
            name_words = set(agent.name.lower().split("_"))
            all_words = desc_words | name_words

            overlap = len(task_words & all_words)
            if overlap > best_score:
                best_score = overlap
                best_agent = agent

        if best_agent is None:
            if self._default_agent:
                return self._registry.get(self._default_agent)
            return agents[0] if agents else None

        logger.info(
            "Routed task '%s' to agent '%s' (score=%d)",
            task[:50],
            best_agent.name,
            best_score,
        )
        return best_agent

    async def dispatch(self, task: str) -> list[Any]:
        """Route a task and run the selected agent.

        Convenience method that routes and executes in one call.

        Args:
            task: The task description.

        Returns:
            List of events from the executed agent.
        """
        agent = self.route(task)
        if agent is None:
            return []

        events = []
        async for event in agent.run(task):
            events.append(event)
        return events

    def describe(self) -> list[dict[str, str]]:
        """Return descriptions of all registered agents."""
        return [
            {"name": agent.name, "description": agent.description}
            for agent in self._registry.list_agents()
        ]
