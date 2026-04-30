"""02-supervisor.py — Supervisor decomposes tasks and delegates to workers.

The supervisor pattern extends dispatcher/worker: instead of routing to
a single worker, the supervisor decomposes a complex task into subtasks,
delegates each to the appropriate worker, and synthesizes the results.

This is a more advanced pattern where:
1. Supervisor receives a complex task
2. Breaks it into subtasks
3. Routes each subtask to the best worker
4. Collects all results
5. Synthesizes a final answer

No SDK imports — self-contained demo using the agents protocol.
"""

import asyncio

from the_agents_playbook.agents import (
    AgentDispatcher,
    AgentEvent,
    AgentRegistry,
    BaseAgent,
)


class ResearchWorker(BaseAgent):
    """Worker that handles research tasks."""

    @property
    def name(self) -> str:
        return "researcher"

    @property
    def description(self) -> str:
        return "Research and search for information on any topic"

    @property
    def tools(self) -> list:
        return []

    async def run(self, prompt: str):
        yield AgentEvent(
            type="text",
            data={"text": f"[Research] Investigated: {prompt}\nFound: Key facts and data points."},
            source=self.name,
        )


class WriterWorker(BaseAgent):
    """Worker that handles writing tasks."""

    @property
    def name(self) -> str:
        return "writer"

    @property
    def description(self) -> str:
        return "Write prose, summaries, and documentation"

    @property
    def tools(self) -> list:
        return []

    async def run(self, prompt: str):
        yield AgentEvent(
            type="text",
            data={"text": f"[Writing] Composed content about: {prompt}\nDraft: Well-structured paragraphs..."},
            source=self.name,
        )


class ReviewWorker(BaseAgent):
    """Worker that handles review/analysis tasks."""

    @property
    def name(self) -> str:
        return "reviewer"

    @property
    def description(self) -> str:
        return "Review, analyze, and provide feedback on content"

    @property
    def tools(self) -> list:
        return []

    async def run(self, prompt: str):
        yield AgentEvent(
            type="text",
            data={"text": f"[Review] Analyzed: {prompt}\nFeedback: Clear structure, minor improvements needed."},
            source=self.name,
        )


class SupervisorAgent:
    """Supervisor that decomposes tasks and delegates to workers.

    Unlike a simple dispatcher, the supervisor:
    1. Breaks complex tasks into subtasks
    2. Runs each subtask with the best worker
    3. Collects all worker results
    4. Synthesizes a final combined answer
    """

    def __init__(self, registry: AgentRegistry) -> None:
        self._dispatcher = AgentDispatcher(registry)

    def decompose(self, task: str) -> list[str]:
        """Break a complex task into subtasks.

        In production, this would use an LLM. Here we use a simple
        rule-based decomposition for demonstration.
        """
        if "report" in task.lower():
            return [
                f"Research findings for: {task}",
                f"Write summary of: {task}",
                f"Review the summary of: {task}",
            ]
        if "and" in task.lower():
            parts = task.split(" and ", 1)
            if len(parts) == 2:
                return [parts[0].strip(), parts[1].strip()]
        return [task]

    async def run(self, task: str) -> list[AgentEvent]:
        """Execute a task using supervisor decomposition.

        Returns all events from all worker executions.
        """
        print(f"\n[Supervisor] Received task: {task}")

        subtasks = self.decompose(task)
        print(f"[Supervisor] Decomposed into {len(subtasks)} subtask(s):")
        for i, st in enumerate(subtasks, 1):
            print(f"  {i}. {st}")

        all_events: list[AgentEvent] = []

        for i, subtask in enumerate(subtasks, 1):
            print(f"\n[Supervisor] Delegating subtask {i}: {subtask[:60]}")

            agent = self._dispatcher.route(subtask)
            if agent:
                print(f"[Supervisor] Routed to: {agent.name}")
                async for event in agent.run(subtask):
                    all_events.append(event)
                    if event.type == "text":
                        print(f"  [{agent.name}] {event.data.get('text', '')[:80]}")
            else:
                print(f"  [Supervisor] No worker available for: {subtask}")

        # Synthesize
        print(f"\n[Supervisor] Synthesizing results from {len(subtasks)} subtask(s)...")
        all_events.append(AgentEvent(
            type="text",
            data={"text": f"[Supervisor] Final synthesis combining all {len(subtasks)} worker outputs."},
            source="supervisor",
        ))

        return all_events


async def main():
    print("=== Supervisor Pattern ===")
    print("The supervisor decomposes, delegates, and synthesizes.\n")

    registry = AgentRegistry()
    registry.register(ResearchWorker())
    registry.register(WriterWorker())
    registry.register(ReviewWorker())

    supervisor = SupervisorAgent(registry)

    # Task 1: Simple delegation (no decomposition needed)
    print("=" * 60)
    events = await supervisor.run("Research the history of Python programming language")

    # Task 2: Complex task requiring decomposition
    print("\n" + "=" * 60)
    events = await supervisor.run(
        "Write a report on climate change effects on ocean ecosystems"
    )

    # Task 3: Multi-part task
    print("\n" + "=" * 60)
    events = await supervisor.run(
        "Research renewable energy trends and write a summary"
    )

    print("\n\n=== Pattern Summary ===\n")
    print("1. Supervisor receives complex task")
    print("2. Decomposes into subtasks (research, write, review)")
    print("3. Routes each subtask to the best worker via dispatcher")
    print("4. Workers execute independently with scoped tools")
    print("5. Supervisor synthesizes all results into final answer")
    print("\nThe pattern is recursive: a worker could itself be a supervisor.")


asyncio.run(main())
