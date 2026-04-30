# Multi-Agent Dispatcher/Worker Pattern

## Problem

The system had a single monolithic agent with access to all tools. In production, you want specialized agents: a researcher with search tools, a calculator with math tools, a writer with no tools at all. Giving every tool to every agent is a security risk — a research agent shouldn't be able to send emails.

There was also no mechanism to route tasks to different agents based on the task type.

## Solution

A new `agents/` package with four files implements the dispatcher/worker pattern:

| Class | File | Role |
|---|---|---|
| `BaseAgent` | `agents/protocol.py` | ABC defining name, description, tools, run() |
| `WorkerAgent` | `agents/worker.py` | Wraps the existing Agent with a scoped ToolRegistry |
| `AgentRegistry` | `agents/registry.py` | Maps agent names to BaseAgent instances |
| `AgentDispatcher` | `agents/dispatcher.py` | Routes tasks to the best matching worker |

### The core principle

**The dispatcher decides, the worker does.**

- The dispatcher has **no tools** itself — it only classifies and routes
- Each worker has a **scoped tool set** — a researcher gets search but not email
- Separation of concerns prevents runaway agent behavior

### BaseAgent protocol

Every agent must implement four properties and one method:

```python
from the_agents_playbook.agents import BaseAgent, AgentEvent

class MyAgent(BaseAgent):
    @property
    def name(self) -> str: ...         # unique identifier for routing

    @property
    def description(self) -> str: ...  # used by dispatcher to match tasks

    @property
    def tools(self) -> list: ...       # scoped tool set for this agent

    async def run(self, prompt: str) -> AsyncGenerator[AgentEvent, None]:
        ...                             # execute the agent
```

### Registration and dispatch

```python
from the_agents_playbook.agents import AgentRegistry, AgentDispatcher

registry = AgentRegistry()
registry.register(ResearchAgent())
registry.register(WriterAgent())
registry.register(CalculatorAgent())

dispatcher = AgentDispatcher(registry)

# Route based on task description
agent = dispatcher.route("Search for papers about transformers")
# → returns ResearchAgent (keyword overlap with "Search")

agent = dispatcher.route("Calculate 15 * 23")
# → returns CalculatorAgent (keyword overlap with "Calculate")

# Route and execute in one call
events = await dispatcher.dispatch("Write a haiku about coding")
```

### Routing algorithm

The dispatcher uses keyword overlap between the task and each agent's description:

```python
task_words = {"search", "for", "papers", "about", "transformers"}
researcher_desc = {"search", "and", "research", "information"}
# overlap = 2 ("search", "research") → best match
```

This is transparent and testable. In production, this could be replaced with an LLM-based classifier without changing the interface.

### WorkerAgent

Wraps the existing `Agent` class with a scoped `ToolRegistry`:

```python
from the_agents_playbook.agents import WorkerAgent

worker = WorkerAgent(
    name="researcher",
    description="Search and research information",
    provider=provider,
    tools=[search_tool, lookup_tool],
)

# Delegates to the internal Agent, but tags all events with source=self.name
async for event in worker.run("Find papers about NLP"):
    print(event)  # event.source == "researcher"
```

### Supervisor pattern

A higher-level pattern where a supervisor decomposes complex tasks:

```
1. Supervisor receives: "Write a report on climate change"
2. Decomposes into: ["Research climate change", "Write summary", "Review summary"]
3. Routes each subtask to the best worker
4. Collects all worker results
5. Synthesizes into a final combined answer
```

This is recursive — a worker could itself be a supervisor with its own sub-workers.

### AgentNotFoundError

Attempting to look up an agent that doesn't exist raises a typed error:

```python
registry.get("nonexistent")
# AgentNotFoundError: Agent 'nonexistent' not found in registry
```

### AgentEvent

Multi-agent events carry a `source` field identifying which agent produced them:

```python
event = AgentEvent(type="text", data={"text": "..."}, source="researcher")
```

This lets consumers attribute output to the correct worker in multi-agent traces.

## Code Reference

- `src/the_agents_playbook/agents/protocol.py` — `BaseAgent` ABC, `AgentEvent`
- `src/the_agents_playbook/agents/worker.py` — `WorkerAgent` wrapping `Agent` with scoped tools
- `src/the_agents_playbook/agents/dispatcher.py` — `AgentDispatcher` with keyword-based routing
- `src/the_agents_playbook/agents/registry.py` — `AgentRegistry`, `AgentNotFoundError`

## Playground Examples

- `09-multi-agent/01-dispatcher-worker.py` — researcher vs writer, dispatcher routes by task
- `09-multi-agent/02-supervisor.py` — supervisor decomposes task, delegates to workers, synthesizes

## LangGraph Examples

- `langgraph-examples/09-multi-agent/01_supervisor.py` — supervisor graph with conditional routing
- `langgraph-examples/09-multi-agent/02_parallel_agents.py` — parallel fan-out with `Send()` API
