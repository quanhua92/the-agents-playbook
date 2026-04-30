# Agent Loop and Entropy Routing

## Problem

An agent needs a loop that reasons, picks tools, observes results, and decides when to stop. This is the ReAct (Reason + Act) pattern. But when should the agent ask the user for help instead of guessing? When tool selection is uncertain, blindly calling the wrong tool wastes tokens and can cause side effects. You need a way to measure the agent's uncertainty and route accordingly.

## Solution

The `Agent` class implements the core ReAct loop. Shannon entropy scores measure tool selection uncertainty. High entropy triggers user input; low entropy means the agent is confident.

### The Agent Loop

`Agent` (`src/the_agents_playbook/loop/agent.py:39`) ties together provider, tools, memory, and context:

1. Store user message in memory
2. Recall relevant memories
3. Build context (static + semi-stable + dynamic layers)
4. Send to LLM with tool specs
5. If LLM requests tool calls: dispatch via `ToolDispatcher`, feed results back to LLM
6. Repeat until LLM returns text (no tool calls) or max iterations reached

The loop yields `AgentEvent` objects for real-time streaming. `AgentConfig` (`src/the_agents_playbook/loop/config.py:8`) controls: `max_tool_iterations=25`, `on_error="abort"`, `entropy_threshold=1.5`, `max_chain_length=3`.

### AgentEvent Types

`AgentEvent` (`src/the_agents_playbook/loop/protocol.py:7`) has a `type` field:

- `"text"` — final text response
- `"text_delta"` — streaming token (from `run_streaming()`)
- `"tool_call"` — LLM requested a tool call
- `"tool_result"` — tool execution completed
- `"status"` — lifecycle status messages
- `"error"` — error occurred

`TurnResult` (`protocol.py:35`) summarizes one LLM call + tool executions: events list, tool_calls_made count, final_response, error.

### Shannon Entropy

`shannon_entropy()` (`src/the_agents_playbook/loop/scoring.py:6`) computes `H(p) = -Σ p_i * log2(p_i)` in bits. `score_tools()` (`scoring.py:25`) converts tool relevance scores to probabilities and returns entropy.

- Uniform distribution over 4 tools: H = 2.0 bits (maximum uncertainty)
- One tool at 100%: H = 0.0 bits (certain)
- The `entropy_threshold` in `AgentConfig` determines when to ask the user

### Tool Chaining

`ToolChainer` (`src/the_agents_playbook/loop/chains.py:34`) executes sequential tool calls with re-scoring between steps. After each step, remaining tools are scored. If entropy drops below threshold or chain reaches max length, execution stops. `ToolChain` (`chains.py:20`) records the execution history.

### Routing Logic

When entropy exceeds the threshold (`entropy_threshold=1.5`), the agent should ask the user instead of guessing. This is the "ask don't guess" principle: high uncertainty means the agent needs human input.

## Code Reference

- `src/the_agents_playbook/loop/agent.py` — `Agent` class (line 39)
- `src/the_agents_playbook/loop/protocol.py` — `AgentEvent` (line 7), `TurnResult` (line 35)
- `src/the_agents_playbook/loop/scoring.py` — `shannon_entropy()` (line 6), `score_tools()` (line 25)
- `src/the_agents_playbook/loop/chains.py` — `ToolChainer` (line 34), `ToolChain` (line 20)
- `src/the_agents_playbook/loop/config.py` — `AgentConfig` (line 8)

## Playground Examples

- `05-the-loop/01-agent-events.py` — agent event types and streaming
- `05-the-loop/02-agent-config.py` — agent configuration options
- `05-the-loop/03-shannon-entropy.py` — entropy scoring and uncertainty measurement
- `05-the-loop/04-react-agent.py` — full ReAct agent loop
- `05-the-loop/05-tool-chaining.py` — sequential tool execution with re-scoring
- `05-the-loop/06-entropy-routing.py` — route to user input when entropy is high

## LangGraph Examples

- `langgraph-examples/05-the-agent/01_simple_graph.py` — basic LangGraph agent
- `langgraph-examples/05-the-agent/02_react_agent.py` — ReAct agent with `create_react_agent`
- `langgraph-examples/05-the-agent/03_conditional_edges.py` — conditional routing in agent graphs
- `langgraph-examples/05-the-agent/04_tool_chaining.py` — multi-step tool execution
