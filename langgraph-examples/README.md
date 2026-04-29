# LangGraph Examples — The Agents Playbook

## Why

The root project builds an AI agent framework from scratch using raw `httpx`, `numpy`, and hand-rolled implementations of every concept: ReAct loops, tool dispatch, memory persistence, workflow DAGs, guardrails, and self-repair. This teaches you **how these things work under the hood**.

This folder demonstrates the **same concepts using LangGraph and LangChain** — the production-grade libraries that provide these capabilities as built-in features. It answers the question: *"Now that I understand how it works, how do I do this in practice?"*

Each chapter here mirrors a chapter in the root project, with side-by-side concept mapping.

## Concept Mapping

| Root SDK (from scratch) | LangGraph Equivalent |
|---|---|
| `BaseProvider` + raw `httpx` POST | `ChatOpenAI` / `ChatAnthropic` from langchain |
| `Tool` ABC + `ToolRegistry` + `ToolResult` | `@tool` decorator + `ToolNode` |
| `Agent` class (240-line ReAct loop) | `create_react_agent` (one function call) |
| `SessionPersistence` (JSONL file save/load) | `MemorySaver` / `SqliteSaver` checkpointing |
| `SessionCompactor` (token-aware summarization) | Reused from root (copied to `shared/`) |
| `Workflow` (DAG runner with Kahn's algorithm) | `StateGraph` with conditional edges |
| `WorkflowState` (mutable dataclass) | TypedDict state with `Annotated` reducers |
| `ContextBuilder` + `ContextLayer` + `LayerPriority` | System message composition in graph nodes |
| `PermissionMiddleware` + `RiskLevel` | `interrupt()` + `Command` resumption |
| `Shannon entropy` routing | Conditional edge functions |
| `RepairLoop` (retry wrapper) | Graph edge looping back to retry node |
| `EvaluationHarness` | Standalone harness calling compiled graph |
| `Settings` + `validate_config` | Copied to `shared/settings.py` + LLM factory helpers |

## How

### Setup

```bash
cd langgraph-examples
uv sync
cp ../.env .env   # or create your own .env with API keys
```

### Dependencies

- `langgraph` — Graph-based agent framework (state, nodes, edges, checkpointing)
- `langchain-openai` — `ChatOpenAI` for OpenAI/OpenRouter APIs
- `langchain-core` — Message types, tool base classes, callbacks
- `langchain-anthropic` — `ChatAnthropic` for Claude APIs
- `httpx`, `numpy`, `pydantic-settings` — Shared utilities (embeddings, vectors, config)

### Running Examples

```bash
cd langgraph-examples
uv run python 01-basic-calls/01_basic_chat.py
uv run python 05-the-agent/02_react_agent.py
```

## Project Structure

```
langgraph-examples/
├── shared/                     # Shared utilities (independent from root SDK)
│   ├── settings.py             # Settings, validate_config, LLM factory helpers
│   ├── vectors.py              # cosine_similarity, normalize (pure numpy)
│   ├── embeddings.py           # OpenAIEmbeddingProvider (httpx-based)
│   ├── vector_store.py         # InMemoryVectorStore (numpy cosine similarity)
│   └── compactor.py            # SessionCompactor (token-aware summarization)
├── 01-basic-calls/             # LLM basics without graph concepts
├── 02-tools/                   # Tool definition and dispatch
├── 03-memory/                  # Checkpointing and conversation persistence
├── 04-state/                   # Typed state, reducers, context composition
├── 05-the-agent/               # Graph construction and ReAct agents
├── 06-human-in-the-loop/       # Interrupt and human approval
├── 07-workflows/               # Plan-execute and parallel execution
├── 08-the-claw/                # Error handling, retry, evaluation
└── pyproject.toml
```

## What — Chapter Details

### 01 — Basic Calls

Replacing raw `httpx` POST requests with LangChain's chat model abstraction.

| File | Concept | Root Equivalent |
|------|---------|----------------|
| `01_basic_chat.py` | `ChatOpenAI.invoke()` — basic chat completion | `02-basic-chat.py` |
| `02_structured_output.py` | `with_structured_output()` — typed Pydantic output | `03-structured-output.py` |
| `03_tool_calling.py` | `@tool` decorator + `.bind_tools()` — native tool calling | `04-tool-choice.py` |
| `04_streaming.py` | `llm.astream()` — token-by-token streaming | `06-streaming.py` |

**Key insight**: No raw HTTP, no manual JSON parsing, no SSE line handling. LangChain handles all of it.

### 02 — Tools

Replacing the custom `Tool` ABC, `ToolRegistry`, and dispatcher with LangChain's tool system.

| File | Concept | Root Equivalent |
|------|---------|----------------|
| `01_tool_protocol.py` | `@tool`, `StructuredTool` — three ways to define tools | `01-tool-protocol.py` |
| `02_tool_node.py` | `ToolNode` — automatic dispatch from `AIMessage.tool_calls` | `02-tool-registry.py` + `03-tool-dispatcher.py` |
| `03_bound_tools.py` | `.bind_tools()` — let the LLM decide when to call tools | `MessageRequest(tools=...)` pattern |

**Key insight**: `ToolNode` is a single class that replaces the entire `ToolRegistry` + `Dispatcher` + `ToolResult` pattern. It automatically handles parallel dispatch of multiple tool calls.

### 03 — Memory

Replacing JSONL session persistence and manual compaction with LangGraph's built-in checkpointing.

| File | Concept | Root Equivalent |
|------|---------|----------------|
| `01_checkpoint.py` | `MemorySaver` — thread-based conversation persistence | `05-session-persistence.py` |
| `02_thread_history.py` | Multiple `thread_id`s — independent conversation contexts | N/A (root uses separate JSONL files) |
| `03_long_context.py` | `SessionCompactor` — token-aware conversation summarization | Compaction pattern in `06-real-embeddings.py` |

**Key insight**: `MemorySaver` persists full graph state (including messages) keyed by `thread_id`. No JSONL files, no manual save/load. Swap with `SqliteSaver` for disk persistence. The `SessionCompactor` is reused because LangGraph has no built-in compaction.

### 04 — State

Replacing the root's mutable `WorkflowState` with LangGraph's immutable TypedDict state and reducers.

| File | Concept | Root Equivalent |
|------|---------|----------------|
| `01_typed_state.py` | TypedDict state — nodes return partial updates | `03-workflow-state.py` |
| `02_reducer_state.py` | `add_messages`, `operator.add` — field-level merge strategies | N/A (root has no reducer concept) |
| `03_context_layers.py` | System message composition from priority-sorted layers | `04-context/04-context-builder.py` |

**Key insight**: LangGraph state is immutable — nodes return partial updates that reducers merge. This replaces the root's mutable `WorkflowState.merge_context()`. The `Annotated[list, add_messages]` reducer automatically handles message deduplication and ordering.

### 05 — The Agent

Replacing the 240-line hand-rolled `Agent` ReAct loop with LangGraph's `create_react_agent`.

| File | Concept | Root Equivalent |
|------|---------|----------------|
| `01_simple_graph.py` | `StateGraph` basics — nodes, edges, START, END | Foundational (no direct equivalent) |
| `02_react_agent.py` | `create_react_agent` — full ReAct agent in one call | `04-react-agent.py` |
| `03_conditional_edges.py` | Conditional routing based on state | `06-entropy-routing.py` |
| `04_tool_chaining.py` | Multi-step tool use within agent | `05-tool-chaining.py` |

**Key insight**: `create_react_agent(llm, tools)` replaces the entire `Agent.__init__()` + `Agent.run()` implementation. The prebuilt agent internally uses a `StateGraph` with `ToolNode` and conditional edges — the same pattern you'd build by hand in `01_simple_graph.py`.

### 06 — Human in the Loop

Replacing `PermissionMiddleware` + `AskUserQuestion` with LangGraph's native `interrupt()` mechanism.

| File | Concept | Root Equivalent |
|------|---------|----------------|
| `01_interrupt.py` | `interrupt()` — pause graph, wait for human input | `02-permission-middleware.py` |
| `02_command_resuming.py` | `Command(resume=...)` — branching after human decision | N/A (root only supports pre-execution checks) |

**Key insight**: LangGraph's `interrupt()` is more powerful than the root's permission checks. It pauses mid-execution, saves state to the checkpoint, and the human's response can influence subsequent routing via conditional edges. Requires a checkpointer (`MemorySaver`).

### 07 — Workflows

Replacing the root's DAG runner (Kahn's algorithm + `asyncio.gather`) with native LangGraph patterns.

| File | Concept | Root Equivalent |
|------|---------|----------------|
| `01_plan_execute.py` | Plan-then-execute with conditional loop edge | `02-plan-and-build.py` + `04-workflow-dag.py` |
| `02_parallel_nodes.py` | `Send()` API — dynamic fan-out parallelism | `06-concurrent-steps.py` |

**Key insight**: Graph edges naturally encode DAG structure — no need for Kahn's algorithm. The `Send()` API dynamically creates parallel branches based on state, which is more flexible than `asyncio.gather` with pre-computed batches.

### 08 — The CLAW

Replacing `RepairLoop`, `DegradationManager`, and `EvaluationHarness` with graph-native patterns.

| File | Concept | Root Equivalent |
|------|---------|----------------|
| `01_error_edges.py` | Error handling via conditional edge routing + retry loop | `01-self-repair.py` + `02-graceful-degradation.py` |
| `02_retry_loop.py` | Retry with max attempts and conditional edge back | `01-self-repair.py` |
| `03_evaluation.py` | Evaluation harness measuring agent pass rate and latency | `03-evaluation-harness.py` |

**Key insight**: In LangGraph, retries are modeled as a loop in the graph topology: `attempt -> conditional_edge -> (retry: loop back | done: succeed | fail: exhausted)`. The graph itself becomes the retry mechanism — no separate `RepairLoop` wrapper needed.

## Shared Module

The `shared/` module contains utilities copied from the root SDK that LangGraph doesn't provide. These are independent from the root package — no `from the_agents_playbook import ...`.

| File | Source | Lines |
|------|--------|-------|
| `settings.py` | `src/the_agents_playbook/settings.py` + LLM factory helpers | ~100 |
| `vectors.py` | `src/the_agents_playbook/utils/vectors.py` (verbatim) | ~30 |
| `embeddings.py` | `src/the_agents_playbook/memory/embedding_provider.py` | ~80 |
| `vector_store.py` | `src/the_agents_playbook/memory/vector_memory.py` + `protocol.py` | ~100 |
| `compactor.py` | `src/the_agents_playbook/memory/session.py` (SessionCompactor only) | ~100 |

Total: ~410 lines of shared code. These are stable, pure-utility modules with no LangGraph-specific logic.

## Configuration

Uses the same `.env` format as the root project:

```
OPENAI_API_KEY=sk-or-v1-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-oss-20b

ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_BASE_URL=https://api.anthropic.com/v1
ANTHROPIC_MODEL=claude-sonnet-4-6

EMBEDDING_API_KEY=sk-or-v1-...
EMBEDDING_BASE_URL=https://openrouter.ai/api/v1
EMBEDDING_MODEL=openai/text-embedding-3-small
```

## License

MIT
