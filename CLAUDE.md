# CLAUDE.md — Project Instructions for The Agents Playbook

## Project Overview

The Agents Playbook is an educational codebase for learning how to build LLM agents from scratch. It has three layers:

1. **SDK** (`src/the_agents_playbook/`) — from-scratch implementations with zero framework dependencies
2. **Playground** (`0X-xxx/`) — self-contained examples that use only the SDK
3. **LangGraph** (`langgraph-examples/`) — the same patterns using the LangGraph framework

## Standard Workflow

Every new concept **must** follow this sequence:

### 1. Identify the Problem

What real-world agent problem does this solve? Frame it as a concrete pain point (e.g., "conversations grow until they exceed the context window", not "we need compaction"). The problem drives the design — write it down before coding.

### 2. Plan the Solution

Design the minimal API surface. Use dataclasses over classes when possible. Use ABCs when you need polymorphism. Favor composition over inheritance. Keep the solution focused — one concept per change.

Key architectural constraints:
- The SDK has **zero** framework dependencies (only httpx, numpy, pydantic-settings)
- Playground examples should use real APIs when possible — the project `.env` has an OpenRouter key; prefer live LLM calls over mocks. Use mocks only when demonstrating internal SDK mechanics that don't need an LLM.
- All async — every `execute()`, `store()`, `recall()`, `run()` is async

### 3. Write SDK Code + Playground Example

**SDK** goes in `src/the_agents_playbook/<package>/`. Each package has:
- `protocol.py` — ABCs, dataclasses, core types
- Implementation files — concrete classes
- `__init__.py` — re-exports public API

**Playground** goes in `0X-xxx/NN-description.py`. Number sequentially within the chapter. The playground example must:
- Be runnable with `uv run python 0X-xxx/NN-description.py`
- Use the SDK and real APIs (`.env` provides OpenRouter key via `settings.py`)
- Print output that demonstrates the concept
- Include a `main()` + `if __name__ == "__main__"` guard

### 4. Write LangGraph Example

Create a mirror example in `langgraph-examples/0X-xxx/NN_description.py` that demonstrates the same concept using LangGraph. It should:
- Use the shared settings from `langgraph-examples/shared/settings.py`
- Use `create_react_agent` or `StateGraph` as appropriate
- Mirror the structure of the playground example but with LangGraph primitives
- Be runnable with `cd langgraph-examples && uv run python 0X-xxx/NN_description.py`

### 5. Write Problem & Solution Doc

Create `docs/problems/<kebab-case-name>.md` following this template:

```markdown
# <Title>

## Problem

2-3 sentences describing the real-world pain point.

## Solution

The architectural pattern. Reference specific classes with file paths and line numbers.
Use tables for class-to-file-to-role mappings. Include formulas if relevant.

## Code Reference

- `src/the_agents_playbook/<package>/<file>.py` — `ClassName` (line N), `function()` (line N)
- List all relevant source files

## Playground Example

- `0X-xxx/NN-description.py` — one-line description

## LangGraph Example

- `langgraph-examples/0X-xxx/NN_description.py` — one-line description
```

Check `docs/problems/` first — all 19 existing docs cover prior work. No duplicates.

### 6. Write Tests

Tests go in `tests/<package>/test_<module>.py`. Run with `uv run pytest tests/`.

## Project Structure

```
src/the_agents_playbook/     # SDK — zero framework deps
  providers/                 # BaseProvider, types, retry, credentials
  tools/                     # Tool ABC, ToolRegistry, ToolDispatcher, cache, MCP
  memory/                    # Fact, DualFileMemory, VectorStore, consolidation, session, segments, decay
  context/                   # LayerPriority, ContextBuilder, PromptTemplate, metadata, TokenBudget
  loop/                      # Agent (ReAct loop), AgentEvent, entropy scoring, ToolChainer, AgentConfig
  guardrails/                # RiskLevel, PermissionMiddleware, Prompter, HookSystem, AskUser, drafts
  workflows/                 # Workflow DAG, BaseStep, StepResult, WorkflowState
  claw/                      # RepairLoop, DegradationManager, SelfReviewer, LLMJudge, EvaluationHarness
  agents/                    # BaseAgent, WorkerAgent, AgentDispatcher, AgentRegistry
  utils/                     # vectors, schema helpers

0X-xxx/                      # Playground examples (self-contained, SDK-only)
  NN-description.py

langgraph-examples/          # Same patterns using LangGraph framework
  0X-xxx/NN_description.py
  shared/                    # Shared utilities (settings, vectors, compactor)
  run_all.sh

docs/problems/               # 19 problem/solution docs covering every concept
  <name>.md

tests/                       # Unit tests (612 total)
  <package>/test_<module>.py
```

## Chapter Map

| Chapter | SDK Package | Topic |
|---------|------------|-------|
| 01-basic-calls | providers | HTTP calls, streaming, structured output, embeddings |
| 02-tools | tools | Tool ABC, registry, dispatch, caching, MCP |
| 03-memory | memory | Facts, dual-file, vector search, consolidation, sessions, segments, decay |
| 04-context | context | Layers, builder, templates, metadata, token budget |
| 05-the-loop | loop | Agent events, config, entropy, ReAct, chaining, streaming |
| 06-human-in-the-loop | guardrails | Risk levels, permissions, prompter, hooks, ask-user, drafts |
| 07-workflows | workflows | Step protocol, plan-and-build, DAG, state, hooks, concurrency |
| 08-the-claw | claw | Self-repair, degradation, self-review, evaluation, LLM judge |
| 09-multi-agent | agents | Dispatcher/worker, supervisor |

## Code Conventions

- All I/O is async — `async def`, `await`, `AsyncGenerator`
- Use `dataclass` for data containers, `ABC` for polymorphism
- Use `pydantic.BaseModel` only for API-facing types (RequestLog, RetryConfig, MessageRequest, etc.)
- `from __future__ import annotations` is not used — type hints are plain strings for forward refs
- Lazy imports to break circular dependencies (import inside method bodies when needed)
- Tests use `pytest` + `pytest-asyncio` with `@pytest.mark.asyncio`
- Provider tests use `httpx.MockTransport` for no-network test isolation
- Settings via `src/the_agents_playbook/settings.py` (pydantic-settings, reads from env)

## Key Patterns

- **Circular imports**: When `evaluation.py` and `agent_evaluator.py` import from each other, use lazy imports inside methods, not at module level.
- **Backward compatibility**: New methods on ABCs get default implementations (return `None`, empty list, etc.) so existing subclasses don't break.
- **Tool errors are data, not exceptions**: `ToolDispatcher.dispatch_one()` catches all errors and returns `ToolResult(error=True)` so the agent loop can feed them to the LLM.
- **Hook errors are isolated**: `HookSystem.emit()` catches errors per-handler so one broken hook doesn't crash the pipeline.

## Running Tests

```bash
uv run pytest tests/ -q          # all 612 tests
uv run pytest tests/<package>/   # single package
uv run python 0X-xxx/NN-desc.py  # playground example
cd langgraph-examples && bash run_all.sh  # all LangGraph examples
```

## Linting & Type Checking

Before committing, always run ruff and pyright. Use `uvx` to run them without adding to project dependencies:

```bash
uvx ruff check .                 # lint
uvx ruff format --check .        # format check
uvx ruff format .                # auto-format
npx pyright .                    # type check (error level)
npx pyright --level warning .    # type check with warnings
```
