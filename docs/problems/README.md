# Problem & Solution Index

Every agent problem this codebase solves, organized by chapter.

## 01 — Providers & API Layer

| Doc | Problem |
|-----|---------|
| [Provider Resilience](provider-resilience.md) | LLM API calls fail constantly — rate limits, auth errors, timeouts. How do you make agents resilient? |
| [Structured Output and Tool Choice](structured-output-and-tool-choice.md) | LLMs return free-form text, but agents need structured JSON. How do you guarantee output shape? |
| [Streaming Responses](streaming-responses.md) | Non-streaming calls block for 10+ seconds. How do you stream token-by-token from the API? |

## 02 — Tools

| Doc | Problem |
|-----|---------|
| [Tool Protocol and Dispatch](tool-protocol-and-dispatch.md) | Agents need a uniform way to define, discover, and invoke tools. How do you build a tool system from scratch? |
| [Tool Caching and MCP](tool-caching-and-mcp.md) | Same tool calls repeat across conversations, wasting tokens. How do you cache results and bridge to external MCP servers? |

## 03 — Memory

| Doc | Problem |
|-----|---------|
| [Embedding and Vector Memory](embedding-and-vector-memory.md) | Keyword search misses semantic matches. How do you store and retrieve memories by meaning? |
| [Dual-File Memory and Consolidation](dual-file-memory-and-consolidation.md) | Raw conversation logs are noisy and hard to query. How do you structure long-term memory? |
| [Session Persistence and Compaction](session-persistence-and-compaction.md) | Conversations grow until they exceed the context window. How do you persist sessions and compress old turns? |
| [Segmented Memory with Tiered Decay](segmented-memory.md) | All memories are treated identically — a user's name has the same weight as a throwaway comment. How do you differentiate? |

## 04 — Context

| Doc | Problem |
|-----|---------|
| [Context Layer Assembly](context-layer-assembly.md) | System prompts mix static instructions, semi-stable persona, and dynamic data. How do you assemble them cleanly? |
| [Token Budget Management](token-budget.md) | A 128k context window doesn't mean you can use all 128k for the prompt. How do you budget and track token usage? |

## 05 — The Agent Loop

| Doc | Problem |
|-----|---------|
| [Agent Loop and Entropy Routing](agent-loop-and-entropy-routing.md) | An agent needs a loop that reasons, picks tools, observes results, and decides when to stop. How do you build ReAct? |
| [Streaming Agent Loop](streaming-agent-loop.md) | The agent loop waits for the entire response before yielding. How do you stream token-by-token with tool calls? |

## 06 — Human-in-the-Loop & Guardrails

| Doc | Problem |
|-----|---------|
| [Risk-Based Permissions and Hooks](risk-based-permissions-and-hooks.md) | Agents can execute dangerous operations. How do you gate actions by risk level and observe behavior? |
| [Draft-Before-Act Safety Pattern](draft-before-act.md) | Worker agents could fire off irreversible actions immediately. How do you stage actions for approval? |

## 07 — Workflows

| Doc | Problem |
|-----|---------|
| [Workflow DAG Orchestration](workflow-dag-orchestration.md) | Complex tasks require multi-step pipelines with dependencies. How do you define, validate, and execute workflows? |

## 08 — Self-Improvement (The CLAW)

| Doc | Problem |
|-----|---------|
| [Self-Repair and Graceful Degradation](self-repair-and-graceful-degradation.md) | Tools fail, APIs go down, context overflows. How does the agent recover and degrade gracefully? |
| [LLM-as-Judge Evaluation](llm-judge-evaluation.md) | You can't improve what you can't measure. How do you evaluate agent quality with deterministic and LLM-based scoring? |

## 09 — Multi-Agent

| Doc | Problem |
|-----|---------|
| [Multi-Agent Dispatcher/Worker](multi-agent-dispatcher-worker.md) | A single monolithic agent with all tools is a security risk. How do you split into specialized agents with scoped tools? |
