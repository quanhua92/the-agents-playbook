# The Agents Playbook - Architectural Blueprint

## Pillar 1: The Provider
Abstracts the LLM API (OpenAI, Anthropic, Gemini).

- **BaseProvider (ABC):**
    - `send_message(request: MessageRequest) -> MessageResponse`
    - `stream(request: MessageRequest) -> AsyncGenerator[ResponseChunk, None]`

- **CredentialManager:**
    - Handles API key resolution from environment variables, vault backends, or runtime overrides.
    - `get_credentials(provider_name: str) -> CredentialConfig`
    - `refresh(provider_name: str) -> CredentialConfig`
    - Supports key rotation: consumers receive a fresh reference on each call, not a cached value.

- **OpenAIProvider / AnthropicProvider:**
    - Handles concrete JSON formatting, tool-schema translation, and SSE parsing.
    - Each provider receives a `CredentialManager` reference at construction.

- **Error Handling (Provider-level):**
    - `ProviderError` hierarchy: `RateLimitError`, `AuthenticationError`, `ContextWindowExceededError`, `ServerError`.
    - Built-in retry with exponential backoff for `RateLimitError` and transient `ServerError` (configurable `max_retries`, `base_delay`).
    - `AuthenticationError` is never retried — raised immediately.

- **Message Types:**
    - `MessageRequest`: { system: str, messages: list[InputMessage], tools: list[ToolSpec], model: str, temperature: float }
    - `MessageResponse`: { message: OutputMessage, usage: Usage, stop_reason: StopReason }
    - `ResponseChunk`: { type: Literal["text", "tool_call", "usage", "stop"], data: str | ToolCall | Usage }

## Pillar 2: The Tools
Separates the "Decision to Act" from the "Action."

- **BaseToolExecutor (ABC):**
    - `execute(tool_name: str, arguments: dict[str, Any]) -> ToolResult`
    - *Example Implementations:* `ShellExecutor`, `PythonExecutor`, `RPCExecutor`.

- **ToolResult:**
    - `output: str`: The text result returned to the LLM.
    - `error: bool | None`: Whether the tool execution failed.
    - `metadata: dict[str, Any]`: Optional structured data (exit codes, timing, etc.).

- **ToolRegistry:**
    - `register(name: str, description: str, schema: dict, executor: BaseToolExecutor)`
    - `get_specs() -> list[ToolSpec]`
    - `dispatch(tool_name: str, arguments: dict[str, Any]) -> ToolResult`
    - Raises `ToolNotFoundError` for unknown tools, propagates executor exceptions as `ToolExecutionError`.

## Pillar 3: The Memory
The CRUD layer for long-term facts and metadata.

- **BaseMemoryProvider (ABC):**
    - `set(key: str, value: Any) -> None`: Learn a fact.
    - `get(key: str) -> Any`: Recall a specific fact by exact key.
    - `delete(key: str) -> None`: Forget/Privacy.
    - `search(query: str, limit: int = 10) -> list[Fact]`: Semantic similarity lookup.

- **Fact:**
    - `key: str`
    - `value: Any`
    - `embedding: list[float] | None`: Pre-computed vector for the value content.
    - `created_at: datetime`
    - `updated_at: datetime`

- **EmbeddingProvider (ABC):**
    - `embed(text: str) -> list[float]`: Produce a vector for a single text.
    - `embed_batch(texts: list[str]) -> list[list[float]]`: Batched embedding for efficiency.
    - Used by `VectorMemoryProvider` at write time (`set`) and query time (`search`) to compute and compare embeddings.

- **InMemoryMemoryProvider:**
    - Dict-backed implementation for development and testing. Uses brute-force cosine similarity for `search`.

- **VectorMemoryProvider:**
    - Production implementation backed by a vector store (e.g., Qdrant, ChromaDB, pgvector).
    - Requires an `EmbeddingProvider` at construction.

## Pillar 4: The Session
Manages the history, token counts, and context engineering.

- **Session:**
    - `id: str`
    - `messages: list[SessionMessage]`
    - `summary: str | None`: The running summary of compacted messages.
    - `usage_tracker: Usage`
    - `persistent_path: Path`
    - `compaction_config: CompactionConfig`
    - **Methods:**
        - `add_message(role: str, content: Any)`
        - `get_context_for_llm(limit: int | None = None, include_summary: bool = True) -> list[InputMessage]`
        - `compact()`: Summarize oldest messages when total tokens exceed `compaction_config.token_threshold`.
        - `persist()`: Atomic write to JSONL.

- **CompactionConfig:**
    - `token_threshold: int`: Trigger compaction when total tokens exceed this (e.g., 100_000).
    - `retain_recent: int`: Number of most recent messages to keep intact (e.g., 20).
    - `summarizer_model: str`: Model ID used for the summarization LLM call.
    - The summarizer receives the oldest messages (excluding the `retain_recent` window) and produces a concise summary. The summary is stored in `session.summary` and prepended as a single system-level message on subsequent `get_context_for_llm` calls.

## Pillar 5: The Agent
The "Autonomous Loop" that ties it all together.

- **AgentConfig:**
    - `max_tool_iterations: int = 25`: Safety limit on ReAct loop cycles.
    - `on_error: Literal["raise", "yield_and_continue", "abort"]`: Controls error propagation in the loop.

- **Agent(provider, tools, memory, session, middleware_chain, config):**
    - `run(prompt: str) -> AsyncGenerator[AgentEvent, None]`
    - **The Internal ReAct Loop:**
        1. `session.add_message("user", prompt)`
        2. Loop (up to `max_tool_iterations`):
            - `context = session.get_context_for_llm()`
            - `async for event in provider.stream(context)`:
                - If `Text`: `yield event` + `session.buffer_text()`
                - If `ToolCall`:
                    - `yield status("Running tool...")`
                    - `result = tools.dispatch(call)`
                    - `session.add_tool_result(result)`
                    - `yield ToolResult(result)`
                    - `continue loop`
            - Break if no ToolCalls in last turn.
            - If a `ProviderError` occurs:
                - `yield ErrorEvent(error)`
                - If `on_error == "raise"`: raise.
                - If `on_error == "abort"`: break.
                - If `on_error == "yield_and_continue"`: yield and break.

## Pillar 6: The Workflow
Orchestrates multiple agents into a deterministic sequence (DAG).

- **BaseStep (ABC):**
    - `id: str`: Unique node identifier.
    - `agent: Agent`: The specialized agent for this node.
    - `dependencies: list[str]`: IDs of steps that must complete before this one runs.
    - `run(input: Any, shared_state: WorkflowState) -> StepResult`

- **PlanStep (inherits BaseStep):**
    - **Role:** Architectural design and strategy.
    - **Constraint:** Limited to "Read-Only" tools + Design tools.
    - **Output:** A structured `Plan` saved to `shared_state`.

- **BuildStep (inherits BaseStep):**
    - **Role:** Implementation and Execution.
    - **Constraint:** Has access to "Write" and "Bash" tools.
    - **Input:** Consumes the `Plan` from `shared_state` to guide its actions.

- **StepResult:**
    - `step_id: str`
    - `success: bool`
    - `output_data: Any`: The primary value created (e.g. a JSON Plan or Code).
    - `summary: str`: Prose summary of actions taken for the next agent's context.
    - `updates: dict[str, Any]`: Values to be merged into `WorkflowState`.
    - `error: Exception | None`: Set when `success` is `False`.

- **Error Handling (Workflow-level):**
    - `on_step_failure: Literal["abort", "skip", "retry"]`: Workflow-level policy.
    - `retry_config: RetryConfig | None`: When `on_step_failure == "retry"`, uses `max_retries` and `backoff` from config.
    - Failed steps (after all retries) are recorded in `WorkflowState.history` with `success=False`. Downstream steps whose `dependencies` include a failed step are skipped automatically.

- **Concurrency:**
    - Steps with no dependency edges between them are scheduled concurrently via `asyncio.gather`.
    - Each step receives an isolated snapshot of `shared_context` at dispatch time. Merge conflicts (two parallel steps writing the same key) are resolved by the `conflict_policy`: `Literal["last_write_wins", "error"]`.

- **Workflow:**
    - `steps: list[BaseStep]`
    - `state: WorkflowState`
    - **Methods:**
        - `run(initial_input: Any) -> AsyncGenerator[WorkflowEvent, None]`
        - `validate() -> list[str]`: Returns a list of validation errors (cycles, missing dependencies, unconnected steps).

- **WorkflowState:**
    - `history: list[StepResult]`: Log of every step executed.
    - `shared_context: dict[str, Any]`: The "Shared Clipboard" for handoffs.
    - `global_memory: BaseMemoryProvider`: Persistent memory across the entire workflow.

## Pillar 7: The Middleware
Pluggable logic that wraps Agent operations (Model Calls, Tool Calls, Memory).

- **BaseMiddleware (ABC):**
    - `handle(context: MiddlewareContext, next_fn: Callable) -> Any`

- **MiddlewareChain:**
    - Constructs middleware as a nested chain (similar to Express.js / ASGI middleware).
    - Order matters: the first middleware in the list is the outermost wrapper. It runs *before* all others on the way in, and *after* all others on the way out.
    - Any middleware may short-circuit by returning a value without calling `next_fn`. This is how `GuardrailMiddleware` blocks dangerous operations and `HumanApprovalMiddleware` holds for user confirmation.
    - `build(middlewares: list[BaseMiddleware], final_handler: Callable) -> Callable`

- **MiddlewareContext:**
    - `operation: Literal["model_call", "tool_call", "memory_access"]`
    - `agent_id: str`
    - `payload: Any`
    - `metadata: dict[str, Any]`: Mutable dict that middleware can read/write to pass data downstream (e.g., a `RedactionMiddleware` can annotate which fields were scrubbed).

- **Common Middleware "Plays":**
    - `LoggingMiddleware`: Telemetry and debugging. Always outermost in production.
    - `GuardrailMiddleware`: Input/Output safety checks. May short-circuit to block operations.
    - `RedactionMiddleware`: Scrubbing PII before sending to API.
    - `HumanApprovalMiddleware`: Injecting HITL into any tool call. Short-circuits until user approves.

## Pillar 8: Configuration & Validation
Schema-based configuration with startup-time validation to fail fast on misconfiguration.

- **ProviderConfig:**
    - `provider_name: str`
    - `model: str`
    - `temperature: float = 0.7`
    - `max_tokens: int = 4096`
    - `retry_config: RetryConfig`

- **RetryConfig:**
    - `max_retries: int = 3`
    - `base_delay: float = 1.0`: Seconds.
    - `max_delay: float = 30.0`: Cap on exponential backoff.
    - `retryable_errors: list[str]`: Error types eligible for retry (default: rate_limit, server_error).

- **WorkflowConfig:**
    - `on_step_failure: Literal["abort", "skip", "retry"] = "abort"`
    - `retry_config: RetryConfig | None = None`
    - `conflict_policy: Literal["last_write_wins", "error"] = "last_write_wins"`

- **AgentConfig:**
    - `max_tool_iterations: int = 25`
    - `on_error: Literal["raise", "yield_and_continue", "abort"] = "abort"`

- **validate_config(config: Any, schema: type) -> list[str]:**
    - Runs Pydantic validation at startup.
    - Returns a list of human-readable error strings.
    - Application refuses to start if validation fails.

## Event Schema
- **AgentEvent:** { event_type: Literal["text", "tool_call", "tool_result", "status", "error"], data: EventData }
- **WorkflowEvent:** { event_type: Literal["step_started", "step_completed", "step_failed", "workflow_completed", "workflow_failed"], data: WorkflowEventData }
- **EventData:**
    - `text: str | None`
    - `tool_call: ToolCall | None`
    - `tool_result: str | None`
    - `status: str | None` (e.g. "Thinking...", "Searching...")
    - `error: str | None`
