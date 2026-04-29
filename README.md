# The Agents Playbook

Build AI Agents from scratch, one concept at a time.

## Project Layout

```
the-agents-playbook/
├── 01-basic-calls/          # Playground examples (run directly)
├── 02-tools/                # Tool protocol, registry, dispatcher, builtins, cache, MCP
├── 03-memory/               # Dual memory, vector search, consolidation, session, compaction
├── 04-context/              # Context layers, templates, metadata injection, builder
├── 05-the-loop/             # Agent loop, entropy scoring, tool chaining
├── 06-human-in-the-loop/    # Permissions, prompters, hooks, ask-user
├── 07-workflows/            # Steps, DAG runner, state, concurrent execution
├── 08-the-claw/             # Self-repair, degradation, evaluation, self-review
├── src/                     # SDK package (the_agents_playbook)
│   └── the_agents_playbook/
│       ├── providers/       # LLM providers (BaseProvider, OpenAI, Anthropic, streaming)
│       ├── tools/           # Tool protocol, registry, dispatcher, builtins, cache, MCP
│       ├── memory/          # Dual file, vector store, embeddings, consolidation, session
│       ├── context/         # Context builder, layers, templates, metadata
│       ├── loop/            # Agent, ReAct loop, scoring, chains
│       ├── guardrails/      # Permissions, prompters, hooks, ask-user
│       ├── workflows/       # Steps, workflow DAG, state, hooks
│       ├── claw/            # Self-repair, degradation, evaluation, self-review
│       ├── utils/           # Shared utilities (schema, vectors)
│       └── settings.py      # Env-based configuration
├── tests/                   # SDK test suite (505 unit + integration tests)
└── TODO-FINAL.md            # Completion tracker
```

## Setup

```bash
uv sync
cp .env.example .env   # fill in your API key
```

## Playground Examples

The `01-basic-calls/` directory contains self-contained scripts you can run to learn each concept. Each script defines its types inline -- they do not import from the SDK. They are designed to be read top-to-bottom to understand the mechanics before using the SDK.

| File | Concept |
|------|---------|
| `01-async-httpx.py` | Async HTTP with httpx (no LLM) |
| `02-basic-chat.py` | Raw chat completion against the OpenAI-compatible API |
| `03-structured-output.py` | `response_format: json_schema` with Pydantic model validation and `$ref` flattening |
| `04-tool-choice.py` | Forced tool calls via `tool_choice` as an alternative to structured output |
| `05-providers.py` | Building a provider abstraction from scratch: ABC, typed request/response, tool specs, and structured output wired together |
| `06-streaming.py` | SSE streaming with `provider.stream()` — token-by-token output in real time |
| `07-embeddings.py` | Embedding text with `OpenAIEmbeddingProvider` and computing cosine similarity |

Run any example:

```bash
uv run python 01-basic-calls/06-streaming.py
```

The `03-memory/` directory includes examples that use the SDK memory system:

| File | Concept |
|------|---------|
| `01-fact-storage.py` | Storing and retrieving facts |
| `02-dual-file-memory.py` | Long-term + short-term dual memory |
| `03-vector-search.py` | Cosine similarity search with mock embeddings |
| `04-consolidation.py` | LLM-assisted fact consolidation |
| `05-session-persistence.py` | JSONL session save/load |
| `06-real-embeddings.py` | Vector search with real OpenAI embeddings via OpenRouter |

## SDK (`src/`)

### Providers

Typed abstraction over LLM APIs with retry, error handling, logging, key rotation, and streaming:

- **`BaseProvider`** -- Abstract base with typed error hierarchy, exponential backoff retry, request logging, API key rotation, connection pool limits, per-request timeout, and SSE streaming
- **`OpenAIProvider`** -- OpenAI-compatible APIs (OpenRouter, OpenAI, etc.)
- **`AnthropicProvider`** -- Anthropic Messages API
- **Types** -- `MessageRequest`, `MessageResponse`, `ResponseChunk`, `StreamUsage`, `ToolSpec`, `ToolChoice`, `ResponseFormat`, `ProviderError`, `RetryConfig`, `CredentialPool`, `PoolConfig`, `RequestLog`

```python
from the_agents_playbook import settings
from the_agents_playbook.providers import OpenAIProvider, MessageRequest, InputMessage

# Sync request
provider = OpenAIProvider()
response = await provider.send_message(
    MessageRequest(
        model=settings.openai_model,
        messages=[InputMessage(role="user", content="Hello")],
    )
)
```

```python
# Streaming
async for chunk in provider.stream(request):
    if chunk.delta_text:
        print(chunk.delta_text, end="", flush=True)
```

### Memory

- **`InMemoryVectorStore`** -- Semantic fact retrieval with cosine similarity + time decay
- **`OpenAIEmbeddingProvider`** -- Concrete embedding provider calling the OpenAI embeddings API via OpenRouter
- **`EmbeddingProvider`** -- ABC for plugging in any embedding backend
- **`SessionPersistence`** -- JSONL conversation save/load
- **`SessionCompactor`** -- Token-aware compaction that summarizes old messages when context exceeds a threshold
- **`DualFileMemory`** -- Long-term + short-term dual-file memory
- **`LLMConsolidator`** -- LLM-assisted fact consolidation from raw history

```python
from the_agents_playbook.memory import OpenAIEmbeddingProvider, InMemoryVectorStore, Fact

embedder = OpenAIEmbeddingProvider()
store = InMemoryVectorStore(embedding_provider=embedder)
await store.store(Fact(content="User prefers Python", source="user"))
facts = await store.recall("programming languages")
```

### Config Validation

```python
from the_agents_playbook import validate_config

warnings = validate_config()
# Returns list of warnings (empty = all good)
# Checks: API key prefixes, key/URL mismatches, embedding config
```

## Tests

```bash
# Unit tests (no API key needed)
uv run pytest tests/

# Integration tests (requires API key in .env)
uv run pytest tests/ -m openai

# All tests
uv run pytest tests/ -m ""

# Verbose with coverage
uv run pytest tests/ -v --cov=src --cov-report=term-missing
```

See `tests/README.md` for full documentation of the test suite.

## Configuration

All settings are loaded from `.env` via `pydantic-settings`:

```
# OpenAI / OpenRouter
OPENAI_API_KEY=sk-or-v1-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-oss-20b

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_BASE_URL=https://api.anthropic.com/v1
ANTHROPIC_MODEL=claude-sonnet-4-6

# Embeddings via OpenRouter
EMBEDDING_API_KEY=sk-or-v1-...
EMBEDDING_BASE_URL=https://openrouter.ai/api/v1
EMBEDDING_MODEL=openai/text-embedding-3-small

MOCK_ONLY=false
```

## License

MIT
