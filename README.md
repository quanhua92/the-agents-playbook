# The Agents Playbook

Build AI Agents from scratch, one concept at a time.

## Project Layout

```
the-agents-playbook/
├── 01-basic-calls/          # Playground examples (run directly)
├── 02-tools/                # Tool protocol, registry, dispatcher, builtins, cache, MCP
├── 03-memory/               # Dual memory, vector search, consolidation, session
├── 04-context/              # Context layers, templates, metadata injection, builder
├── 05-the-loop/             # Agent loop, entropy scoring, tool chaining
├── 06-human-in-the-loop/    # Permissions, prompters, hooks, ask-user
├── 07-workflows/            # Steps, DAG runner, state, concurrent execution
├── 08-the-claw/             # Self-repair, degradation, evaluation, self-review
├── src/                     # SDK package (the_agents_playbook)
│   └── the_agents_playbook/
│       ├── providers/       # LLM provider abstraction (BaseProvider, OpenAIProvider)
│       ├── tools/           # Tool protocol, registry, dispatcher, builtins, cache, MCP
│       ├── memory/          # Dual file, vector store, consolidation, session
│       ├── context/         # Context builder, layers, templates, metadata
│       ├── loop/            # Agent, ReAct loop, scoring, chains
│       ├── guardrails/      # Permissions, prompters, hooks, ask-user
│       ├── workflows/       # Steps, workflow DAG, state, hooks
│       ├── claw/            # Self-repair, degradation, evaluation, self-review
│       ├── utils/           # Shared utilities (schema, vectors)
│       └── settings.py      # Env-based configuration
├── tests/                   # SDK test suite (unit + integration)
└── ROADMAP.md               # Architecture blueprint
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

Run any example:

```bash
uv run python 01-basic-calls/02-basic-chat.py
```

## SDK (`src/`)

The `providers` package is a typed abstraction over LLM APIs with retry, error handling, logging, and key rotation built in:

- **`BaseProvider`** -- Abstract base with typed error hierarchy, exponential backoff retry, request logging, API key rotation, connection pool limits, and per-request timeout overrides
- **`OpenAIProvider`** -- Concrete implementation for OpenAI-compatible APIs (OpenRouter, etc.)
- **`AnthropicProvider`** -- Concrete implementation for Anthropic Messages API
- **Types** -- `MessageRequest`, `MessageResponse`, `ToolSpec`, `ToolChoice`, `ResponseFormat`, `ProviderError`, `RetryConfig`, `CredentialPool`, `PoolConfig`, `RequestLog`

```python
from the_agents_playbook import settings
from the_agents_playbook.providers import OpenAIProvider, MessageRequest, InputMessage

provider = OpenAIProvider()
response = await provider.send_message(
    MessageRequest(
        model=settings.openai_model,
        messages=[InputMessage(role="user", content="Hello")],
    )
)
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
OPENAI_API_KEY=sk-or-v1-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-oss-20b

ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_BASE_URL=https://api.anthropic.com/v1
ANTHROPIC_MODEL=claude-sonnet-4-6

EMBEDDING_API_KEY=sk-...
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small

MOCK_ONLY=false
```

## License

MIT
