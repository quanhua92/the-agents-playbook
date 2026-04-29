# Test Suite

## Structure

```
tests/
├── __init__.py
├── conftest.py                 # Shared pytest fixtures (mock API responses)
└── providers/
    ├── __init__.py
    ├── test_types.py           # Unit tests for all type models
    ├── test_base_provider.py   # Unit tests for BaseProvider (mocked HTTP)
    └── test_openai.py          # Integration tests for OpenAIProvider (real API)
```

## Running Tests

```bash
# Unit tests only (no API key needed, fastest)
uv run pytest tests/

# Integration tests only (requires OPENAI_API_KEY in .env)
uv run pytest tests/ -m openai

# All tests combined
uv run pytest tests/ -m ""

# Verbose with coverage report
uv run pytest tests/ -v --cov=src --cov-report=term-missing

# Run a single test file
uv run pytest tests/providers/test_types.py -v

# Run a specific test class or function
uv run pytest tests/providers/test_base_provider.py::TestRetry -v
uv run pytest tests/providers/test_types.py::TestCredentialPool::test_multiple_keys_round_robin -v
```

## Test Categories

### Unit Tests (`test_types.py` + `test_base_provider.py`)

70 tests that run entirely in-process with mocked HTTP. No API keys required.

**`test_types.py`** (40 tests) — Pure model validation:
- `ProviderErrorCode` — all enum values exist and are strings
- `ProviderError` — default fields, custom fields, is an Exception
- `RetryConfig` — defaults, custom values, validation (zero retries ok, negative rejects)
- `PoolConfig` — defaults and custom values
- `RequestLog` — default construction, error field population
- `CredentialPool` — single key, round-robin rotation, wrap-around, empty rejection, `__len__`
- `InputMessage` / `OutputMessage` — defaults and field overrides
- `MessageRequest` — defaults, timeout_seconds, validation (out-of-range rejects)
- `MessageResponse` — basic construction
- `ToolSpec` — `to_api_dict()` output shape
- `ToolChoice` — `to_api_dict()` for auto, required, function modes
- `ResponseFormat` — json_schema and json_object modes

**`test_base_provider.py`** (30 tests) — BaseProvider behavior with `respx` HTTP mocks:
- **Error classification** (`TestCheckStatus`) — 200/201 pass through, 401/403 → AUTH_FAILED (non-retryable), 429 → RATE_LIMITED (retryable), 400 → BAD_REQUEST, 500/502 → SERVER_ERROR (retryable), 418 → UNKNOWN, non-JSON bodies handled
- **Retry** (`TestRetry`) — 429 retries then succeeds, retries exhaust and raise, 401 raises immediately (no retry), 500 retries then succeeds, `max_retries=0` disables retry
- **Request logging** (`TestRequestLogging`) — callback fires on success with status/duration, callback fires on failure with error_code, request counter increments, log records model and endpoint
- **Credential rotation** (`TestCredentialRotation`) — auto-rotate on 401 with multi-key pool, no rotation with single-key pool, `_build_auth_from_pool()` default Bearer format, no-pool raises error, client headers include pool key
- **Pool limits** (`TestPoolLimits`) — default config stored, custom config stored, request succeeds with custom pool
- **Per-request timeout** (`TestPerRequestTimeout`) — default timeout uses client config, custom `timeout_seconds` applied, explicit `None` no override
- **Backward compat** (`TestBackwardCompat`) — init with no args, custom retry_config, close, close is idempotent

### Integration Tests (`test_openai.py`)

9 tests that hit the real LLM API via the `OPENAI_API_KEY` in `.env`. These are excluded from the default `uv run pytest` run to keep CI fast and avoid requiring secrets.

- `test_basic_chat_completion` — sends a message, gets a response with content
- `test_system_prompt` — verifies system prompt influences output
- `test_structured_output_json_schema` — returns valid JSON matching a Pydantic model
- `test_structured_output_json_object` — returns valid JSON with generic json_object mode
- `test_tool_call` — model calls the `get_weather` tool with `tool_choice=required`
- `test_tool_choice_auto` — model decides whether to call a tool or reply with text
- `test_bad_request_raises_provider_error` — invalid model name triggers `ProviderError` with BAD_REQUEST code
- `test_request_log_callback` — verifies `on_request_log` callback fires with correct fields
- `test_per_request_timeout` — request with `timeout_seconds=10.0` succeeds

## Tooling

| Tool | Purpose |
|------|---------|
| **pytest** | Test framework |
| **pytest-asyncio** | `asyncio_mode = "auto"` — async test functions run automatically |
| **pytest-cov** | Coverage reporting (`--cov=src`) |
| **respx** | HTTP mocking for `httpx` — intercepts requests without real network calls |

## Configuration

All config lives in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "--cov=src --cov-report=term-missing -m 'not openai'"
markers = [
    "openai: integration tests that hit the real API (requires OPENAI_API_KEY in .env)",
]
```

- Default run (`uv run pytest tests/`) excludes `openai`-marked tests
- Integration tests are opt-in: `uv run pytest tests/ -m openai`

## Writing New Tests

### Unit test (mocked HTTP)

```python
import httpx
import respx
from the_agents_playbook.providers.types import ProviderError, ProviderErrorCode

@respx.mock
async def test_my_feature(mock_chat_response):
    respx.post("https://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=mock_chat_response),
    )
    result = await provider.send_message(request)
    assert result.message.content is not None
```

### Integration test (real API)

```python
import pytest

@pytest.mark.openai
@pytest.mark.asyncio
async def test_my_feature(provider):
    response = await provider.send_message(request)
    assert response.stop_reason == "stop"
```

### Key patterns

- Use `StubProvider` (defined in `test_base_provider.py`) to test BaseProvider logic without real HTTP
- Use the `mock_chat_response` fixture from `conftest.py` for a standard API response shape
- Mark integration tests with `@pytest.mark.openai` so they're excluded by default
- Async test functions work automatically — no need for `@pytest.mark.asyncio` when `asyncio_mode = "auto"`
