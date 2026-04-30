# Provider Resilience

## Problem

LLM API calls fail constantly. Rate limits hit during bursts of tool calls. Auth tokens expire. Network timeouts occur under load. Server errors (HTTP 500) happen unpredictably. Without retry logic and credential rotation, every transient failure crashes the agent. Naive retry (fixed delay, no backoff) makes rate limiting worse by hammering the API.

## Solution

Wrap every LLM call in a resilience layer: classify errors, retry only retryable ones with exponential backoff and jitter, rotate credentials on auth failure, and log every request for observability.

### Error Classification

`ProviderErrorCode` (`src/the_agents_playbook/providers/types.py:18`) defines 8 error categories: `AUTH_FAILED`, `RATE_LIMITED`, `SERVER_ERROR`, `CONTEXT_TOO_LONG`, `BAD_REQUEST`, `TIMEOUT`, `CONNECTION_FAILED`, `UNKNOWN`. Each `ProviderError` (`types.py:29`) carries a `retryable` flag — HTTP 429 and 5xx are retryable; 400 and 401 are not.

### Retry with Exponential Backoff + Jitter

`BaseProvider._with_retry()` (`src/the_agents_playbook/providers/base.py:178`) implements the retry loop:

```
delay = min(base_delay * 2^attempt, max_delay)
if jitter: delay *= random.uniform(0.5, 1.5)
```

`RetryConfig` (`types.py:52`) controls behavior: `max_retries=3`, `base_delay=1.0s`, `max_delay=30.0s`, `jitter=True`. Only `RATE_LIMITED`, `SERVER_ERROR`, `TIMEOUT`, and `CONNECTION_FAILED` codes are retried.

### Credential Rotation

`CredentialPool` (`types.py:98`) manages multiple API keys. On `AUTH_FAILED`, the provider automatically rotates to the next key via `itertools.cycle` and recreates the HTTP client with the new credential (`base.py:199-214`).

### Request Logging

`RequestLog` (`types.py:74`) captures request_id, provider, model, endpoint, status_code, error_code, duration_ms, token counts, cost, and retry_count. Fired via `on_request_log` callback (`base.py:162`).

### Connection Pooling

`PoolConfig` (`types.py:135`) controls HTTP connection reuse: `max_connections=10`, `max_keepalive_connections=5`, `keepalive_expiry=30s`. Configured via `httpx.Limits` in `BaseProvider._get_client()` (`base.py:306`).

## Code Reference

- `src/the_agents_playbook/providers/types.py` — `ProviderErrorCode`, `ProviderError`, `RetryConfig`, `CredentialPool`, `PoolConfig`, `RequestLog`
- `src/the_agents_playbook/providers/base.py` — `BaseProvider` with `_with_retry()`, `_check_status()`, `_get_client()`
- `src/the_agents_playbook/providers/openai.py` — `OpenAIProvider` concrete implementation
- `src/the_agents_playbook/providers/anthropic.py` — `AnthropicProvider` concrete implementation

## Playground Example

- `01-basic-calls/05-providers.py` — demonstrates provider switching between OpenAI and Anthropic

## LangGraph Example

- `langgraph-examples/08-the-claw/02_retry_loop.py` — retry loop pattern in a LangGraph agent graph
