"""Tests for BaseProvider: error classification, retry, logging, key rotation, pool limits."""

from typing import Any

import httpx
import pytest
import respx

from the_agents_playbook.providers.base import BaseProvider
from the_agents_playbook.providers.types import (
    CredentialPool,
    InputMessage,
    MessageRequest,
    MessageResponse,
    OutputMessage,
    PoolConfig,
    ProviderErrorCode,
    ProviderError,
    RequestLog,
    ResponseChunk,
    RetryConfig,
)


# ---------------------------------------------------------------------------
# Concrete stub provider for testing
# ---------------------------------------------------------------------------


class StubProvider(BaseProvider):
    """Minimal concrete provider for testing BaseProvider logic."""

    def __init__(self, base_url: str = "https://api.example.com/v1", **kwargs):
        self._base_url = base_url
        super().__init__(**kwargs)

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
        }

    def _chat_endpoint(self) -> str:
        return self._base_url + "/chat/completions"

    def _build_body(self, request: MessageRequest) -> dict[str, Any]:
        messages = [{"role": "system", "content": request.system}]
        messages.extend(m.model_dump() for m in request.messages)
        return {"model": request.model, "messages": messages}

    def _parse_response(self, response: httpx.Response) -> MessageResponse:
        raw = response.json()
        choice = raw["choices"][0]
        message = choice["message"]
        return MessageResponse(
            message=OutputMessage(
                role=message["role"],
                content=message.get("content"),
            ),
            stop_reason=choice.get("finish_reason", "unknown"),
        )

    def _build_stream_body(self, request: MessageRequest) -> dict[str, Any]:
        body = self._build_body(request)
        body["stream"] = True
        return body

    def _parse_stream_chunk(self, data: str) -> ResponseChunk | None:
        import json as _json

        try:
            raw = _json.loads(data)
        except _json.JSONDecodeError:
            return None
        choice = raw.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        chunk = ResponseChunk()
        if "content" in delta and delta["content"]:
            chunk.delta_text = delta["content"]
        return chunk


def make_request(model: str = "test-model", **kwargs) -> MessageRequest:
    return MessageRequest(
        model=model,
        messages=[InputMessage(role="user", content="Hello")],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Section 1: Error classification (_check_status)
# ---------------------------------------------------------------------------


class TestCheckStatus:
    def test_200_ok(self):
        provider = StubProvider()
        resp = httpx.Response(200, json={"ok": True})
        provider._check_status(resp)  # should not raise

    def test_201_ok(self):
        provider = StubProvider()
        resp = httpx.Response(201, json={"ok": True})
        provider._check_status(resp)  # should not raise

    def test_401_auth_failed(self):
        provider = StubProvider()
        resp = httpx.Response(401, json={"error": "invalid key"})
        with pytest.raises(ProviderError) as exc_info:
            provider._check_status(resp)
        err = exc_info.value
        assert err.code == ProviderErrorCode.AUTH_FAILED
        assert err.status_code == 401
        assert err.retryable is False
        assert err.raw_body == {"error": "invalid key"}

    def test_403_auth_failed(self):
        provider = StubProvider()
        resp = httpx.Response(403, json={"error": "forbidden"})
        with pytest.raises(ProviderError) as exc_info:
            provider._check_status(resp)
        assert exc_info.value.code == ProviderErrorCode.AUTH_FAILED

    def test_429_rate_limited(self):
        provider = StubProvider()
        resp = httpx.Response(429, json={"error": "rate limit"})
        with pytest.raises(ProviderError) as exc_info:
            provider._check_status(resp)
        err = exc_info.value
        assert err.code == ProviderErrorCode.RATE_LIMITED
        assert err.status_code == 429
        assert err.retryable is True

    def test_400_bad_request(self):
        provider = StubProvider()
        resp = httpx.Response(400, json={"error": "bad input"})
        with pytest.raises(ProviderError) as exc_info:
            provider._check_status(resp)
        err = exc_info.value
        assert err.code == ProviderErrorCode.BAD_REQUEST
        assert err.retryable is False

    def test_500_server_error(self):
        provider = StubProvider()
        resp = httpx.Response(500, json={"error": "internal"})
        with pytest.raises(ProviderError) as exc_info:
            provider._check_status(resp)
        err = exc_info.value
        assert err.code == ProviderErrorCode.SERVER_ERROR
        assert err.status_code == 500
        assert err.retryable is True

    def test_502_server_error(self):
        provider = StubProvider()
        resp = httpx.Response(502, json={"error": "bad gateway"})
        with pytest.raises(ProviderError) as exc_info:
            provider._check_status(resp)
        assert exc_info.value.code == ProviderErrorCode.SERVER_ERROR

    def test_418_unknown(self):
        provider = StubProvider()
        resp = httpx.Response(418, json={"error": "teapot"})
        with pytest.raises(ProviderError) as exc_info:
            provider._check_status(resp)
        err = exc_info.value
        assert err.code == ProviderErrorCode.UNKNOWN
        assert err.status_code == 418
        assert err.retryable is False

    def test_non_json_body(self):
        provider = StubProvider()
        resp = httpx.Response(500, content=b"Internal Server Error")
        with pytest.raises(ProviderError) as exc_info:
            provider._check_status(resp)
        assert exc_info.value.raw_body is None


# ---------------------------------------------------------------------------
# Section 2: Retry with backoff
# ---------------------------------------------------------------------------


class TestRetry:
    @respx.mock
    async def test_429_retries_then_succeeds(self, mock_chat_response):
        respx.post("https://api.example.com/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(429, json={"error": "rate limit"}),
                httpx.Response(429, json={"error": "rate limit"}),
                httpx.Response(200, json=mock_chat_response),
            ]
        )
        provider = StubProvider(
            retry_config=RetryConfig(
                max_retries=3, base_delay=0.01, max_delay=0.05, jitter=False
            ),
        )
        result = await provider.send_message(make_request())
        assert result.message.content == "Hello! How can I help you today?"

    @respx.mock
    async def test_429_exhausts_retries(self, mock_error_response):
        respx.post("https://api.example.com/v1/chat/completions").mock(
            return_value=httpx.Response(429, json=mock_error_response)
        )
        provider = StubProvider(
            retry_config=RetryConfig(
                max_retries=2, base_delay=0.01, max_delay=0.05, jitter=False
            ),
        )
        with pytest.raises(ProviderError) as exc_info:
            await provider.send_message(make_request())
        assert exc_info.value.code == ProviderErrorCode.RATE_LIMITED

    @respx.mock
    async def test_401_raises_immediately_no_retry(self, mock_error_response):
        respx.post("https://api.example.com/v1/chat/completions").mock(
            return_value=httpx.Response(401, json=mock_error_response)
        )
        provider = StubProvider(
            retry_config=RetryConfig(max_retries=3, base_delay=0.01)
        )
        with pytest.raises(ProviderError) as exc_info:
            await provider.send_message(make_request())
        assert exc_info.value.code == ProviderErrorCode.AUTH_FAILED

    @respx.mock
    async def test_500_retries_then_succeeds(self, mock_chat_response):
        respx.post("https://api.example.com/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(500, json={"error": "internal"}),
                httpx.Response(200, json=mock_chat_response),
            ]
        )
        provider = StubProvider(
            retry_config=RetryConfig(
                max_retries=3, base_delay=0.01, max_delay=0.05, jitter=False
            ),
        )
        result = await provider.send_message(make_request())
        assert result.message.content is not None

    async def test_no_retry_when_max_retries_zero(self):
        provider = StubProvider(retry_config=RetryConfig(max_retries=0))
        call_count = 0

        async def failing_fn():
            nonlocal call_count
            call_count += 1
            raise ProviderError(
                "429",
                code=ProviderErrorCode.RATE_LIMITED,
                retryable=True,
                status_code=429,
            )

        with pytest.raises(ProviderError):
            await provider._with_retry(failing_fn)
        assert call_count == 1


# ---------------------------------------------------------------------------
# Section 3: Request logging
# ---------------------------------------------------------------------------


class TestRequestLogging:
    @respx.mock
    async def test_on_request_log_called_on_success(self, mock_chat_response):
        collected: list[RequestLog] = []
        respx.post("https://api.example.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_chat_response)
        )
        provider = StubProvider(
            provider_name="test-provider",
            on_request_log=collected.append,
            retry_config=RetryConfig(max_retries=0),
        )
        await provider.send_message(make_request())
        assert len(collected) == 1
        log = collected[0]
        assert log.request_id == "test-provider-0"
        assert log.provider == "test-provider"
        assert log.status_code == 200
        assert log.duration_ms is not None
        assert log.duration_ms > 0
        assert log.error_code is None

    @respx.mock
    async def test_on_request_log_called_on_failure(self, mock_error_response):
        collected: list[RequestLog] = []
        respx.post("https://api.example.com/v1/chat/completions").mock(
            return_value=httpx.Response(429, json=mock_error_response)
        )
        provider = StubProvider(
            provider_name="test-provider",
            on_request_log=collected.append,
            retry_config=RetryConfig(max_retries=0),
        )
        with pytest.raises(ProviderError):
            await provider.send_message(make_request())
        assert len(collected) == 1
        log = collected[0]
        assert log.error_code == ProviderErrorCode.RATE_LIMITED
        assert log.status_code == 429
        assert log.error_message is not None

    @respx.mock
    async def test_request_counter_increments(self, mock_chat_response):
        collected: list[RequestLog] = []
        respx.post("https://api.example.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_chat_response)
        )
        provider = StubProvider(
            provider_name="test",
            on_request_log=collected.append,
            retry_config=RetryConfig(max_retries=0),
        )
        await provider.send_message(make_request())
        await provider.send_message(make_request())
        assert collected[0].request_id == "test-0"
        assert collected[1].request_id == "test-1"

    @respx.mock
    async def test_log_records_model_and_endpoint(self, mock_chat_response):
        collected: list[RequestLog] = []
        respx.post("https://api.example.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_chat_response)
        )
        provider = StubProvider(
            on_request_log=collected.append,
            retry_config=RetryConfig(max_retries=0),
        )
        await provider.send_message(make_request(model="my-model"))
        log = collected[0]
        assert log.model == "my-model"
        assert log.endpoint == "https://api.example.com/v1/chat/completions"


# ---------------------------------------------------------------------------
# Section 4: Credential rotation
# ---------------------------------------------------------------------------


class TestCredentialRotation:
    @respx.mock
    async def test_auto_rotate_on_401_with_multi_key_pool(self, mock_chat_response):
        pool = CredentialPool(keys=["key-1", "key-2"])

        route = respx.post("https://api.example.com/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(401, json={"error": "invalid"}),
                httpx.Response(200, json=mock_chat_response),
            ]
        )
        route.side_effect = [
            httpx.Response(401, json={"error": "invalid"}),
            httpx.Response(200, json=mock_chat_response),
        ]

        provider = StubProvider(
            credential_pool=pool,
            retry_config=RetryConfig(
                max_retries=3, base_delay=0.01, max_delay=0.05, jitter=False
            ),
        )
        result = await provider.send_message(make_request())
        assert result.message.content is not None
        assert pool.current == "key-2"  # rotated after first 401

    @respx.mock
    async def test_no_rotation_with_single_key_pool(self, mock_error_response):
        pool = CredentialPool(keys=["only-key"])
        respx.post("https://api.example.com/v1/chat/completions").mock(
            return_value=httpx.Response(401, json=mock_error_response)
        )
        provider = StubProvider(
            credential_pool=pool,
            retry_config=RetryConfig(
                max_retries=3, base_delay=0.01, max_delay=0.05, jitter=False
            ),
        )
        with pytest.raises(ProviderError) as exc_info:
            await provider.send_message(make_request())
        # Should raise immediately — single key pool skips rotation
        assert exc_info.value.code == ProviderErrorCode.AUTH_FAILED

    def test_build_auth_from_pool_default(self):
        pool = CredentialPool(keys=["my-secret-key"])
        provider = StubProvider(credential_pool=pool)
        header, value = provider._build_auth_from_pool()
        assert header == "Authorization"
        assert value == "Bearer my-secret-key"

    def test_build_auth_from_pool_no_pool_raises(self):
        provider = StubProvider()
        with pytest.raises(ProviderError, match="No credential pool"):
            provider._build_auth_from_pool()

    @respx.mock
    async def test_client_headers_include_pool_key(self, mock_chat_response):
        pool = CredentialPool(keys=["pool-key-abc"])
        respx.post("https://api.example.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_chat_response),
        )
        provider = StubProvider(
            credential_pool=pool,
            retry_config=RetryConfig(max_retries=0),
        )
        await provider.send_message(make_request())
        # The last request should have used the pool key
        last_request = respx.calls[-1].request
        assert last_request.headers["authorization"] == "Bearer pool-key-abc"


# ---------------------------------------------------------------------------
# Section 5: Connection pool limits
# ---------------------------------------------------------------------------


class TestPoolLimits:
    def test_default_pool_config_stored(self):
        provider = StubProvider()
        assert provider._pool_config is None
        # _get_client falls back to default PoolConfig
        expected = PoolConfig()
        assert expected.max_connections == 10
        assert expected.max_keepalive_connections == 5
        assert expected.keepalive_expiry == 30.0

    def test_custom_pool_config_stored(self):
        cfg = PoolConfig(
            max_connections=20, max_keepalive_connections=10, keepalive_expiry=60.0
        )
        provider = StubProvider(pool_config=cfg)
        assert provider._pool_config.max_connections == 20
        assert provider._pool_config.max_keepalive_connections == 10
        assert provider._pool_config.keepalive_expiry == 60.0

    @respx.mock
    async def test_request_succeeds_with_custom_pool(self, mock_chat_response):
        respx.post("https://api.example.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_chat_response),
        )
        provider = StubProvider(
            pool_config=PoolConfig(
                max_connections=20, max_keepalive_connections=10, keepalive_expiry=60.0
            ),
            retry_config=RetryConfig(max_retries=0),
        )
        result = await provider.send_message(make_request())
        assert result.message.content is not None


# ---------------------------------------------------------------------------
# Section 6: Per-request timeout
# ---------------------------------------------------------------------------


class TestPerRequestTimeout:
    @respx.mock
    async def test_default_timeout_uses_client_config(self, mock_chat_response):
        route = respx.post("https://api.example.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_chat_response),
        )
        provider = StubProvider(retry_config=RetryConfig(max_retries=0))
        await provider.send_message(make_request())
        # When no per-request timeout, the request should not have an explicit timeout
        # (uses client default). respx doesn't expose the timeout, but we can verify
        # the request was made successfully.
        assert route.called

    @respx.mock
    async def test_custom_timeout_seconds(self, mock_chat_response):
        respx.post("https://api.example.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_chat_response),
        )
        provider = StubProvider(retry_config=RetryConfig(max_retries=0))
        # This should apply a 5s read timeout override
        result = await provider.send_message(make_request(timeout_seconds=5.0))
        assert result.message.content is not None

    @respx.mock
    async def test_none_timeout_no_override(self, mock_chat_response):
        respx.post("https://api.example.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_chat_response),
        )
        provider = StubProvider(retry_config=RetryConfig(max_retries=0))
        # Explicitly None — should not override
        result = await provider.send_message(make_request(timeout_seconds=None))
        assert result.message.content is not None


# ---------------------------------------------------------------------------
# BaseProvider backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_init_with_no_args(self):
        provider = StubProvider()
        assert provider._retry_config == RetryConfig()
        assert provider._provider_name == "unknown"
        assert provider._on_request_log is None
        assert provider._credential_pool is None
        assert provider._pool_config is None

    def test_init_with_custom_retry_config(self):
        cfg = RetryConfig(max_retries=5, base_delay=2.0)
        provider = StubProvider(retry_config=cfg)
        assert provider._retry_config.max_retries == 5
        assert provider._retry_config.base_delay == 2.0

    async def test_close(self):
        provider = StubProvider()
        client = await provider._get_client()
        assert not client.is_closed
        await provider.close()
        assert client.is_closed

    async def test_close_idempotent(self):
        provider = StubProvider()
        await provider._get_client()
        await provider.close()
        await provider.close()  # should not raise
