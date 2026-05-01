"""Tests for provider type models: errors, configs, messages, tool specs."""

import pytest

from the_agents_playbook.providers.types import (
    CredentialPool,
    InputMessage,
    MessageRequest,
    MessageResponse,
    OutputMessage,
    PoolConfig,
    ProviderError,
    ProviderErrorCode,
    RequestLog,
    ResponseFormat,
    RetryConfig,
    ToolChoice,
    ToolSpec,
)


# ---------------------------------------------------------------------------
# ProviderErrorCode
# ---------------------------------------------------------------------------


class TestProviderErrorCode:
    def test_all_codes_are_strings(self):
        for code in ProviderErrorCode:
            assert isinstance(code.value, str)

    def test_expected_members_exist(self):
        expected = {
            "auth_failed",
            "rate_limited",
            "server_error",
            "context_too_long",
            "bad_request",
            "timeout",
            "connection_failed",
            "unknown",
        }
        actual = {code.value for code in ProviderErrorCode}
        assert actual == expected


# ---------------------------------------------------------------------------
# ProviderError
# ---------------------------------------------------------------------------


class TestProviderError:
    def test_defaults(self):
        err = ProviderError("something went wrong")
        assert err.code == ProviderErrorCode.UNKNOWN
        assert err.status_code is None
        assert err.retryable is False
        assert err.raw_body is None
        assert str(err) == "something went wrong"

    def test_custom_fields(self):
        err = ProviderError(
            "rate limited",
            code=ProviderErrorCode.RATE_LIMITED,
            status_code=429,
            retryable=True,
            raw_body={"error": "too many requests"},
        )
        assert err.code == ProviderErrorCode.RATE_LIMITED
        assert err.status_code == 429
        assert err.retryable is True
        assert err.raw_body == {"error": "too many requests"}

    def test_is_exception(self):
        with pytest.raises(ProviderError):
            raise ProviderError("boom")


# ---------------------------------------------------------------------------
# RetryConfig
# ---------------------------------------------------------------------------


class TestRetryConfig:
    def test_defaults(self):
        cfg = RetryConfig()
        assert cfg.max_retries == 3
        assert cfg.base_delay == 1.0
        assert cfg.max_delay == 30.0
        assert cfg.jitter is True
        assert ProviderErrorCode.RATE_LIMITED in cfg.retryable_codes
        assert ProviderErrorCode.SERVER_ERROR in cfg.retryable_codes
        assert ProviderErrorCode.AUTH_FAILED not in cfg.retryable_codes

    def test_custom(self):
        cfg = RetryConfig(max_retries=0, base_delay=0.5)
        assert cfg.max_retries == 0
        assert cfg.base_delay == 0.5

    def test_zero_retries_is_valid(self):
        cfg = RetryConfig(max_retries=0)
        assert cfg.max_retries == 0

    def test_invalid_base_delay(self):
        with pytest.raises(Exception):
            RetryConfig(base_delay=0)

    def test_invalid_max_retries(self):
        with pytest.raises(Exception):
            RetryConfig(max_retries=-1)


# ---------------------------------------------------------------------------
# PoolConfig
# ---------------------------------------------------------------------------


class TestPoolConfig:
    def test_defaults(self):
        cfg = PoolConfig()
        assert cfg.max_connections == 10
        assert cfg.max_keepalive_connections == 5
        assert cfg.keepalive_expiry == 30.0

    def test_custom(self):
        cfg = PoolConfig(max_connections=20, keepalive_expiry=60.0)
        assert cfg.max_connections == 20
        assert cfg.keepalive_expiry == 60.0


# ---------------------------------------------------------------------------
# RequestLog
# ---------------------------------------------------------------------------


class TestRequestLog:
    def test_defaults(self):
        log = RequestLog(
            request_id="test-0",
            provider="openai",
            model="gpt-4",
            endpoint="https://api.openai.com/v1/chat/completions",
        )
        assert log.request_id == "test-0"
        assert log.status_code is None
        assert log.error_code is None
        assert log.error_message is None
        assert log.duration_ms is None
        assert log.input_tokens is None
        assert log.output_tokens is None
        assert log.cost_usd is None
        assert log.retry_count == 0
        assert isinstance(log.timestamp, float)

    def test_with_error_fields(self):
        log = RequestLog(
            request_id="test-1",
            provider="openai",
            model="gpt-4",
            endpoint="https://api.openai.com/v1/chat/completions",
            status_code=429,
            error_code=ProviderErrorCode.RATE_LIMITED,
            error_message="Rate limited",
            duration_ms=150.5,
            retry_count=2,
        )
        assert log.status_code == 429
        assert log.error_code == ProviderErrorCode.RATE_LIMITED
        assert log.retry_count == 2


# ---------------------------------------------------------------------------
# CredentialPool
# ---------------------------------------------------------------------------


class TestCredentialPool:
    def test_single_key(self):
        pool = CredentialPool(keys=["key-1"])
        assert pool.current == "key-1"
        assert len(pool) == 1

    def test_multiple_keys_round_robin(self):
        pool = CredentialPool(keys=["a", "b", "c"])
        assert pool.current == "a"
        assert pool.rotate() == "b"
        assert pool.rotate() == "c"
        assert pool.rotate() == "a"  # wraps around

    def test_empty_keys_raises(self):
        with pytest.raises(ValueError, match="at least one key"):
            CredentialPool(keys=[])

    def test_len(self):
        pool = CredentialPool(keys=["x", "y"])
        assert len(pool) == 2


# ---------------------------------------------------------------------------
# Message models
# ---------------------------------------------------------------------------


class TestInputMessage:
    def test_user_message(self):
        msg = InputMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_default_role(self):
        msg = InputMessage(content="Hi")
        assert msg.role == "assistant"


class TestOutputMessage:
    def test_defaults(self):
        out = OutputMessage()
        assert out.role == "assistant"
        assert out.content is None
        assert out.reasoning is None
        assert out.tool_calls == []

    def test_with_fields(self):
        out = OutputMessage(content="Sure!", reasoning="thinking...")
        assert out.content == "Sure!"
        assert out.reasoning == "thinking..."

    def test_with_tool_calls(self):
        out = OutputMessage(
            tool_calls=[{"id": "call_1", "function": {"name": "get_weather"}}]
        )
        assert len(out.tool_calls) == 1


class TestMessageRequest:
    def test_defaults(self):
        req = MessageRequest(model="gpt-4")
        assert req.model == "gpt-4"
        assert req.system == "You are a helpful assistant."
        assert req.temperature == 0.7
        assert req.max_tokens == 4096
        assert req.messages == []
        assert req.tools == []
        assert req.tool_choice is None
        assert req.response_format is None
        assert req.timeout_seconds is None

    def test_with_messages(self):
        req = MessageRequest(
            model="gpt-4",
            messages=[InputMessage(role="user", content="Test")],
        )
        assert len(req.messages) == 1

    def test_timeout_seconds(self):
        req = MessageRequest(model="gpt-4", timeout_seconds=5.0)
        assert req.timeout_seconds == 5.0

    def test_timeout_seconds_out_of_range(self):
        with pytest.raises(Exception):
            MessageRequest(model="gpt-4", timeout_seconds=0.1)
        with pytest.raises(Exception):
            MessageRequest(model="gpt-4", timeout_seconds=700.0)


class TestMessageResponse:
    def test_basic(self):
        resp = MessageResponse(
            message=OutputMessage(content="Hello!"),
            stop_reason="stop",
        )
        assert resp.message.content == "Hello!"
        assert resp.stop_reason == "stop"


# ---------------------------------------------------------------------------
# ToolSpec, ToolChoice, ResponseFormat
# ---------------------------------------------------------------------------


class TestToolSpec:
    def test_to_api_dict(self):
        spec = ToolSpec(
            name="get_weather",
            description="Get the weather for a city",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        d = spec.to_api_dict()
        assert d["type"] == "function"
        assert d["function"]["name"] == "get_weather"
        assert d["function"]["description"] == "Get the weather for a city"
        assert "parameters" in d["function"]


class TestToolChoice:
    def test_auto(self):
        tc = ToolChoice(type="auto")
        assert tc.to_api_dict() == "auto"

    def test_required(self):
        tc = ToolChoice(type="required")
        assert tc.to_api_dict() == "required"

    def test_function(self):
        tc = ToolChoice(type="function", function_name="get_weather")
        d = tc.to_api_dict()
        assert d["type"] == "function"
        assert d["function"]["name"] == "get_weather"

    def test_default(self):
        tc = ToolChoice()
        assert tc.type == "auto"


class TestResponseFormat:
    def test_json_schema(self):
        rf = ResponseFormat(
            type="json_schema",
            json_schema_name="WeatherOutput",
            json_schema={"type": "object", "properties": {}},
            strict=True,
        )
        assert rf.type == "json_schema"
        assert rf.json_schema_name == "WeatherOutput"
        assert rf.strict is True

    def test_json_object(self):
        rf = ResponseFormat(type="json_object")
        assert rf.type == "json_object"
        assert rf.json_schema is None

    def test_default(self):
        rf = ResponseFormat()
        assert rf.type == "json_schema"
        assert rf.strict is True
