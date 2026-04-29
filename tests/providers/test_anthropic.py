"""Unit tests for AnthropicProvider — mocked HTTP with respx."""

import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from the_agents_playbook.providers.anthropic import AnthropicProvider
from the_agents_playbook.providers.types import (
    InputMessage,
    MessageRequest,
    MessageResponse,
    RetryConfig,
    ToolChoice,
    ToolSpec,
)


@pytest.fixture
def anthropic_response() -> dict[str, Any]:
    """A typical Anthropic Messages API response."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello! How can I help you?"}],
        "model": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 8},
    }


@pytest.fixture
def anthropic_tool_use_response() -> dict[str, Any]:
    """An Anthropic response with tool_use content blocks."""
    return {
        "id": "msg_test_tool",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_01A",
                "name": "get_weather",
                "input": {"city": "Tokyo"},
            }
        ],
        "model": "claude-sonnet-4-6",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 20, "output_tokens": 15},
    }


def make_request(**overrides) -> MessageRequest:
    defaults = dict(
        model="claude-sonnet-4-6",
        messages=[InputMessage(role="user", content="Hello")],
        max_tokens=100,
    )
    defaults.update(overrides)
    return MessageRequest(**defaults)


class TestAnthropicBuildHeaders:
    def test_headers_include_x_api_key(self):
        with patch(
            "the_agents_playbook.providers.anthropic.settings"
        ) as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.anthropic_base_url = "https://api.anthropic.com/v1"
            provider = AnthropicProvider()
            headers = provider._build_headers()
        assert headers["x-api-key"] == "test-key"
        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["Content-Type"] == "application/json"

    def test_build_auth_from_pool_uses_x_api_key(self):
        from the_agents_playbook.providers.types import CredentialPool

        pool = CredentialPool(keys=["my-key"])
        provider = AnthropicProvider(credential_pool=pool)
        header, value = provider._build_auth_from_pool()
        assert header == "x-api-key"
        assert value == "my-key"


class TestAnthropicBuildBody:
    @pytest.fixture(autouse=True)
    def _patch_settings(self):
        with patch(
            "the_agents_playbook.providers.anthropic.settings"
        ) as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.anthropic_base_url = "https://api.anthropic.com/v1"
            mock_settings.anthropic_model = "claude-sonnet-4-20250514"
            self.mock_settings = mock_settings
            yield

    def test_system_goes_to_top_level(self):
        provider = AnthropicProvider()
        body = provider._build_body(
            make_request(system="You are a math tutor.")
        )
        assert body["system"] == "You are a math tutor."
        assert all(m["role"] != "system" for m in body["messages"])

    def test_no_system_omits_key(self):
        provider = AnthropicProvider()
        body = provider._build_body(
            make_request(system="")
        )
        assert "system" not in body or body["system"] == ""

    def test_tools_converted_to_anthropic_format(self):
        provider = AnthropicProvider()
        spec = ToolSpec(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        body = provider._build_body(make_request(tools=[spec]))
        assert len(body["tools"]) == 1
        tool = body["tools"][0]
        assert tool["name"] == "search"
        assert tool["description"] == "Search the web"
        assert tool["input_schema"] == {
            "type": "object",
            "properties": {"q": {"type": "string"}},
        }
        # Should NOT have "type": "function" wrapper (Anthropic format)
        assert "type" not in tool

    def test_tool_choice_required_maps_to_any(self):
        provider = AnthropicProvider()
        body = provider._build_body(
            make_request(tool_choice=ToolChoice(type="required"))
        )
        assert body["tool_choice"] == {"type": "any"}

    def test_tool_choice_function_maps_correctly(self):
        provider = AnthropicProvider()
        body = provider._build_body(
            make_request(tool_choice=ToolChoice(type="function", function_name="search"))
        )
        assert body["tool_choice"] == {"type": "tool", "name": "search"}

    def test_tool_choice_auto_omitted(self):
        provider = AnthropicProvider()
        body = provider._build_body(
            make_request(tool_choice=ToolChoice(type="auto"))
        )
        assert "tool_choice" not in body


class TestAnthropicParseResponse:
    @pytest.fixture(autouse=True)
    def _patch_settings(self):
        with patch(
            "the_agents_playbook.providers.anthropic.settings"
        ) as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.anthropic_base_url = "https://api.anthropic.com/v1"
            mock_settings.anthropic_model = "claude-sonnet-4-6"
            yield

    def test_parse_text_response(self, anthropic_response):
        provider = AnthropicProvider()
        resp = httpx.Response(200, json=anthropic_response)
        result = provider._parse_response(resp)
        assert isinstance(result, MessageResponse)
        assert result.message.content == "Hello! How can I help you?"
        assert result.stop_reason == "end_turn"

    def test_parse_multiple_text_blocks_joined(self):
        provider = AnthropicProvider()
        raw = {
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": " world"},
            ],
            "stop_reason": "end_turn",
        }
        resp = httpx.Response(200, json=raw)
        result = provider._parse_response(resp)
        assert result.message.content == "Hello\n world"

    def test_parse_tool_use_response(self, anthropic_tool_use_response):
        provider = AnthropicProvider()
        resp = httpx.Response(200, json=anthropic_tool_use_response)
        result = provider._parse_response(resp)
        assert result.stop_reason == "tool_calls"
        assert len(result.message.tool_calls) == 1
        call = result.message.tool_calls[0]
        assert call["id"] == "toolu_01A"
        assert call["function"]["name"] == "get_weather"
        args = json.loads(call["function"]["arguments"])
        assert args == {"city": "Tokyo"}

    def test_parse_empty_content(self):
        provider = AnthropicProvider()
        raw = {"content": [], "stop_reason": "end_turn"}
        resp = httpx.Response(200, json=raw)
        result = provider._parse_response(resp)
        assert result.message.content is None
        assert result.message.tool_calls == []

    def test_unknown_stop_reason_maps_to_unknown(self):
        provider = AnthropicProvider()
        raw = {"content": [{"type": "text", "text": "hi"}], "stop_reason": "something_weird"}
        resp = httpx.Response(200, json=raw)
        result = provider._parse_response(resp)
        assert result.stop_reason == "unknown"

    def test_max_tokens_stop_reason(self):
        provider = AnthropicProvider()
        raw = {"content": [{"type": "text", "text": "truncated"}], "stop_reason": "max_tokens"}
        resp = httpx.Response(200, json=raw)
        result = provider._parse_response(resp)
        assert result.stop_reason == "max_tokens"


class TestAnthropicSend:
    @pytest.fixture(autouse=True)
    def _patch_settings(self):
        with patch(
            "the_agents_playbook.providers.anthropic.settings"
        ) as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.anthropic_base_url = "https://api.anthropic.com/v1"
            mock_settings.anthropic_model = "claude-sonnet-4-20250514"
            yield

    @respx.mock
    async def test_send_message_succeeds(self, anthropic_response):
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=anthropic_response),
        )
        provider = AnthropicProvider(
            retry_config=RetryConfig(max_retries=0),
        )
        result = await provider.send_message(make_request())
        assert result.message.content == "Hello! How can I help you?"
        assert result.stop_reason == "end_turn"

    @respx.mock
    async def test_send_message_with_system(self, anthropic_response):
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=anthropic_response),
        )
        provider = AnthropicProvider(
            retry_config=RetryConfig(max_retries=0),
        )
        await provider.send_message(make_request(system="You are helpful."))
        # Verify the system was sent as top-level, not in messages
        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["system"] == "You are helpful."
        assert all(m["role"] != "system" for m in sent_body["messages"])

    @respx.mock
    async def test_send_message_429_retries(self, anthropic_response):
        respx.post("https://api.anthropic.com/v1/messages").mock(
            side_effect=[
                httpx.Response(429, json={"error": "rate limited"}),
                httpx.Response(200, json=anthropic_response),
            ],
        )
        provider = AnthropicProvider(
            retry_config=RetryConfig(max_retries=3, base_delay=0.01, max_delay=0.05, jitter=False),
        )
        result = await provider.send_message(make_request())
        assert result.message.content is not None

    @respx.mock
    async def test_request_log_callback(self, anthropic_response):
        from the_agents_playbook.providers.types import RequestLog

        collected: list[RequestLog] = []
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=anthropic_response),
        )
        provider = AnthropicProvider(
            provider_name="anthropic-test",
            on_request_log=collected.append,
            retry_config=RetryConfig(max_retries=0),
        )
        await provider.send_message(make_request())
        assert len(collected) == 1
        log = collected[0]
        assert log.request_id == "anthropic-test-0"
        assert log.provider == "anthropic-test"
        assert log.status_code == 200
