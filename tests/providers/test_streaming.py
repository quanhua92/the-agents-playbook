"""Unit tests for streaming support in providers."""

import json
from unittest.mock import patch

import httpx
import pytest
import respx

from the_agents_playbook.providers.anthropic import AnthropicProvider
from the_agents_playbook.providers.base import BaseProvider
from the_agents_playbook.providers.openai import OpenAIProvider
from the_agents_playbook.providers.types import (
    InputMessage,
    MessageRequest,
    ResponseChunk,
    RetryConfig,
)


def make_request(**overrides) -> MessageRequest:
    defaults = dict(
        model="test-model",
        messages=[InputMessage(role="user", content="Hello")],
        max_tokens=100,
    )
    defaults.update(overrides)
    return MessageRequest(**defaults)


# ---------------------------------------------------------------------------
# OpenAI streaming
# ---------------------------------------------------------------------------


class TestOpenAIStreaming:
    @pytest.fixture(autouse=True)
    def _patch_settings(self):
        with patch(
            "the_agents_playbook.providers.openai.settings"
        ) as mock_settings:
            mock_settings.openai_api_key = "test-key"
            mock_settings.openai_base_url = "https://api.example.com/v1"
            mock_settings.openai_model = "gpt-4o"
            yield

    def test_build_stream_body_has_stream_true(self):
        provider = OpenAIProvider()
        body = provider._build_stream_body(make_request())
        assert body["stream"] is True

    def test_parse_text_delta(self):
        provider = OpenAIProvider()
        data = json.dumps({
            "choices": [{"delta": {"content": "Hello"}, "index": 0}]
        })
        chunk = provider._parse_stream_chunk(data)
        assert chunk is not None
        assert chunk.delta_text == "Hello"

    def test_parse_reasoning_delta(self):
        provider = OpenAIProvider()
        data = json.dumps({
            "choices": [{"delta": {"reasoning": "Let me think"}, "index": 0}]
        })
        chunk = provider._parse_stream_chunk(data)
        assert chunk is not None
        assert chunk.delta_reasoning == "Let me think"

    def test_parse_tool_call_delta(self):
        provider = OpenAIProvider()
        data = json.dumps({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "id": "call_123",
                        "function": {"name": "get_weather", "arguments": ""},
                    }],
                },
                "index": 0,
            }],
        })
        chunk = provider._parse_stream_chunk(data)
        assert chunk is not None
        assert chunk.tool_call_id == "call_123"
        assert chunk.tool_call_name == "get_weather"

    def test_parse_finish_reason(self):
        provider = OpenAIProvider()
        data = json.dumps({
            "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]
        })
        chunk = provider._parse_stream_chunk(data)
        assert chunk is not None
        assert chunk.stop_reason == "stop"

    def test_parse_empty_delta(self):
        provider = OpenAIProvider()
        data = json.dumps({"choices": [{"delta": {}, "index": 0}]})
        chunk = provider._parse_stream_chunk(data)
        assert chunk is not None
        assert chunk.delta_text is None
        assert chunk.stop_reason is None

    def test_parse_invalid_json_returns_none(self):
        provider = OpenAIProvider()
        chunk = provider._parse_stream_chunk("not json")
        assert chunk is None

    @respx.mock
    async def test_stream_full_flow(self):
        sse_lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "Hello"}, "index": 0}]}),
            "data: " + json.dumps({"choices": [{"delta": {"content": " world"}, "index": 0}]}),
            "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]}),
            "data: [DONE]",
        ]
        respx.post("https://api.example.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, content="\n".join(sse_lines))
        )
        provider = OpenAIProvider(retry_config=RetryConfig(max_retries=0))
        chunks = []
        async for chunk in provider.stream(make_request()):
            chunks.append(chunk)

        assert len(chunks) == 4
        assert chunks[0].delta_text == "Hello"
        assert chunks[1].delta_text == " world"
        assert chunks[2].stop_reason == "stop"
        assert chunks[3].finish is True


# ---------------------------------------------------------------------------
# Anthropic streaming
# ---------------------------------------------------------------------------


class TestAnthropicStreaming:
    @pytest.fixture(autouse=True)
    def _patch_settings(self):
        with patch(
            "the_agents_playbook.providers.anthropic.settings"
        ) as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.anthropic_base_url = "https://api.anthropic.com/v1"
            mock_settings.anthropic_model = "claude-sonnet-4-6"
            yield

    def test_build_stream_body_has_stream_true(self):
        provider = AnthropicProvider()
        body = provider._build_stream_body(make_request())
        assert body["stream"] is True

    def test_parse_text_delta(self):
        provider = AnthropicProvider()
        data = json.dumps({
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello"},
        })
        chunk = provider._parse_stream_chunk(data)
        assert chunk is not None
        assert chunk.delta_text == "Hello"

    def test_parse_tool_use_json_delta(self):
        provider = AnthropicProvider()
        data = json.dumps({
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
        })
        chunk = provider._parse_stream_chunk(data)
        assert chunk is not None
        assert chunk.tool_call_arguments == '{"city":'

    def test_parse_message_start_returns_none(self):
        provider = AnthropicProvider()
        data = json.dumps({"type": "message_start", "message": {"id": "msg_1"}})
        chunk = provider._parse_stream_chunk(data)
        assert chunk is None

    def test_parse_message_delta_with_stop_reason(self):
        provider = AnthropicProvider()
        data = json.dumps({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
        })
        chunk = provider._parse_stream_chunk(data)
        assert chunk is not None
        assert chunk.stop_reason == "end_turn"

    def test_parse_message_stop(self):
        provider = AnthropicProvider()
        data = json.dumps({"type": "message_stop"})
        chunk = provider._parse_stream_chunk(data)
        assert chunk is not None
        assert chunk.finish is True

    def test_parse_content_block_start_tool_use(self):
        provider = AnthropicProvider()
        data = json.dumps({
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "search"},
        })
        chunk = provider._parse_stream_chunk(data)
        assert chunk is not None
        assert chunk.tool_call_id == "toolu_1"
        assert chunk.tool_call_name == "search"

    def test_parse_content_block_start_text_returns_none(self):
        provider = AnthropicProvider()
        data = json.dumps({
            "type": "content_block_start",
            "content_block": {"type": "text", "text": ""},
        })
        chunk = provider._parse_stream_chunk(data)
        assert chunk is None

    @respx.mock
    async def test_stream_full_flow(self):
        sse_lines = [
            "event: message_start",
            "data: " + json.dumps({"type": "message_start", "message": {}}),
            "event: content_block_delta",
            "data: " + json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}}),
            "event: content_block_delta",
            "data: " + json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": " there"}}),
            "event: message_delta",
            "data: " + json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn"}}),
            "event: message_stop",
            "data: " + json.dumps({"type": "message_stop"}),
        ]
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, content="\n".join(sse_lines))
        )
        provider = AnthropicProvider(retry_config=RetryConfig(max_retries=0))
        chunks = []
        async for chunk in provider.stream(make_request()):
            chunks.append(chunk)

        texts = [c.delta_text for c in chunks if c.delta_text]
        assert texts == ["Hi", " there"]
        stops = [c.stop_reason for c in chunks if c.stop_reason]
        assert stops == ["end_turn"]
        finishes = [c for c in chunks if c.finish]
        assert len(finishes) == 1


# ---------------------------------------------------------------------------
# BaseProvider SSE line parsing
# ---------------------------------------------------------------------------


class TestSSELineParsing:
    def test_data_prefix_stripped(self):
        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._build_stream_body = lambda r: {}
        provider._parse_stream_chunk = lambda d: ResponseChunk(delta_text=d)
        result = provider._parse_sse_line("data: hello")
        assert result.delta_text == "hello"

    def test_data_prefix_without_space(self):
        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._build_stream_body = lambda r: {}
        provider._parse_stream_chunk = lambda d: ResponseChunk(delta_text=d)
        result = provider._parse_sse_line("data:hello")
        assert result.delta_text == "hello"

    def test_done_signal(self):
        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._build_stream_body = lambda r: {}
        provider._parse_stream_chunk = lambda d: None
        result = provider._parse_sse_line("data: [DONE]")
        assert result.finish is True

    def test_empty_line_returns_none(self):
        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._build_stream_body = lambda r: {}
        provider._parse_stream_chunk = lambda d: None
        assert provider._parse_sse_line("") is None
        assert provider._parse_sse_line("  ") is None

    def test_comment_line_returns_none(self):
        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._build_stream_body = lambda r: {}
        provider._parse_stream_chunk = lambda d: None
        assert provider._parse_sse_line(": this is a comment") is None

    def test_event_line_returns_none(self):
        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._build_stream_body = lambda r: {}
        provider._parse_stream_chunk = lambda d: None
        assert provider._parse_sse_line("event: content_block_delta") is None
