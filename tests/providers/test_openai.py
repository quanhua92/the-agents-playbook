"""Integration tests for OpenAIProvider — hits the real API via .env settings.

Run with:
    uv run pytest tests/providers/test_openai.py -v
    uv run pytest tests/ -v -m openai     (only integration tests)
    uv run pytest tests/ -v -m "not openai" (only unit tests)
"""

import json

import httpx
import pytest
from pydantic import BaseModel

from the_agents_playbook import settings
from the_agents_playbook.providers.openai import OpenAIProvider
from the_agents_playbook.providers.types import (
    InputMessage,
    MessageRequest,
    MessageResponse,
    OutputMessage,
    ResponseFormat,
    ToolChoice,
    ToolSpec,
)


# Skip the entire module if no API key is configured
pytestmark = pytest.mark.openai


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def provider():
    p = OpenAIProvider(provider_name="openai-test")
    yield p
    await p.close()


def make_request(**overrides) -> MessageRequest:
    defaults = dict(
        model=settings.openai_model,
        messages=[InputMessage(role="user", content="Say exactly: hello world")],
        max_tokens=50,
    )
    defaults.update(overrides)
    return MessageRequest(**defaults)


# ---------------------------------------------------------------------------
# Basic chat completion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_basic_chat_completion(provider):
    """Send a simple message and get a response back."""
    request = make_request()
    response = await provider.send_message(request)

    assert isinstance(response, MessageResponse)
    assert response.message.role == "assistant"
    assert isinstance(response.message.content, str)
    assert len(response.message.content) > 0
    assert response.stop_reason in ("stop", "length")


@pytest.mark.asyncio
async def test_system_prompt(provider):
    """Verify system prompt is respected."""
    request = make_request(
        messages=[InputMessage(role="user", content="What is 1+1?")],
        system="You are a math tutor. Always answer with just a number, nothing else.",
        max_tokens=20,
    )
    response = await provider.send_message(request)
    assert response.message.content is not None
    # Should be a short numeric answer
    content = response.message.content.strip()
    assert any(c.isdigit() for c in content)


# ---------------------------------------------------------------------------
# Structured output — json_schema
# ---------------------------------------------------------------------------


class Answer(BaseModel):
    capital: str
    country: str


answer_schema = Answer.model_json_schema()


@pytest.mark.asyncio
async def test_structured_output_json_schema(provider):
    """Request JSON schema output and verify it matches the schema."""
    request = make_request(
        messages=[InputMessage(role="user", content="What is the capital of France?")],
        response_format=ResponseFormat(
            type="json_schema",
            json_schema_name="Answer",
            json_schema=answer_schema,
            strict=True,
        ),
        max_tokens=100,
    )
    response = await provider.send_message(request)
    assert response.message.content is not None

    parsed = json.loads(response.message.content)
    answer = Answer(**parsed)
    assert answer.capital.lower() == "paris"
    assert answer.country.lower() == "france"


# ---------------------------------------------------------------------------
# Structured output — json_object
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_structured_output_json_object(provider):
    """Request generic json_object output."""
    request = make_request(
        messages=[
            InputMessage(role="user", content='Return JSON: {"color": "blue", "shape": "circle"}'),
        ],
        response_format=ResponseFormat(type="json_object"),
        max_tokens=50,
    )
    response = await provider.send_message(request)
    assert response.message.content is not None
    parsed = json.loads(response.message.content)
    assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Tool use
# ---------------------------------------------------------------------------


get_weather_spec = ToolSpec(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "The city name"},
        },
        "required": ["city"],
        "additionalProperties": False,
    },
)


@pytest.mark.asyncio
async def test_tool_call(provider):
    """Request the model to call a tool."""
    request = make_request(
        messages=[InputMessage(role="user", content="What's the weather in Tokyo?")],
        tools=[get_weather_spec],
        tool_choice=ToolChoice(type="required"),
        max_tokens=100,
    )
    response = await provider.send_message(request)

    assert isinstance(response, MessageResponse)
    # Should have tool calls, not text content
    assert len(response.message.tool_calls) > 0
    call = response.message.tool_calls[0]
    assert call["function"]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_tool_choice_auto(provider):
    """With auto tool_choice, model decides whether to call the tool."""
    request = make_request(
        messages=[InputMessage(role="user", content="Hello, how are you?")],
        tools=[get_weather_spec],
        tool_choice=ToolChoice(type="auto"),
        max_tokens=50,
    )
    response = await provider.send_message(request)
    # For a greeting, the model should reply with text, not call weather
    assert isinstance(response, MessageResponse)
    # Either a text reply or a tool call is valid; just verify it's a proper response
    assert response.stop_reason in ("stop", "length", "tool_calls")


# ---------------------------------------------------------------------------
# Retry and error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bad_request_raises_provider_error(provider):
    """A malformed request should raise ProviderError with BAD_REQUEST code."""
    from the_agents_playbook.providers.types import ProviderError, ProviderErrorCode

    # Use an invalid model name to trigger a 400
    request = MessageRequest(
        model="nonexistent-invalid-model-xyz",
        messages=[InputMessage(role="user", content="test")],
        max_tokens=10,
    )
    with pytest.raises(ProviderError) as exc_info:
        await provider.send_message(request)
    assert exc_info.value.code in (
        ProviderErrorCode.BAD_REQUEST,
        ProviderErrorCode.UNKNOWN,
    )
    assert exc_info.value.status_code is not None


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_log_callback(provider):
    """Verify the on_request_log callback fires with correct fields."""
    from the_agents_playbook.providers.types import RequestLog

    collected: list[RequestLog] = []

    # Create a new provider with the callback
    log_provider = OpenAIProvider(
        provider_name="openai-log-test",
        on_request_log=collected.append,
        retry_config=None,  # use default
    )
    try:
        request = make_request()
        response = await log_provider.send_message(request)

        assert len(collected) >= 1
        log = collected[0]
        assert log.request_id == "openai-log-test-0"
        assert log.provider == "openai-log-test"
        assert log.model == settings.openai_model
        assert log.status_code == 200
        assert log.duration_ms is not None
        assert log.duration_ms > 0
        assert log.error_code is None
    finally:
        await log_provider.close()


# ---------------------------------------------------------------------------
# Per-request timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_request_timeout(provider):
    """Verify a request with a short timeout still succeeds for simple queries."""
    request = make_request(timeout_seconds=10.0)
    response = await provider.send_message(request)
    assert response.message.content is not None
