"""Shared fixtures for the providers test suite."""

from typing import Any

import pytest


@pytest.fixture
def mock_chat_response() -> dict[str, Any]:
    """A typical OpenAI-compatible chat completion response."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
        },
    }


@pytest.fixture
def mock_error_response() -> dict[str, Any]:
    """A typical error response body from an LLM API."""
    return {
        "error": {
            "message": "Rate limit exceeded",
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded",
        }
    }
