from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class ProviderErrorCode(str, Enum):
    AUTH_FAILED = "auth_failed"
    RATE_LIMITED = "rate_limited"
    SERVER_ERROR = "server_error"
    CONTEXT_TOO_LONG = "context_too_long"
    BAD_REQUEST = "bad_request"
    TIMEOUT = "timeout"
    CONNECTION_FAILED = "connection_failed"
    UNKNOWN = "unknown"


class ProviderError(Exception):
    """Base error for all provider failures."""

    def __init__(
        self,
        message: str,
        code: ProviderErrorCode = ProviderErrorCode.UNKNOWN,
        status_code: int | None = None,
        retryable: bool = False,
        raw_body: dict | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable
        self.raw_body = raw_body


# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------


class RetryConfig(BaseModel):
    """Controls retry behavior for transient provider errors."""

    max_retries: int = Field(default=3, ge=0)
    base_delay: float = Field(default=1.0, gt=0)
    max_delay: float = Field(default=30.0, gt=0)
    jitter: bool = Field(default=True)
    retryable_codes: set[ProviderErrorCode] = Field(
        default_factory=lambda: {
            ProviderErrorCode.RATE_LIMITED,
            ProviderErrorCode.SERVER_ERROR,
            ProviderErrorCode.TIMEOUT,
            ProviderErrorCode.CONNECTION_FAILED,
        }
    )


# ---------------------------------------------------------------------------
# Request logging
# ---------------------------------------------------------------------------


@dataclass
class RequestLog:
    """Structured log entry for a single LLM request."""

    request_id: str
    provider: str
    model: str
    endpoint: str
    status_code: int | None = None
    error_code: ProviderErrorCode | None = None
    error_message: str | None = None
    duration_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    retry_count: int = 0
    timestamp: float = field(default_factory=monotonic)


class InputMessage(BaseModel):
    role: Literal["user", "assistant"] = "assistant"
    content: str


class OutputMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class ToolSpec(BaseModel):
    """A function definition that the LLM can call."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ResponseFormat(BaseModel):
    """Controls how the LLM formats its response."""

    type: Literal["json_object", "json_schema"] = "json_schema"
    json_schema_name: str | None = None
    json_schema: dict[str, Any] | None = None
    strict: bool = True


class ToolChoice(BaseModel):
    """Controls which tool the LLM must call."""

    type: Literal["auto", "required", "function"] = "auto"
    function_name: str | None = None

    def to_api_dict(self) -> dict[str, Any] | str:
        if self.type == "auto":
            return "auto"
        if self.type == "required":
            return "required"
        return {
            "type": "function",
            "function": {"name": self.function_name},
        }


class MessageRequest(BaseModel):
    model: str
    system: str = "You are a helpful assistant."
    temperature: float = 0.7
    max_tokens: int = 4096
    messages: list[InputMessage] = Field(default_factory=list)
    # Structured output and tool use — all default to empty/None for backward compatibility
    tools: list[ToolSpec] = Field(default_factory=list)
    tool_choice: ToolChoice | None = None
    response_format: ResponseFormat | None = None


class MessageResponse(BaseModel):
    message: OutputMessage
    stop_reason: str
