import itertools
import logging
from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Any, Literal

from pydantic import BaseModel, Field

_pool_logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Credential rotation
# ---------------------------------------------------------------------------


class CredentialPool:
    """Manages multiple API keys with automatic rotation on failure.

    Usage:
        pool = CredentialPool(keys=["key-1", "key-2", "key-3"])
        pool.current  # "key-1"
        pool.rotate() # "key-2"
        pool.rotate() # "key-3"
        pool.rotate() # "key-1"  (wraps around)
    """

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("CredentialPool requires at least one key")
        self._keys = list(keys)
        self._cycle = itertools.cycle(range(len(keys)))
        self._index = next(self._cycle)

    @property
    def current(self) -> str:
        return self._keys[self._index]

    def rotate(self) -> str:
        """Move to the next key and return it."""
        self._index = next(self._cycle)
        _pool_logger.info(f"Rotated to credential index {self._index}")
        return self.current

    def __len__(self) -> int:
        return len(self._keys)


# ---------------------------------------------------------------------------
# Connection pool configuration
# ---------------------------------------------------------------------------


class PoolConfig(BaseModel):
    """Controls HTTP connection pool behavior."""

    max_connections: int = Field(default=10, gt=0)
    max_keepalive_connections: int = Field(default=5, gt=0)
    keepalive_expiry: float = Field(default=30.0, ge=0)


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
    timeout_seconds: float | None = Field(default=None, ge=1.0, le=600.0)


class MessageResponse(BaseModel):
    message: OutputMessage
    stop_reason: str


# ---------------------------------------------------------------------------
# Streaming types
# ---------------------------------------------------------------------------


class ResponseChunk(BaseModel):
    """A single chunk from a streaming response."""

    delta_text: str | None = None
    delta_reasoning: str | None = None
    tool_call_id: str | None = None
    tool_call_name: str | None = None
    tool_call_arguments: str | None = None
    stop_reason: str | None = None
    finish: bool = False


class StreamUsage(BaseModel):
    """Token usage reported at the end of a stream."""

    input_tokens: int = 0
    output_tokens: int = 0
