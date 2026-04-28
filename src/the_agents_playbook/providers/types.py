from typing import Any, Literal

from pydantic import BaseModel, Field


class InputMessage(BaseModel):
    role: Literal["user", "assistant"] = "assistant"
    content: str


class OutputMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str
    reasoning: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class MessageRequest(BaseModel):
    model: str
    system: str = "You are a helpful assistant."
    temperature: float = 0.7
    max_tokens: int = 4096
    messages: list[InputMessage] = Field(default_factory=list)


class MessageResponse(BaseModel):
    message: OutputMessage
    stop_reason: str
