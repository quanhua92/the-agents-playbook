from typing import Any, Literal

from pydantic import BaseModel, Field


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
