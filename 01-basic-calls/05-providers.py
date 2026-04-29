import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from the_agents_playbook import settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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


class BaseProvider(ABC):
    _client: httpx.AsyncClient | None = None

    # --- Abstract methods  ---
    @abstractmethod
    def _build_body(self, request: MessageRequest) -> dict[str, Any]:
        pass

    @abstractmethod
    def _build_headers(self) -> dict[str, str]:
        pass

    @abstractmethod
    def _chat_endpoint(self) -> str:
        pass

    @abstractmethod
    def _parse_response(self, response: httpx.Response) -> MessageResponse:
        pass

    # --- Public API ---
    async def send_message(self, request: MessageRequest) -> MessageResponse:
        client = await self._get_client()
        body = self._build_body(request)
        logging.info(
            f"Client configured with base URL: {client.base_url} and headers: {client.headers}"
        )
        logging.info(
            f"Sending request to {self._chat_endpoint()} with body: {json.dumps(body, indent=2)}"
        )
        response = await client.post(self._chat_endpoint(), json=body)
        response.raise_for_status()
        return self._parse_response(response)

    # -- Helpers ---
    # -- HTTP client management ---
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None and not self._client.is_closed:
            return self._client
        self._client = httpx.AsyncClient(
            headers=self._build_headers(),
            timeout=httpx.Timeout(10.0, connect=5.0, read=120.0, write=10.0, pool=10.0),
        )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


class OpenAIProvider(BaseProvider):
    def _chat_endpoint(self) -> str:
        return settings.openai_base_url + "/chat/completions"

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
            "User-Agent": "TheAgentsPlaybook/0.1",
        }

    def _build_body(self, request: MessageRequest) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend(m.model_dump() for m in request.messages)

        body: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream_options": {
                "include_usage": True,
                "include_reasoning": True,
            },
        }

        if request.tools:
            body["tools"] = [t.to_api_dict() for t in request.tools]

        if request.tool_choice:
            body["tool_choice"] = request.tool_choice.to_api_dict()

        if request.response_format:
            rf = request.response_format
            if rf.type == "json_schema" and rf.json_schema:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": rf.json_schema_name,
                        "strict": rf.strict,
                        "schema": rf.json_schema,
                    },
                }
            elif rf.type == "json_object":
                body["response_format"] = {"type": "json_object"}

        return body

    def _parse_response(self, response: httpx.Response) -> MessageResponse:
        raw = response.json()
        choice = raw["choices"][0]
        message = choice["message"]

        logging.info(f"Raw response from OpenAI API: {json.dumps(raw, indent=2)}")

        output = MessageResponse(
            message=OutputMessage(
                role=message["role"],
                content=message["content"],
                reasoning=message.get("reasoning"),
                tool_calls=message.get("tool_calls", []),
            ),
            stop_reason=choice.get("finish_reason", "unknown"),
        )
        logging.info(f"Received response from OpenAI API: {output}")
        return output


async def main():
    p = OpenAIProvider()

    # --- 1. Basic chat (backward compatibility — no new fields) ---
    print("=== Basic Chat ===")
    resp = await p.send_message(
        MessageRequest(
            messages=[
                InputMessage(role="user", content="What is the capital of Vietnam?")
            ],
            model=settings.openai_model,
        )
    )
    print(f"Content: {resp.message.content}")
    print(f"Stop:    {resp.stop_reason}\n")

    # --- 2. Structured output via response_format ---
    print("=== Structured Output (response_format) ===")
    resp = await p.send_message(
        MessageRequest(
            model=settings.openai_model,
            system="Extract movie review data from the user's text.",
            messages=[
                InputMessage(
                    role="user",
                    content="Avatar (2009) by James Cameron. Great visuals. 7.5/10. Sci-fi. Recommended.",
                )
            ],
            response_format=ResponseFormat(
                json_schema_name="MovieReview",
                json_schema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "year": {"type": "integer"},
                        "rating": {"type": "number"},
                        "recommended": {"type": "boolean"},
                    },
                    "required": ["title", "year", "rating", "recommended"],
                    "additionalProperties": False,
                },
            ),
        )
    )
    print(f"Content: {resp.message.content}")
    print(f"Stop:    {resp.stop_reason}\n")

    # --- 3. Tool choice forcing ---
    print("=== Tool Choice ===")
    resp = await p.send_message(
        MessageRequest(
            model=settings.openai_model,
            system="Extract movie review data by calling the submit_review tool.",
            messages=[
                InputMessage(
                    role="user",
                    content="Avatar (2009) by James Cameron. Great visuals. 7.5/10. Sci-fi. Recommended.",
                )
            ],
            tools=[
                ToolSpec(
                    name="submit_review",
                    description="Submit a structured movie review",
                    parameters={
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "year": {"type": "integer"},
                            "rating": {"type": "number"},
                            "recommended": {"type": "boolean"},
                        },
                        "required": ["title", "year", "rating", "recommended"],
                        "additionalProperties": False,
                    },
                )
            ],
            tool_choice=ToolChoice(type="function", function_name="submit_review"),
        )
    )
    print(f"Tool calls: {json.dumps(resp.message.tool_calls, indent=2)}")
    print(f"Stop:       {resp.stop_reason}\n")

    await p.close()


asyncio.run(main())
