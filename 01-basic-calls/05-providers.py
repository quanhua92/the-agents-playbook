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


def main():
    p = OpenAIProvider()
    asyncio.run(
        p.send_message(
            MessageRequest(
                messages=[
                    InputMessage(role="user", content="What is the capital of Vietnam?")
                ],
                model=settings.openai_model,
            )
        )
    )


main()
