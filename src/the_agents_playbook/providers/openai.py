import json
import logging
from typing import Any

import httpx

from the_agents_playbook import settings
from the_agents_playbook.providers.base import BaseProvider
from the_agents_playbook.providers.types import (
    InputMessage,
    MessageRequest,
    MessageResponse,
    OutputMessage,
)


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


async def main():
    logging.basicConfig(level=logging.INFO)
    # 1. Initialize provider
    provider = OpenAIProvider()

    # 2. Create a mock request
    request = MessageRequest(
        messages=[InputMessage(role="user", content="What is the capital of Vietnam?")],
        model=settings.openai_model,
    )

    # 3. Run and print
    logging.info("--- Testing OpenAI Provider ---")
    response = await provider.send_message(request)
    logging.info(f"Response: {response}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
