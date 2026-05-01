"""06-streaming.py — Stream a chat completion token-by-token via SSE.

This example shows how to use provider.stream() to get ResponseChunk
objects as they arrive, instead of waiting for the full response.

Run:
    uv run python 01-basic-calls/06-streaming.py
"""

import asyncio

from the_agents_playbook import settings
from the_agents_playbook.providers import (
    InputMessage,
    MessageRequest,
    OpenAIProvider,
)


async def run():
    provider = OpenAIProvider()

    request = MessageRequest(
        model=settings.openai_model,
        system="You are a helpful assistant. Keep answers brief.",
        messages=[
            InputMessage(role="user", content="Explain streaming in one sentence.")
        ],
        max_tokens=100,
    )

    print("Streaming response:\n")
    full_text = []
    async for chunk in provider.stream(request):
        if chunk.delta_text:
            print(chunk.delta_text, end="", flush=True)
            full_text.append(chunk.delta_text)
        if chunk.delta_reasoning:
            print(f"\n[reasoning] {chunk.delta_reasoning}", end="", flush=True)
        if chunk.stop_reason:
            print(f"\n\nStop reason: {chunk.stop_reason}")
        if chunk.finish:
            print("[stream complete]")

    print(f"\n\nFull response ({len(''.join(full_text))} chars): {''.join(full_text)}")

    await provider.close()


if __name__ == "__main__":
    asyncio.run(run())
