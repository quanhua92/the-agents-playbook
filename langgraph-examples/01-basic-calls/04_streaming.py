"""04_streaming.py -- astream() for token-by-token streaming.

Replaces the root project's provider.stream() + ResponseChunk in 06-streaming.py.
No SSE parsing, no ResponseChunk objects -- just iterate over the async generator.
"""

import asyncio

from langchain_core.messages import HumanMessage

from shared import get_openai_llm


async def main():
    llm = get_openai_llm()

    print("=== Streaming ===\n")
    full_text: list[str] = []

    async for chunk in llm.astream([
        HumanMessage(content="Explain LangGraph in two sentences."),
    ]):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_text.append(chunk.content)

    print(f"\n\n=== Summary ===")
    print(f"Total chunks received: {len(full_text)}")
    print(f"Full response: {''.join(full_text)}")

    # Streaming with structured output
    print("\n=== Streaming with Tools ===\n")
    from langchain_core.tools import tool

    @tool
    def calculate(expression: str) -> str:
        """Evaluate a math expression."""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error: {e}"

    llm_with_tools = llm.bind_tools([calculate])
    full_text2: list[str] = []

    async for chunk in llm_with_tools.astream("What is 15 * 27?"):
        # Tool calls appear as chunks too
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_text2.append(chunk.content)
        if chunk.tool_call_chunks:
            for tc in chunk.tool_call_chunks:
                if tc.get("name"):
                    print(f"\n[Tool call: {tc['name']}]", end="")
                if tc.get("args"):
                    print(f" args={tc['args']}", end="")

    print(f"\n\nFull response: {''.join(full_text2) if full_text2 else '(tool call only)'}")


if __name__ == "__main__":
    asyncio.run(main())
