"""05_streaming.py -- Real-time streaming with a LangGraph ReAct agent.

Uses .astream_events() to show real-time interleaved streaming of
text deltas and tool calls from a LangGraph agent.

This demonstrates the same concept as run_streaming() in the root
project's Agent class, but using LangGraph's built-in streaming API.
"""

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from shared import get_openai_llm


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@tool
def get_word_length(word: str) -> str:
    """Get the number of characters in a word."""
    return str(len(word))


def main():
    print("=== Streaming LangGraph Agent ===\n")

    llm = get_openai_llm()

    agent = create_react_agent(
        model=llm,
        tools=[calculate, get_word_length],
        prompt="You are a helpful assistant. Show your work step by step.",
    )

    task = "What is 123 * 456? Then count the letters in 'supercalifragilistic'."
    print(f"Task: {task}\n")
    print("--- Streaming events ---\n")

    # Use astream_events for token-level streaming
    for event in agent.astream_events(
        {"messages": [HumanMessage(content=task)]},
        version="v2",
    ):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            # Text delta
            chunk = event["data"]["chunk"]
            if chunk.content:
                print(chunk.content, end="", flush=True)

        elif kind == "on_tool_start":
            # Tool call starting
            print(f"\n\n[tool_call] {event['name']}({event['data'].get('input', {})})")

        elif kind == "on_tool_end":
            # Tool result
            output = event["data"].get("output", "")
            print(f"[tool_result] {output}")

    print("\n\n=== Stream Complete ===")
    print("Note how text appears token-by-token with tool calls interleaved.")
    print("This is the same pattern as Agent.run_streaming() in the root project.")


if __name__ == "__main__":
    main()
