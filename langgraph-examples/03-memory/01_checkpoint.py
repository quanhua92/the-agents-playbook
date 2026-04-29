"""01_checkpoint.py -- MemorySaver replaces SessionPersistence (JSONL).

In the root project:
  session = SessionPersistence()
  await session.save(messages, Path("session.jsonl"))
  messages = await session.load(Path("session.jsonl"))

In LangGraph:
  checkpointer = MemorySaver()
  agent = create_react_agent(llm, tools, checkpointer=checkpointer)
  # Turn 1
  agent.invoke({"messages": [("user", "Hello")]}, config)
  # Turn 2 -- agent remembers from checkpoint
  agent.invoke({"messages": [("user", "What did I say?")]}, config)

MemorySaver persists full graph state keyed by thread_id.
Swap with SqliteSaver for disk persistence.
"""

from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from shared import get_openai_llm


@tool
def echo(message: str) -> str:
    """Echo back the input message."""
    return f"Echo: {message}"


def main():
    llm = get_openai_llm()
    checkpointer = MemorySaver()

    agent = create_react_agent(llm, [echo], checkpointer=checkpointer)

    # --- Thread 1: multi-turn conversation ---
    config1 = {"configurable": {"thread_id": "conversation-1"}}

    print("=== Turn 1 ===")
    result = agent.invoke(
        {"messages": [("user", "My favorite color is blue.")]},
        config1,
    )
    print(result["messages"][-1].content)

    print("\n=== Turn 2 (agent remembers) ===")
    result = agent.invoke(
        {"messages": [("user", "What's my favorite color?")]},
        config1,
    )
    print(result["messages"][-1].content)

    # --- Thread 2: independent memory ---
    print("\n=== Different Thread (independent) ===")
    config2 = {"configurable": {"thread_id": "conversation-2"}}
    result2 = agent.invoke(
        {"messages": [("user", "What's my favorite color?")]},
        config2,
    )
    print(result2["messages"][-1].content)
    print("(Should say it doesn't know -- different thread)")


if __name__ == "__main__":
    main()
