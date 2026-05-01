"""01_checkpoint.py -- MemorySaver replaces SessionPersistence (JSONL).

In the root project:
  session = SessionPersistence()
  await session.save(messages, Path("session.jsonl"))
  messages = await session.load(Path("session.jsonl"))

In LangGraph:
  checkpointer = MemorySaver()
  agent = create_agent(llm, tools, checkpointer=checkpointer)
  # Turn 1
  agent.invoke({"messages": [HumanMessage(content="Hello")]}, config)
  # Turn 2 -- agent remembers from checkpoint
  agent.invoke({"messages": [HumanMessage(content="What did I say?")]}, config)

MemorySaver persists full graph state keyed by thread_id.
Swap with SqliteSaver for disk persistence.
"""

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

from shared import get_openai_llm


@tool
def echo(message: str) -> str:
    """Echo back the input message."""
    return f"Echo: {message}"


def main():
    llm = get_openai_llm()
    checkpointer = MemorySaver()

    agent = create_agent(llm, [echo], checkpointer=checkpointer)

    # --- Thread 1: multi-turn conversation ---
    config1: RunnableConfig = {"configurable": {"thread_id": "conversation-1"}}

    print("=== Turn 1 ===")
    result = agent.invoke(
        {"messages": [HumanMessage(content="My favorite color is blue.")]},
        config1,
    )
    print(result["messages"][-1].content)

    print("\n=== Turn 2 (agent remembers) ===")
    result = agent.invoke(
        {"messages": [HumanMessage(content="What's my favorite color?")]},
        config1,
    )
    print(result["messages"][-1].content)

    # --- Thread 2: independent memory ---
    print("\n=== Different Thread (independent) ===")
    config2: RunnableConfig = {"configurable": {"thread_id": "conversation-2"}}
    result2 = agent.invoke(
        {"messages": [HumanMessage(content="What's my favorite color?")]},
        config2,
    )
    print(result2["messages"][-1].content)
    print("(Should say it doesn't know -- different thread)")


if __name__ == "__main__":
    main()
