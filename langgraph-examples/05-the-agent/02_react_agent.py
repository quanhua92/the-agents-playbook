"""02_react_agent.py -- create_react_agent replaces root's Agent class.

In the root project (240 lines):
  agent = Agent(
      provider=OpenAIProvider(),
      registry=registry,
      memory=memory,
      context_builder=builder,
      config=AgentConfig(max_tool_iterations=10),
  )
  async for event in agent.run("Fix the bug"):
      if event.type == "tool_call": ...
      elif event.type == "text": ...

In LangGraph (one function call):
  agent = create_react_agent(llm, tools, checkpointer=checkpointer)
  result = agent.invoke({"messages": [("user", "Fix the bug")]}, config)

The prebuilt agent internally uses a StateGraph with ToolNode and
conditional edges -- the same pattern you'd build by hand.
"""

from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from shared import get_openai_llm


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


def main():
    llm = get_openai_llm()
    checkpointer = MemorySaver()

    # This single call replaces the entire Agent class
    agent = create_react_agent(llm, [add_numbers, multiply_numbers], checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "math-session"}}

    # Task requiring multi-step tool use: (3 + 5) * 4
    print("=== Task: What is (3 + 5) * 4? ===\n")
    result = agent.invoke(
        {"messages": [("user", "What is (3 + 5) * 4?")]},
        config,
    )

    # Show full message trace
    print("=== Message Trace ===")
    for msg in result["messages"]:
        role = msg.type
        if role == "human":
            print(f"  [USER] {msg.content}")
        elif role == "ai":
            if msg.content:
                print(f"  [AI] {msg.content}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  [TOOL CALL] {tc['name']}({tc['args']})")
        elif role == "tool":
            print(f"  [TOOL RESULT] {msg.content}")

    # Follow-up turn (uses checkpoint memory)
    print("\n=== Follow-up: What was the first step? ===\n")
    result2 = agent.invoke(
        {"messages": [("user", "What was the first calculation step?")]},
        config,
    )
    print(f"  [AI] {result2['messages'][-1].content}")

    print("\n=== Root Comparison ===")
    print("Root Agent class:     ~240 lines of ReAct loop code")
    print("create_react_agent:   1 function call")
    print("Both produce:         multi-step tool use with memory")


if __name__ == "__main__":
    main()
