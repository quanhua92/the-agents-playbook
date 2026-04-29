"""04_tool_chaining.py -- Multi-step tool use within a ReAct agent.

In the root project:
  chainer = ToolChainer(registry, max_chain_length=3, entropy_threshold=0.5)
  # Manually manages sequential tool calls with entropy re-scoring

In LangGraph:
  agent = create_react_agent(llm, [search, lookup, calculate])
  result = agent.invoke({"messages": [("user", "...")]})
  # The LLM decides when to chain tools and when to stop

The agent handles multi-step tool use natively. The LLM reasons about
which tools to call in what order to accomplish the task.
"""

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from shared import get_openai_llm


@tool
def search(topic: str) -> str:
    """Search for information on a topic."""
    results = {
        "python": "Python is a high-level programming language known for readability.",
        "tokyo": "Tokyo is the capital of Japan with approximately 14 million residents.",
        "gdp": "World GDP in 2024 was approximately $105 trillion USD.",
    }
    for key, val in results.items():
        if key in topic.lower():
            return val
    return f"No results found for '{topic}'."


@tool
def lookup_population(city: str) -> str:
    """Look up the population of a city."""
    data = {
        "tokyo": "14 million",
        "london": "9 million",
        "new york": "8 million",
        "paris": "2.2 million",
    }
    return data.get(city.lower(), f"Population data for '{city}' not available.")


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


def main():
    llm = get_openai_llm()
    agent = create_react_agent(llm, [search, lookup_population, calculate])

    # Task requiring multiple tool calls chained together
    prompt = (
        "Search for information about Tokyo, "
        "then look up its population, "
        "and calculate what that is times 2."
    )

    print(f"=== Task ===\n{prompt}\n")
    result = agent.invoke({"messages": [("user", prompt)]})

    # Show the tool chain
    print("=== Tool Chain ===")
    tool_call_count = 0
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_call_count += 1
                print(f"  Step {tool_call_count}: {tc['name']}({tc['args']})")
        elif hasattr(msg, "type") and msg.type == "tool":
            print(f"    -> {msg.content}")

    print(f"\n=== Final Answer ===")
    print(result["messages"][-1].content)
    print(f"\nTotal tool calls: {tool_call_count}")

    print("\n=== Root Comparison ===")
    print("Root ToolChainer: manual sequential dispatch with entropy re-scoring")
    print("LangGraph agent:  LLM decides when to chain, when to stop")


if __name__ == "__main__":
    main()
