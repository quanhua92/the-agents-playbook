"""03_bound_tools.py -- .bind_tools() attaches tools to the model.

In the root project:
  request = MessageRequest(
      model=settings.openai_model,
      messages=messages,
      tools=registry.get_specs(),
      tool_choice=ToolChoice(type="auto"),
  )
  response = await provider.send_message(request)

In LangChain:
  llm_with_tools = llm.bind_tools([calculate, lookup_fact])
  response = llm_with_tools.invoke(messages)

The model decides whether to use tools based on the query content.
"""

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from shared import get_openai_llm


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@tool
def lookup_fact(topic: str) -> str:
    """Look up a fact about a topic."""
    facts = {
        "pi": "3.14159...",
        "speed of light": "299,792,458 m/s",
        "earth circumference": "40,075 km",
    }
    return facts.get(topic.lower(), f"No fact found for '{topic}'")


def main():
    llm = get_openai_llm()
    llm_with_tools = llm.bind_tools([calculate, lookup_fact])

    queries = [
        ("What is 15 * 27?", "math -- should call calculate"),
        ("What is the speed of light?", "fact lookup -- should call lookup_fact"),
        ("What is the capital of France?", "general knowledge -- no tool needed"),
    ]

    print("=== Bound Tools: LLM decides when to use tools ===\n")

    for query, expectation in queries:
        print(f"Query: {query}")
        print(f"  Expected: {expectation}")

        response = llm_with_tools.invoke([HumanMessage(content=query)])

        if response.tool_calls:
            for tc in response.tool_calls:
                print(f"  -> Tool: {tc['name']}({tc['args']})")
                # Execute and show result
                tool_map = {"calculate": calculate, "lookup_fact": lookup_fact}
                if tc["name"] in tool_map:
                    result = tool_map[tc["name"]].invoke(tc["args"])
                    print(f"     Result: {result}")
        else:
            print(f"  -> Direct: {response.content[:100]}")
        print()


if __name__ == "__main__":
    main()
