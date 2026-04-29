"""02_tool_node.py -- Tool dispatch replaces ToolRegistry.dispatch().

In the root project:
  registry = ToolRegistry()
  registry.register(WeatherTool())
  result = await registry.dispatch("get_weather", {"city": "Tokyo"})

In LangGraph:
  # Direct dispatch (what ToolNode does internally):
  result = get_weather.invoke({"city": "Tokyo"})

  # Within a graph (ToolNode handles routing automatically):
  agent = create_react_agent(llm, [get_weather, get_population])

ToolNode automatically:
  - Routes each tool_call to the matching @tool
  - Dispatches multiple tool calls
  - Returns ToolMessage objects for each result
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from shared import get_openai_llm


@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    data = {"Tokyo": "28C, rainy", "London": "15C, cloudy", "New York": "22C, sunny"}
    return data.get(city, f"Unknown city: {city}")


@tool
def get_population(city: str) -> str:
    """Get population of a city."""
    data = {"Tokyo": "14 million", "London": "9 million", "New York": "8 million"}
    return data.get(city, f"Unknown city: {city}")


def dispatch_tools(tools: dict, ai_message: AIMessage) -> list[ToolMessage]:
    """Manual dispatch -- mimics what ToolNode does internally.

    ToolNode in a compiled graph does this automatically:
      1. Read tool_calls from AIMessage
      2. Find matching @tool by name
      3. Execute each tool with its args
      4. Return ToolMessage list
    """
    messages: list[ToolMessage] = []
    for tc in ai_message.tool_calls:
        tool_name = tc["name"]
        tool_fn = tools.get(tool_name)
        if tool_fn is None:
            messages.append(ToolMessage(
                content=f"Error: unknown tool '{tool_name}'",
                tool_call_id=tc["id"],
                name=tool_name,
            ))
            continue

        result = tool_fn.invoke(tc["args"])
        messages.append(ToolMessage(
            content=result,
            tool_call_id=tc["id"],
            name=tool_name,
        ))
    return messages


def main():
    tool_map = {"get_weather": get_weather, "get_population": get_population}

    # --- Mock dispatch: simulate an AIMessage with tool calls ---
    print("=== Mock Dispatch ===\n")

    mock_ai_message = AIMessage(
        content="",
        tool_calls=[
            {"id": "call_1", "name": "get_weather", "args": {"city": "Tokyo"}},
            {"id": "call_2", "name": "get_population", "args": {"city": "London"}},
        ],
    )

    results = dispatch_tools(tool_map, mock_ai_message)

    for msg in results:
        print(f"  Tool '{msg.name}' (id={msg.tool_call_id}): {msg.content}")

    # --- Real LLM dispatch ---
    print("\n=== Real LLM Dispatch ===\n")

    llm = get_openai_llm().bind_tools([get_weather, get_population])
    response = llm.invoke("What's the weather and population of Tokyo?")

    if response.tool_calls:
        print(f"LLM requested {len(response.tool_calls)} tool call(s):")
        for tc in response.tool_calls:
            print(f"  {tc['name']}({tc['args']})")

        tool_results = dispatch_tools(tool_map, response)
        print("\nResults:")
        for msg in tool_results:
            print(f"  {msg.name}: {msg.content}")
    else:
        print(f"LLM responded directly: {response.content}")

    print("\n=== Root Comparison ===")
    print("Root: registry.register(tool) -> registry.dispatch(name, args)")
    print("Here: tool.invoke(args) or ToolNode within compiled graph")
    print("Both: name-based routing + parallel dispatch")


if __name__ == "__main__":
    main()
