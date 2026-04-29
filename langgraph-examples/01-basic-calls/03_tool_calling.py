"""03_tool_calling.py -- @tool decorator + bind_tools for native tool calling.

Replaces the root project's manual ToolSpec definition + tool_choice in 04-tool-choice.py.
The @tool decorator generates the JSON Schema automatically from the function signature.
"""

from langchain_core.tools import tool

from shared import get_openai_llm


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    weather_data = {
        "New York": "22C, sunny",
        "London": "15C, cloudy",
        "Tokyo": "28C, rainy",
    }
    return weather_data.get(city, f"Weather data for '{city}' not available.")


@tool
def get_population(city: str) -> str:
    """Get the population of a given city."""
    population_data = {
        "New York": "8.3 million",
        "London": "9.0 million",
        "Tokyo": "14.0 million",
    }
    return population_data.get(city, f"Population data for '{city}' not available.")


def main():
    llm = get_openai_llm()
    llm_with_tools = llm.bind_tools([get_weather, get_population])

    # The model decides which tool to call based on the query
    queries = [
        "What's the weather in Tokyo?",
        "How many people live in London?",
        "What is the capital of France?",  # No tool needed
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        response = llm_with_tools.invoke(q)

        if response.tool_calls:
            for tc in response.tool_calls:
                print(f"  -> Tool: {tc['name']}({tc['args']})")
                # Execute the tool
                result = get_weather.invoke(tc['args']) if tc['name'] == 'get_weather' else get_population.invoke(tc['args'])
                print(f"     Result: {result}")
        else:
            print(f"  -> Direct: {response.content[:80]}...")

    # Show the auto-generated tool schema
    print("\n=== Auto-Generated Schema ===")
    print(f"get_weather schema: {get_weather.get_input_schema().model_json_schema()}")


if __name__ == "__main__":
    main()
