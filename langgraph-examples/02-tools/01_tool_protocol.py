"""01_tool_protocol.py -- @tool decorator vs root's Tool ABC.

In the root project, every tool implements the Tool ABC with:
  - name property
  - description property
  - parameters property (JSON Schema dict)
  - async execute(**kwargs) -> ToolResult

In LangChain, there are three ways to define tools:
  1. @tool decorator (simplest -- schema inferred from signature)
  2. @tool with Pydantic args_schema (explicit schema)
  3. StructuredTool.from_function (closest to root's Tool ABC)
"""

from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field


# --- Approach 1: @tool decorator (simplest) ---
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    weather_data = {
        "New York": "22C, sunny",
        "London": "15C, cloudy",
        "Tokyo": "28C, rainy",
    }
    return weather_data.get(city, f"Weather data for '{city}' not available.")


# --- Approach 2: @tool with explicit Pydantic input schema ---
class SearchInput(BaseModel):
    query: str = Field(description="Search query string")
    limit: int = Field(default=5, ge=1, le=20, description="Max results to return")


@tool(args_schema=SearchInput)
def search(query: str, limit: int = 5) -> str:
    """Search for information on a topic."""
    return f"Results for '{query}' (showing up to {limit}): item1, item2, item3."


# --- Approach 3: StructuredTool (closest to root's Tool ABC) ---
def execute_analysis(text: str, depth: int = 1) -> dict:
    """Analyze text content at a given depth."""
    return {"analysis": text.upper(), "depth": depth, "length": len(text)}


analysis_tool = StructuredTool.from_function(
    func=execute_analysis,
    name="analyze_text",
    description="Analyze text content at a given depth.",
)


def main():
    tools = [get_weather, search, analysis_tool]

    print("=== Tool Protocol Comparison ===\n")

    for t in tools:
        print(f"Tool: {t.name}")
        print(f"  Description: {t.description}")
        schema = t.get_input_schema().model_json_schema()
        print(f"  Schema: {schema}")
        print()

    # Execute each tool
    print("=== Execution ===\n")

    result = get_weather.invoke({"city": "Tokyo"})
    print(f"get_weather(city='Tokyo'): {result}")

    result = search.invoke({"query": "python", "limit": 3})
    print(f"search(query='python', limit=3): {result}")

    result = analysis_tool.invoke({"text": "hello world", "depth": 1})
    print(f"analyze_text(text='hello world', depth=1): {result}")

    # Side-by-side with root's Tool ABC
    print("\n=== Root vs LangChain Comparison ===")
    print("Root Tool ABC:           LangChain @tool:")
    print("  .name property           t.name")
    print("  .description property    t.description")
    print("  .parameters (JSON)       t.get_input_schema().model_json_schema()")
    print("  .execute(**kw) -> ToolResult  t.invoke(args) -> str/dict")
    print("  class WeatherTool(Tool)  @tool def get_weather(city): ...")


if __name__ == "__main__":
    main()
