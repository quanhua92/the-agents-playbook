"""01-tool-protocol.py — Define a custom Tool and execute it by hand.

This example shows the Tool contract: every tool has a name, description,
JSON Schema parameters, and an async execute() method that returns ToolResult.
"""

import asyncio
from typing import Any

from the_agents_playbook.tools import Tool, ToolResult


class WeatherTool(Tool):
    """A simple tool that returns hardcoded weather data.

    In a real agent, this would call a weather API. Here we hardcode
    the response to focus on the Tool protocol itself.
    """

    @property
    def name(self) -> str:
        return "get_weather"

    @property
    def description(self) -> str:
        return "Get the current weather for a given city."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name",
                },
            },
            "required": ["city"],
            "additionalProperties": False,
        }

    async def execute(self, city: str, **kwargs: Any) -> ToolResult:
        # In a real tool, this would call an API. Hardcoded for demonstration.
        weather_data = {
            "New York": {"temp": 22, "condition": "sunny"},
            "London": {"temp": 15, "condition": "cloudy"},
            "Tokyo": {"temp": 28, "condition": "rainy"},
        }

        if city in weather_data:
            data = weather_data[city]
            return ToolResult(
                output=f"Weather in {city}: {data['temp']}C, {data['condition']}"
            )
        else:
            return ToolResult(
                output=f"City '{city}' not found. Available: {', '.join(weather_data.keys())}",
                error=True,
            )


async def main():
    tool = WeatherTool()

    # The Tool protocol in action
    print(f"Tool name:        {tool.name}")
    print(f"Tool description: {tool.description}")
    print(f"Parameters:       {tool.parameters}")
    print()

    # Execute with valid input
    result = await tool.execute(city="Tokyo")
    print(f"Tokyo:   {result.output} (error={result.error})")

    # Execute with unknown city
    result = await tool.execute(city="Paris")
    print(f"Paris:   {result.output} (error={result.error})")


asyncio.run(main())
