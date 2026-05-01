"""Tests for tools/builtins/search.py — WebSearchTool."""

import httpx
import respx

from the_agents_playbook.tools.builtins.search import WebSearchTool


@respx.mock
async def test_search_returns_results():
    """Mock the DuckDuckGo API and verify the tool parses the response."""
    mock_response = {
        "Abstract": "Python is a programming language.",
        "AbstractSource": "Wikipedia",
        "AbstractURL": "https://en.wikipedia.org/wiki/Python",
        "Heading": "Python",
        "RelatedTopics": [
            {"Text": "Python 3.12 released", "FirstURL": "https://example.com/py312"},
            {"Text": "Python web frameworks", "FirstURL": "https://example.com/web"},
        ],
    }

    respx.get("https://api.duckduckgo.com").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    tool = WebSearchTool()
    result = await tool.execute(query="Python")

    assert result.error is False
    assert "Python is a programming language" in result.output
    assert "Related Topics" in result.output


@respx.mock
async def test_search_no_results():
    respx.get("https://api.duckduckgo.com").mock(
        return_value=httpx.Response(200, json={"Abstract": "", "RelatedTopics": []})
    )

    tool = WebSearchTool()
    result = await tool.execute(query="xyznonexistent123")

    assert result.error is False
    assert "No results found" in result.output


@respx.mock
async def test_search_max_results():
    mock_response = {
        "Abstract": "",
        "RelatedTopics": [
            {"Text": f"Result {i}", "FirstURL": f"https://example.com/{i}"}
            for i in range(10)
        ],
    }

    respx.get("https://api.duckduckgo.com").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    tool = WebSearchTool()
    result = await tool.execute(query="test", max_results=3)

    assert result.error is False
    # Should contain at most 3 results (plus the "Related Topics:" header)
    lines = result.output.split("\n")
    result_lines = [ln for ln in lines if ln.strip().startswith("- ")]
    assert len(result_lines) <= 3


@respx.mock
async def test_search_timeout():
    respx.get("https://api.duckduckgo.com").mock(side_effect=httpx.Response(408))

    tool = WebSearchTool(timeout_seconds=0.001)
    result = await tool.execute(query="test")

    assert result.error is True
    assert "timed out" in result.output.lower() or "failed" in result.output.lower()


def test_tool_name():
    tool = WebSearchTool()
    assert tool.name == "web_search"


def test_parameters_schema():
    tool = WebSearchTool()
    schema = tool.parameters
    assert schema["required"] == ["query"]
    assert "query" in schema["properties"]
    assert "max_results" in schema["properties"]
