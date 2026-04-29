"""WebSearchTool — httpx-based web search."""

import logging
from typing import Any

import httpx

from ..protocol import Tool, ToolResult

logger = logging.getLogger(__name__)


class WebSearchTool(Tool):
    """Perform a web search using an HTTP API.

    By default, uses DuckDuckGo's instant answer API (no API key required).
    Override the base_url for other search providers.
    """

    def __init__(
        self,
        base_url: str = "https://api.duckduckgo.com",
        timeout_seconds: float = 10.0,
    ) -> None:
        self._base_url = base_url
        self._timeout = timeout_seconds

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for information. Returns search results as text."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5)",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    async def execute(self, query: str, max_results: int = 5, **kwargs: Any) -> ToolResult:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    self._base_url,
                    params={
                        "q": query,
                        "format": "json",
                        "no_html": 1,
                        "skip_disambig": 1,
                    },
                )
                response.raise_for_status()
                data = response.json()

            # DuckDuckGo returns an Abstract (summary), RelatedTopics, and Results
            parts = []

            abstract = data.get("Abstract", "")
            if abstract:
                parts.append(f"Summary: {abstract}")
                source = data.get("AbstractSource", "")
                if source:
                    parts.append(f"Source: {source}")
                url = data.get("AbstractURL", "")
                if url:
                    parts.append(f"URL: {url}")

            heading = data.get("Heading", "")
            if heading and abstract:
                parts.insert(0, f"Topic: {heading}")

            # RelatedTopics can contain sub-topics
            topics = data.get("RelatedTopics", [])
            if topics:
                parts.append("\nRelated Topics:")
                count = 0
                for topic in topics:
                    if count >= max_results:
                        break
                    if isinstance(topic, dict):
                        text = topic.get("Text", "")
                        url = topic.get("FirstURL", "")
                        if text:
                            parts.append(f"  - {text}")
                            if url:
                                parts.append(f"    {url}")
                            count += 1

            if not parts:
                return ToolResult(
                    output=f"No results found for: {query}",
                    error=False,
                )

            return ToolResult(output="\n".join(parts))

        except httpx.TimeoutException:
            return ToolResult(
                output=f"Search timed out after {self._timeout}s",
                error=True,
            )
        except Exception as e:
            return ToolResult(
                output=f"Search failed: {e}",
                error=True,
            )
