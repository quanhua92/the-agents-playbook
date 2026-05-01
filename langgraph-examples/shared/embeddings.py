"""Concrete embedding provider using the OpenAI embeddings API.

Adapted from the_agents_playbook/memory/embedding_provider.py.
Uses httpx directly (no langchain dependency) for embedding calls.
"""

import logging

import httpx
import numpy as np

from .settings import settings

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """Pluggable embedding provider -- converts text to numpy vectors."""

    async def embed(self, text: str) -> np.ndarray:
        """Convert text to a numpy vector."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Convert multiple texts to numpy vectors."""
        ...


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by the OpenAI /embeddings endpoint.

    Works with OpenAI directly or any compatible API (e.g., OpenRouter)
    by configuring embedding_base_url.

    Usage:
        embedder = OpenAIEmbeddingProvider()
        vec = await embedder.embed("Hello world")
        vecs = await embedder.embed_batch(["Hello", "World"])
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self._api_key = api_key or settings.embedding_api_key
        self._base_url = (base_url or settings.embedding_base_url).rstrip("/")
        self._model = model or settings.embedding_model
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None and not self._client.is_closed:
            return self._client
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(30.0, connect=5.0),
        )
        return self._client

    async def embed(self, text: str) -> np.ndarray:
        """Embed a single text string into a numpy vector."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts in a single API call."""
        if not texts:
            return []

        client = await self._get_client()
        response = await client.post(
            self._base_url + "/embeddings",
            json={
                "model": self._model,
                "input": texts,
            },
        )
        response.raise_for_status()
        data = response.json()

        sorted_items = sorted(data["data"], key=lambda x: x["index"])
        return [np.array(item["embedding"], dtype=np.float32) for item in sorted_items]

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
