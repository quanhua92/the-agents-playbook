"""Unit tests for OpenAIEmbeddingProvider — mocked HTTP with respx."""

import json
from unittest.mock import patch

import httpx
import numpy as np
import pytest
import respx

from the_agents_playbook.memory.embedding_provider import OpenAIEmbeddingProvider


MOCK_EMBEDDING_RESPONSE = {
    "object": "list",
    "data": [
        {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
        {"object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 1},
    ],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 5, "total_tokens": 5},
}

SINGLE_EMBEDDING_RESPONSE = {
    "object": "list",
    "data": [
        {"object": "embedding", "embedding": [0.7, 0.8, 0.9], "index": 0},
    ],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 3, "total_tokens": 3},
}


@pytest.fixture(autouse=True)
def _patch_settings():
    with patch(
        "the_agents_playbook.memory.embedding_provider.settings"
    ) as mock_settings:
        mock_settings.embedding_api_key = "test-key"
        mock_settings.embedding_base_url = "https://openrouter.ai/api/v1"
        mock_settings.embedding_model = "openai/text-embedding-3-small"
        yield


class TestOpenAIEmbeddingProvider:
    def test_init_uses_settings_defaults(self):
        provider = OpenAIEmbeddingProvider()
        assert provider._base_url == "https://openrouter.ai/api/v1"
        assert provider._model == "openai/text-embedding-3-small"
        assert provider._api_key == "test-key"

    def test_init_uses_custom_values(self):
        provider = OpenAIEmbeddingProvider(
            api_key="my-key",
            base_url="https://my-api.com/v1",
            model="my-model",
        )
        assert provider._api_key == "my-key"
        assert provider._base_url == "https://my-api.com/v1"
        assert provider._model == "my-model"

    @respx.mock
    async def test_embed_single_text(self):
        respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(200, json=SINGLE_EMBEDDING_RESPONSE)
        )
        provider = OpenAIEmbeddingProvider()
        vec = await provider.embed("Hello")
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        assert list(vec) == [0.7, 0.8, 0.9]
        await provider.close()

    @respx.mock
    async def test_embed_batch(self):
        respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(200, json=MOCK_EMBEDDING_RESPONSE)
        )
        provider = OpenAIEmbeddingProvider()
        vecs = await provider.embed_batch(["Hello", "World"])
        assert len(vecs) == 2
        assert list(vecs[0]) == [0.1, 0.2, 0.3]
        assert list(vecs[1]) == [0.4, 0.5, 0.6]
        await provider.close()

    @respx.mock
    async def test_embed_batch_empty_returns_empty(self):
        provider = OpenAIEmbeddingProvider()
        vecs = await provider.embed_batch([])
        assert vecs == []
        await provider.close()

    @respx.mock
    async def test_embed_batch_preserves_order(self):
        """Even if API returns items out of order, result should be ordered by index."""
        out_of_order = {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [1.0, 0.0], "index": 2},
                {"object": "embedding", "embedding": [0.0, 1.0], "index": 0},
                {"object": "embedding", "embedding": [0.5, 0.5], "index": 1},
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 6, "total_tokens": 6},
        }
        respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(200, json=out_of_order)
        )
        provider = OpenAIEmbeddingProvider()
        vecs = await provider.embed_batch(["a", "b", "c"])
        assert list(vecs[0]) == [0.0, 1.0]
        assert list(vecs[1]) == [0.5, 0.5]
        assert list(vecs[2]) == [1.0, 0.0]
        await provider.close()

    @respx.mock
    async def test_embed_single_delegates_to_batch(self):
        respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(200, json=SINGLE_EMBEDDING_RESPONSE)
        )
        provider = OpenAIEmbeddingProvider()
        await provider.embed("Hello")
        call = respx.calls[-1]
        body = json.loads(call.request.content)
        assert body["input"] == ["Hello"]
        await provider.close()

    @respx.mock
    async def test_api_error_raises(self):
        respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(
                401, json={"error": {"message": "Invalid API key"}}
            )
        )
        provider = OpenAIEmbeddingProvider()
        with pytest.raises(httpx.HTTPStatusError):
            await provider.embed("test")
        await provider.close()

    async def test_close_idempotent(self):
        provider = OpenAIEmbeddingProvider()
        await provider.close()
        await provider.close()  # should not raise
