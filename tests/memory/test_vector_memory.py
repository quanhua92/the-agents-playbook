"""Tests for memory/vector_memory.py — InMemoryVectorStore."""

from time import monotonic

import numpy as np
import pytest

from the_agents_playbook.memory import Fact
from the_agents_playbook.memory.protocol import EmbeddingProvider
from the_agents_playbook.memory.vector_memory import InMemoryVectorStore


class FixedEmbedder(EmbeddingProvider):
    """Returns a fixed embedding for predictable cosine similarity."""

    def __init__(self):
        self.call_count = 0

    async def embed(self, text: str) -> np.ndarray:
        self.call_count += 1
        # Each unique text gets a deterministic embedding
        vec = np.zeros(8)
        for i, ch in enumerate(text):
            vec[i % 8] += ord(ch)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


@pytest.fixture
def embedder() -> FixedEmbedder:
    return FixedEmbedder()


@pytest.fixture
def store(embedder: FixedEmbedder) -> InMemoryVectorStore:
    return InMemoryVectorStore(embedding_provider=embedder, decay_lambda=0.01)


async def test_store_and_size(store: InMemoryVectorStore):
    assert store.size == 0
    await store.store(Fact(content="hello", source="test"))
    assert store.size == 1


async def test_store_computes_embedding(store: InMemoryVectorStore, embedder: FixedEmbedder):
    fact = Fact(content="hello", source="test")
    assert fact.embedding is None
    await store.store(fact)
    assert fact.embedding is not None
    assert isinstance(fact.embedding, np.ndarray)
    assert embedder.call_count == 1


async def test_store_preserves_existing_embedding(embedder: FixedEmbedder):
    pre_embedded = np.array([1.0, 0.0, 0.0])
    fact = Fact(content="hello", source="test", embedding=pre_embedded)
    store = InMemoryVectorStore(embedding_provider=embedder)
    await store.store(fact)
    assert np.array_equal(fact.embedding, pre_embedded)
    assert embedder.call_count == 0  # should not call embedder


async def test_recall_returns_facts(store: InMemoryVectorStore):
    await store.store(Fact(content="Python programming", source="user"))
    await store.store(Fact(content="JavaScript web dev", source="user"))

    results = await store.recall("Python")
    assert len(results) >= 1
    assert "Python" in results[0].content


async def test_recall_respects_top_k(store: InMemoryVectorStore):
    for i in range(5):
        await store.store(Fact(content=f"topic {i}", source="test"))

    results = await store.recall("topic", top_k=2)
    assert len(results) <= 2


async def test_recall_empty_store(store: InMemoryVectorStore):
    results = await store.recall("anything")
    assert results == []


async def test_search_by_similarity(embedder: FixedEmbedder):
    store = InMemoryVectorStore(embedding_provider=embedder)
    await store.store(Fact(content="Python programming language", source="user"))

    # Query with exact same text should give high score
    scored = await store.search_by_similarity("Python programming language", top_k=5)
    assert len(scored) == 1
    fact, score = scored[0]
    assert score > 0.9  # near-identical embedding

    # Query with min_score filter
    scored = await store.search_by_similarity("completely different topic", min_score=0.99)
    assert len(scored) == 0


async def test_time_decay_ranks_fresh_higher(embedder: FixedEmbedder):
    store = InMemoryVectorStore(embedding_provider=embedder, decay_lambda=10.0)

    await store.store(Fact(content="old fact", source="old"))

    import asyncio
    await asyncio.sleep(0.05)  # 50ms — enough for decay_lambda=10 to matter

    await store.store(Fact(content="fresh fact", source="fresh"))

    results = await store.recall("fact")
    # Fresh should come first due to time decay
    assert results[0].source == "fresh"


async def test_consolidate_noop(store: InMemoryVectorStore):
    await store.consolidate()  # Should not raise


async def test_clear(store: InMemoryVectorStore):
    await store.store(Fact(content="a", source="test"))
    await store.store(Fact(content="b", source="test"))
    assert store.size == 2
    store.clear()
    assert store.size == 0


async def test_recall_with_zero_decay():
    """With decay_lambda=0, time has no effect on ranking."""
    embedder = FixedEmbedder()
    store = InMemoryVectorStore(embedding_provider=embedder, decay_lambda=0.0)

    await store.store(Fact(content="first", source="test"))
    await store.store(Fact(content="second", source="test"))

    results = await store.recall("first")
    assert len(results) >= 1
