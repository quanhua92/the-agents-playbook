"""Tests for memory/protocol.py — Fact, EmbeddingProvider, BaseMemoryProvider."""

import numpy as np
import pytest

from the_agents_playbook.memory.protocol import (
    BaseMemoryProvider,
    EmbeddingProvider,
    Fact,
)


class FakeEmbedder(EmbeddingProvider):
    async def embed(self, text: str) -> np.ndarray:
        return np.array([len(text) % 10, ord(text[0]) % 10], dtype=float)


class FakeMemory(BaseMemoryProvider):
    def __init__(self):
        self._facts: list[Fact] = []

    async def store(self, fact: Fact) -> None:
        self._facts.append(fact)

    async def recall(self, query: str, top_k: int = 5) -> list[Fact]:
        return self._facts[:top_k]

    async def consolidate(self) -> None:
        pass


def test_fact_defaults():
    fact = Fact(content="test", source="unit")
    assert fact.content == "test"
    assert fact.source == "unit"
    assert isinstance(fact.timestamp, float)
    assert fact.embedding is None
    assert fact.tags == []


def test_fact_with_all_fields():
    fact = Fact(
        content="test",
        source="unit",
        timestamp=123.0,
        embedding=np.array([1.0, 2.0]),
        tags=["a", "b"],
    )
    assert fact.timestamp == 123.0
    assert fact.tags == ["a", "b"]
    assert np.array_equal(fact.embedding, np.array([1.0, 2.0]))


def test_fact_is_dataclass():
    import dataclasses

    assert dataclasses.is_dataclass(Fact)


async def test_embedder_returns_ndarray():
    embedder = FakeEmbedder()
    result = await embedder.embed("hello")
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)


async def test_base_memory_store_and_recall():
    memory = FakeMemory()
    fact = Fact(content="stored fact", source="test")
    await memory.store(fact)
    results = await memory.recall("anything")
    assert len(results) == 1
    assert results[0].content == "stored fact"


def test_base_memory_requires_subclass():
    with pytest.raises(TypeError):
        BaseMemoryProvider()


def test_base_memory_consolidate_noop():
    # FakeMemory.consolidate exists and doesn't error
    import asyncio

    memory = FakeMemory()
    asyncio.run(memory.consolidate())
