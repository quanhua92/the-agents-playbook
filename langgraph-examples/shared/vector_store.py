"""In-memory vector store with cosine similarity and time decay.

Adapted from the_agents_playbook/memory/vector_memory.py.
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

import numpy as np

from .embeddings import EmbeddingProvider
from .vectors import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class Fact:
    """A single piece of knowledge extracted from conversation or tool results."""

    content: str
    source: str
    timestamp: float = field(default_factory=monotonic)
    embedding: np.ndarray | None = None
    tags: list[str] = field(default_factory=list)


class BaseMemoryProvider(ABC):
    """Abstract interface for memory storage and retrieval."""

    @abstractmethod
    async def store(self, fact: Fact) -> None:
        ...

    @abstractmethod
    async def recall(self, query: str, top_k: int = 5) -> list[Fact]:
        ...

    @abstractmethod
    async def consolidate(self) -> None:
        ...


class InMemoryVectorStore(BaseMemoryProvider):
    """In-memory vector store for semantic fact retrieval.

    Stores Fact objects with numpy embeddings. Recall uses cosine similarity
    with exponential time decay so recent facts rank higher.

    Usage:
        store = InMemoryVectorStore(embedding_provider=my_embedder, decay_lambda=0.01)
        await store.store(Fact(content="User prefers Python", source="user"))
        facts = await store.recall("programming languages")
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        decay_lambda: float = 0.01,
    ) -> None:
        self._embedder = embedding_provider
        self._decay_lambda = decay_lambda
        self._facts: list[Fact] = []

    @property
    def size(self) -> int:
        return len(self._facts)

    async def store(self, fact: Fact) -> None:
        """Store a fact, computing its embedding if not already present."""
        if fact.embedding is None:
            fact.embedding = await self._embedder.embed(fact.content)
        self._facts.append(fact)
        logger.debug("Stored fact (%d total): %s", len(self._facts), fact.content[:50])

    async def recall(self, query: str, top_k: int = 5) -> list[Fact]:
        """Recall the most relevant facts using cosine similarity + time decay."""
        if not self._facts:
            return []

        query_embedding = await self._embedder.embed(query)
        now = monotonic()

        scored: list[tuple[float, Fact]] = []
        for fact in self._facts:
            if fact.embedding is None:
                continue

            sim = cosine_similarity(query_embedding, fact.embedding)
            age = now - fact.timestamp
            decay = math.exp(-self._decay_lambda * age)
            score = sim * decay
            scored.append((score, fact))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in scored[:top_k]]

    async def search_by_similarity(
        self, query: str, top_k: int = 5, min_score: float = 0.0
    ) -> list[tuple[Fact, float]]:
        """Recall facts with their scores, filtering by minimum score threshold."""
        if not self._facts:
            return []

        query_embedding = await self._embedder.embed(query)

        scored: list[tuple[float, Fact]] = []
        for fact in self._facts:
            if fact.embedding is None:
                continue

            sim = cosine_similarity(query_embedding, fact.embedding)
            if sim >= min_score:
                scored.append((sim, fact))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [(fact, score) for score, fact in scored[:top_k]]

    async def consolidate(self) -> None:
        pass

    def clear(self) -> None:
        self._facts.clear()
