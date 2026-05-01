"""InMemoryVectorStore — numpy-based vector similarity search with time decay."""

import logging
import math
from time import monotonic


from ..utils.vectors import cosine_similarity
from .protocol import BaseMemoryProvider, EmbeddingProvider, Fact

logger = logging.getLogger(__name__)


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
        """Recall the most relevant facts using cosine similarity + time decay.

        Score = cosine_similarity * exp(-lambda * age_in_seconds)
        """
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
        """No-op for vector store. Consolidation logic is external."""
        pass

    def clear(self) -> None:
        """Remove all stored facts."""
        self._facts.clear()
