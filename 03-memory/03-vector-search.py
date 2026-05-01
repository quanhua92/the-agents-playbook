"""03-vector-search.py — Cosine similarity search with exponential time decay.

Uses InMemoryVectorStore with a deterministic embedding provider for demo.
In production, replace MockEmbeddingProvider with one that calls an
embedding API (e.g., OpenAI text-embedding-3-small).
"""

import asyncio

import numpy as np

from the_agents_playbook.memory import Fact
from the_agents_playbook.memory.protocol import EmbeddingProvider
from the_agents_playbook.memory.vector_memory import InMemoryVectorStore


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding for demo — hashes text into a fixed-dim vector.

    In production, use an LLM embedding API endpoint instead.
    """

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim
        self._cache: dict[str, np.ndarray] = {}

    async def embed(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]

        # Deterministic pseudo-embedding from character hashes
        vec = np.zeros(self._dim)
        for i, ch in enumerate(text):
            vec[i % self._dim] += ord(ch) / 127.0

        # Normalize so cosine similarity is meaningful
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        self._cache[text] = vec
        return vec


async def main():
    embedder = MockEmbeddingProvider()
    store = InMemoryVectorStore(embedding_provider=embedder, decay_lambda=0.001)

    # Store facts
    print("=== Storing Facts ===")
    facts = [
        Fact(content="User is a Python developer", source="user"),
        Fact(content="User works on AI agent systems", source="project"),
        Fact(content="User prefers functional programming patterns", source="user"),
        Fact(content="The project uses httpx for HTTP requests", source="project"),
        Fact(content="User dislikes excessive logging", source="user"),
    ]
    for f in facts:
        await store.store(f)
        print(f"  Stored: {f.content}")

    print(f"\nStore size: {store.size}")

    # Search by similarity
    print("\n=== Vector Search: 'Python programming' ===")
    results = await store.recall("Python programming")
    for f in results:
        print(f"  [{f.source}] {f.content}")

    print("\n=== Vector Search: 'AI agents' ===")
    results = await store.recall("AI agents")
    for f in results:
        print(f"  [{f.source}] {f.content}")

    # Show scores
    print("\n=== Scored Search: 'coding preferences' (min_score=0.5) ===")
    scored = await store.search_by_similarity(
        "coding preferences", top_k=3, min_score=0.5
    )
    for fact, score in scored:
        print(f"  score={score:.4f}  [{fact.source}] {fact.content}")

    # Time decay demo
    print("\n=== Time Decay Demo ===")
    import asyncio as aio

    print("Storing 'will be old soon' fact...")
    old_fact = Fact(content="This fact is getting old", source="time-demo")
    await store.store(old_fact)

    print("Waiting 1 second for time decay...")
    await aio.sleep(1.0)

    # New fact (fresh timestamp)
    await store.store(Fact(content="This fact is fresh", source="time-demo"))

    results = await store.recall("fact")
    print("Recall 'fact' (fresh should rank higher):")
    for f in results:
        print(f"  [{f.source}] {f.content}")

    store.clear()
    print(f"\nCleared. Store size: {store.size}")


asyncio.run(main())
