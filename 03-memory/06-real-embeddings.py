"""06-real-embeddings.py — Vector search with real OpenAI embeddings via OpenRouter.

Calls the actual embedding API instead of using a mock hash-based embedder.
Requires EMBEDDING_API_KEY in .env.

Run:
    uv run python 03-memory/06-real-embeddings.py
"""

import asyncio


from the_agents_playbook.memory import (
    Fact,
    InMemoryVectorStore,
    OpenAIEmbeddingProvider,
)


async def main():
    embedder = OpenAIEmbeddingProvider()
    store = InMemoryVectorStore(embedding_provider=embedder, decay_lambda=0.001)

    # Store facts
    print("=== Storing Facts ===")
    facts = [
        Fact(content="User is a Python developer specializing in AI", source="user"),
        Fact(content="User works on autonomous agent systems", source="project"),
        Fact(content="User prefers functional programming over OOP", source="user"),
        Fact(
            content="The project uses httpx for async HTTP requests", source="project"
        ),
        Fact(
            content="User dislikes excessive logging in production code", source="user"
        ),
    ]
    for f in facts:
        await store.store(f)
        print(f"  Stored: {f.content}")

    print(f"\nStore size: {store.size}")

    # Semantic search — the embeddings capture actual meaning
    print("\n=== Search: 'programming languages' ===")
    results = await store.recall("programming languages")
    for f in results:
        print(f"  [{f.source}] {f.content}")

    print("\n=== Search: 'software engineering preferences' ===")
    results = await store.recall("software engineering preferences")
    for f in results:
        print(f"  [{f.source}] {f.content}")

    # Show similarity scores
    print("\n=== Scored Search: 'code architecture' (min_score=0.3) ===")
    scored = await store.search_by_similarity(
        "code architecture", top_k=3, min_score=0.3
    )
    for fact, score in scored:
        print(f"  score={score:.4f}  [{fact.source}] {fact.content}")

    # Compare semantic similarity vs a mock embedder
    print("\n=== Semantic vs Lexical ===")
    query = "AI systems"
    results = await store.recall(query)
    print(f"  Query: '{query}'")
    for f in results:
        print(f"  [{f.source}] {f.content}")
    print(
        "  (Real embeddings understand that 'AI systems' relates to 'autonomous agent systems')"
    )

    store.clear()
    print(f"\nCleared. Store size: {store.size}")

    await embedder.close()


asyncio.run(main())
