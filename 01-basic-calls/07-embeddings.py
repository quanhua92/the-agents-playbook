"""07-embeddings.py — Embed text using the OpenAI embeddings API via OpenRouter.

Run:
    uv run python 01-basic-calls/07-embeddings.py
"""

import asyncio
import numpy as np

from the_agents_playbook.memory import OpenAIEmbeddingProvider


async def run():
    embedder = OpenAIEmbeddingProvider()

    # Single text
    vec = await embedder.embed("The quick brown fox jumps over the lazy dog.")
    print(f"Single embed — dim={len(vec)}, norm={np.linalg.norm(vec):.4f}")
    print(f"  first 5 values: {vec[:5]}")

    # Batch
    texts = [
        "I love programming in Python.",
        "Dogs are loyal animals.",
        "The capital of France is Paris.",
    ]
    vecs = await embedder.embed_batch(texts)

    # Compute cosine similarity between pairs
    from the_agents_playbook.utils.vectors import cosine_similarity

    print(f"\nBatch embed — {len(vecs)} vectors, dim={len(vecs[0])}")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = cosine_similarity(vecs[i], vecs[j])
            print(f'  "{texts[i][:30]}..." vs "{texts[j][:30]}..." → {sim:.4f}')

    await embedder.close()


if __name__ == "__main__":
    asyncio.run(run())
