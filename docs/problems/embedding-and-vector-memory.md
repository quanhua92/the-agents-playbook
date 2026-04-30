# Embedding and Vector Memory

## Problem

Keyword search misses semantic matches. Searching for "automobile" won't find a memory about "car". Searching for "sick day" won't match "PTO request". As memory grows, substring matching becomes increasingly unreliable. You need to search by meaning, not by characters.

## Solution

Convert text to high-dimensional vectors (embeddings), store them alongside facts, and retrieve by cosine similarity. Add exponential time decay so recent facts rank higher than stale ones.

### EmbeddingProvider ABC

`EmbeddingProvider` (`src/the_agents_playbook/memory/protocol.py:22`) defines the interface:

- `embed(text) -> np.ndarray` — single text to vector
- `embed_batch(texts) -> list[np.ndarray]` — batch embedding (more efficient)

`OpenAIEmbeddingProvider` (`src/the_agents_playbook/memory/embedding_provider.py:15`) implements this using the OpenAI `/embeddings` endpoint. Supports any OpenAI-compatible API by configuring `embedding_base_url`. Results are sorted by index to ensure correct order (`embedding_provider.py:76`).

### InMemoryVectorStore

`InMemoryVectorStore` (`src/the_agents_playbook/memory/vector_memory.py:16`) implements `BaseMemoryProvider` with semantic recall:

- `store(fact)` — computes embedding if missing, appends to internal list
- `recall(query, top_k)` — embeds query, scores all facts, returns top-k

Scoring formula (`vector_memory.py:66`):

```
score = cosine_similarity(query_vec, fact_vec) * exp(-decay_lambda * age_seconds)
```

`search_by_similarity(query, top_k, min_score)` (`vector_memory.py:73`) returns facts with their scores, filtering by a minimum similarity threshold.

Cosine similarity and normalization utilities live in `src/the_agents_playbook/utils/vectors.py`.

### Time Decay

The `decay_lambda` parameter (default 0.01) controls how quickly older facts lose relevance. A fact stored 100 seconds ago gets multiplied by `exp(-0.01 * 100) = exp(-1) ≈ 0.37` — about 63% decay. This prevents stale facts from dominating recall.

## Code Reference

- `src/the_agents_playbook/memory/protocol.py` — `EmbeddingProvider` ABC (line 22), `Fact` dataclass (line 11)
- `src/the_agents_playbook/memory/embedding_provider.py` — `OpenAIEmbeddingProvider` (line 15)
- `src/the_agents_playbook/memory/vector_memory.py` — `InMemoryVectorStore` (line 16)
- `src/the_agents_playbook/utils/vectors.py` — `cosine_similarity()`, `normalize()`

## Playground Examples

- `01-basic-calls/07-embeddings.py` — embedding text to vectors, similarity comparison
- `03-memory/03-vector-search.py` — semantic memory recall with cosine similarity
- `03-memory/06-real-embeddings.py` — real API-based embeddings with OpenAI

## LangGraph Examples

- `langgraph-examples/shared/embeddings.py` — shared embedding utilities
- `langgraph-examples/shared/vector_store.py` — shared vector store
- `langgraph-examples/shared/vectors.py` — vector math utilities
