# Dual-File Memory and Consolidation

## Problem

Agents need persistent memory across sessions, but raw conversation logs are noisy. A 50-message conversation about fixing a bug contains maybe 3 durable facts (user prefers tabs, project uses PostgreSQL, deadline is Friday). The rest is greeting, acknowledgment, and debugging noise. Storing raw logs means every recall returns irrelevant context. You need structured memory that survives restarts.

## Solution

Two-file architecture: `MEMORY.md` for structured facts, `HISTORY.md` for raw log. Use LLM consolidation to extract facts from the noisy log into the structured store.

### Fact Data Model

`Fact` (`src/the_agents_playbook/memory/protocol.py:11`) is the atomic unit of memory:

```python
content: str     # The fact text
source: str      # Where it came from ("user", "system", "consolidation")
tags: list[str]  # Optional categorization
embedding: np.ndarray | None  # For vector search (optional)
timestamp: float  # When it was stored
```

### DualFileMemory

`DualFileMemory` (`src/the_agents_playbook/memory/file_memory.py:75`) stores facts as markdown:

- `MEMORY.md`: structured facts separated by `---`, each with `content:`, `source:`, `tags:` fields
- `HISTORY.md`: append-only event log, one line per event

Key behaviors:
- `store(fact)` (`file_memory.py:112`) — appends to HISTORY.md and updates MEMORY.md (deduplicates by content)
- `store_event(text, source)` (`file_memory.py:131`) — appends to HISTORY.md only (no fact extraction)
- `read_facts()` (`file_memory.py:136`) — sync parse of MEMORY.md into `list[Fact]`
- `recall(query, top_k)` (`file_memory.py:147`) — substring matching across stored facts

Serialization uses `_serialize_facts()` (`file_memory.py:59`) and parsing uses `_parse_facts()` (`file_memory.py:20`).

### LLMConsolidation

`LLMConsolidator` (`src/the_agents_playbook/memory/consolidation.py:28`) extracts structured facts from raw conversation history:

1. Read `HISTORY.md` (truncate to last 200 lines if needed)
2. Send to LLM with a consolidation prompt that requests JSON array of `{content, source}` objects
3. Parse the LLM response (handles both direct array and `{"facts": [...]}` formats)
4. Store new facts in MEMORY.md (DualFileMemory deduplicates by content)

The consolidation prompt (`consolidation.py:12`) instructs the LLM to extract only factual, durable information and ignore transient conversational noise.

**Key distinction:** Compaction = summarize and discard (loses intent). Consolidation = extract and index (preserves facts in searchable form).

## Code Reference

- `src/the_agents_playbook/memory/protocol.py` — `Fact` (line 11), `BaseMemoryProvider` (line 31)
- `src/the_agents_playbook/memory/file_memory.py` — `DualFileMemory` (line 75)
- `src/the_agents_playbook/memory/consolidation.py` — `LLMConsolidator` (line 28)

## Playground Examples

- `03-memory/01-fact-storage.py` — basic fact storage and recall
- `03-memory/02-dual-file-memory.py` — MEMORY.md + HISTORY.md architecture
- `03-memory/04-consolidation.py` — LLM-powered fact extraction from conversation history
