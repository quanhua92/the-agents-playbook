# Segmented Memory with Tiered Decay

## Problem

The original `Fact` class treats all memories identically. A user's name and a throwaway comment like "I'm at a coffee shop" have the same weight, same retention, and same recall priority. Over time this means either permanent facts get pushed out by ephemeral noise, or ephemeral noise clutters recall forever.

## Solution

Three new classes introduce segment classification, tiered decay, and lifecycle management:

| Class | File | Role |
|---|---|---|
| `MemorySegment` | `memory/segments.py` | Enum of 7 categories |
| `MemoryRecord` | `memory/record.py` | Extended `Fact` with segment, tier, importance, lifecycle |
| `MemoryDecay` | `memory/decay.py` | Scoring formula + prune/archive logic |

### Segments

Each memory is classified into one of seven segments, each with different retention behavior:

```
IDENTITY     → permanent, importance=1.0, decay=0     (name, email, phone)
PREFERENCE   → long-term,  importance=0.8, decay=0.001 (likes, format prefs)
CORRECTION   → permanent, importance=0.9, decay=0     (user corrections)
RELATIONSHIP → long-term,  importance=0.7, decay=0.002 (work role, family)
PROJECT      → medium,     importance=0.6, decay=0.01  (active project info)
KNOWLEDGE    → medium,     importance=0.5, decay=0.008 (facts from tool results)
CONTEXT      → short-term, importance=0.3, decay=0.05  (temporary situation)
```

### The Scoring Formula

```
score = importance × exp(-λ × days) × (1 + ln(1 + access_count) × 0.1)
```

Three factors determine how relevant a memory is right now:

1. **importance** — base weight from segment defaults (identity = 1.0, context = 0.3)
2. **exponential decay** — `exp(-λ × days)` fades the score over time. For permanent memories, λ=0 so the factor is always 1.0. For context, λ=0.05 so the score halves roughly every 14 days.
3. **access boost** — frequently recalled memories get a small multiplier. `record_access()` increments the counter each time a memory is returned by recall.

### Lifecycle

Records transition through three states:

```
ACTIVE → ARCHIVED → PRUNED
```

- **ACTIVE** — normal, participates in recall and decay
- **ARCHIVED** — manually frozen or auto-archived when score drops below 0.05. No longer decays but still exists.
- **PRUNED** — score dropped below 0.01. Ready for deletion from storage.

Permanent memories (IDENTITY, CORRECTION) never transition — they're excluded from both archiving and pruning.

### How it integrates

`BaseMemoryProvider` in `memory/protocol.py` gains three new methods:

```python
async def store_record(self, record: MemoryRecord) -> None
async def recall_by_segment(self, segment: MemorySegment, top_k: int) -> list[MemoryRecord]
async def archive(self, memory_id: str) -> None
```

All three have default implementations so existing memory providers (file, vector, etc.) continue working without changes. Subclasses override to use the full record metadata.

### Creating a record

```python
from the_agents_playbook.memory import MemoryRecord, MemorySegment

# Tier, importance, and decay_rate inherit from segment defaults automatically
record = MemoryRecord(
    content="User's name is Alice",
    source="user",
    segment=MemorySegment.IDENTITY,
)
# record.tier == MemoryTier.PERMANENT
# record.importance == 1.0
# record.decay_rate == 0.0
```

### Running decay

```python
from the_agents_playbook.memory import MemoryDecay

decay = MemoryDecay()

# Score a record 30 days after creation
score = decay.score(record, days=30)

# Prune all records below threshold (returns newly pruned list)
pruned = decay.prune(records)

# Two-phase: archive stale, then prune very stale
archived, pruned = decay.decay_and_archive(records)
```

## Code Reference

- `src/the_agents_playbook/memory/segments.py` — `MemorySegment` enum, `MemoryTier` enum, `SegmentConfig`, `SEGMENT_DEFAULTS`
- `src/the_agents_playbook/memory/record.py` — `MemoryRecord` extending `Fact` with segment, tier, importance, lifecycle
- `src/the_agents_playbook/memory/decay.py` — `MemoryDecay` with `score()`, `prune()`, `decay_and_archive()`
- `src/the_agents_playbook/memory/protocol.py` — `BaseMemoryProvider` extended with `store_record()`, `recall_by_segment()`, `archive()`

## Playground Example

- `03-memory/07-segmented-decay.py` — store facts with different segments, simulate time passing, run decay

## LangGraph Example

- `langgraph-examples/03-memory/04_segmented_memory.py` — segmented recall in a LangGraph agent
