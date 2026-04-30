"""Tests for segmented memory with tiered decay."""

import math
import time

import pytest

from the_agents_playbook.memory.decay import MemoryDecay, PRUNE_THRESHOLD
from the_agents_playbook.memory.record import MemoryLifecycle, MemoryRecord
from the_agents_playbook.memory.segments import (
    MemorySegment,
    MemoryTier,
    SEGMENT_DEFAULTS,
    SegmentConfig,
)


# ---------------------------------------------------------------------------
# Segment defaults
# ---------------------------------------------------------------------------

class TestSegmentDefaults:
    def test_all_segments_have_defaults(self):
        for segment in MemorySegment:
            assert segment in SEGMENT_DEFAULTS

    def test_permanent_segments_have_zero_decay(self):
        for segment, config in SEGMENT_DEFAULTS.items():
            if config.tier == MemoryTier.PERMANENT:
                assert config.decay_rate == 0.0

    def test_identity_is_permanent(self):
        cfg = SEGMENT_DEFAULTS[MemorySegment.IDENTITY]
        assert cfg.tier == MemoryTier.PERMANENT
        assert cfg.importance == 1.0

    def test_context_is_short_term(self):
        cfg = SEGMENT_DEFAULTS[MemorySegment.CONTEXT]
        assert cfg.tier == MemoryTier.SHORT_TERM
        assert cfg.decay_rate > 0.01

    def test_segment_config_is_frozen(self):
        cfg = SEGMENT_DEFAULTS[MemorySegment.IDENTITY]
        with pytest.raises(AttributeError):
            cfg.decay_rate = 999


# ---------------------------------------------------------------------------
# MemoryRecord
# ---------------------------------------------------------------------------

class TestMemoryRecord:
    def test_default_segment_is_knowledge(self):
        record = MemoryRecord(content="test", source="user")
        assert record.segment == MemorySegment.KNOWLEDGE

    def test_inherits_defaults_from_segment(self):
        record = MemoryRecord(
            content="Alice", source="user",
            segment=MemorySegment.IDENTITY,
        )
        assert record.tier == MemoryTier.PERMANENT
        assert record.importance == 1.0
        assert record.decay_rate == 0.0

    def test_explicit_overrides_take_precedence(self):
        record = MemoryRecord(
            content="test", source="user",
            segment=MemorySegment.CONTEXT,
            importance=0.99,
        )
        assert record.importance == 0.99
        # decay_rate still comes from segment default
        assert record.decay_rate == SEGMENT_DEFAULTS[MemorySegment.CONTEXT].decay_rate

    def test_is_permanent(self):
        record = MemoryRecord(
            content="test", source="user",
            segment=MemorySegment.IDENTITY,
        )
        assert record.is_permanent

    def test_record_access_updates_count(self):
        record = MemoryRecord(content="test", source="user")
        assert record.access_count == 0
        assert record.last_accessed_at is None

        record.record_access()
        assert record.access_count == 1
        assert record.last_accessed_at is not None

        record.record_access()
        assert record.access_count == 2

    def test_lifecycle_defaults_to_active(self):
        record = MemoryRecord(content="test", source="user")
        assert record.lifecycle == MemoryLifecycle.ACTIVE

    def test_supersedes_field(self):
        record = MemoryRecord(
            content="corrected", source="user",
            segment=MemorySegment.CORRECTION,
            supersedes="old-fact-id",
        )
        assert record.supersedes == "old-fact-id"


# ---------------------------------------------------------------------------
# MemoryDecay scoring
# ---------------------------------------------------------------------------

class TestMemoryDecayScore:
    def test_permanent_record_never_decays(self):
        record = MemoryRecord(
            content="name", source="user",
            segment=MemorySegment.IDENTITY,
        )
        decay = MemoryDecay()
        assert decay.score(record, 0) == pytest.approx(1.0)
        assert decay.score(record, 365) == pytest.approx(1.0)
        assert decay.score(record, 36500) == pytest.approx(1.0)

    def test_context_decays_rapidly(self):
        record = MemoryRecord(
            content="at coffee shop", source="user",
            segment=MemorySegment.CONTEXT,
        )
        decay = MemoryDecay()
        score_0 = decay.score(record, 0)
        score_30 = decay.score(record, 30)
        assert score_30 < score_0 * 0.5

    def test_access_boost_increases_score(self):
        record = MemoryRecord(
            content="JWT tokens", source="tool",
            segment=MemorySegment.KNOWLEDGE,
        )
        decay = MemoryDecay()
        score_before = decay.score(record, 30)

        record.record_access()
        record.record_access()
        record.record_access()

        score_after = decay.score(record, 30)
        assert score_after > score_before

    def test_score_formula_components(self):
        """Verify score = importance * exp(-lambda * days) * access_boost."""
        record = MemoryRecord(
            content="test", source="user",
            segment=MemorySegment.PROJECT,
        )
        decay = MemoryDecay()
        days = 10.0

        expected = (
            record.importance
            * math.exp(-record.decay_rate * days)
            * (1 + math.log(1 + record.access_count) * 0.1)
        )
        assert decay.score(record, days) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# MemoryDecay pruning
# ---------------------------------------------------------------------------

class TestMemoryDecayPrune:
    def test_permanent_records_never_pruned(self):
        record = MemoryRecord(
            content="name", source="user",
            segment=MemorySegment.IDENTITY,
        )
        # Backdate far enough that it would normally be pruned
        record.timestamp = time.monotonic() - 36500 * 86400
        decay = MemoryDecay()
        pruned = decay.prune([record])
        assert len(pruned) == 0
        assert record.lifecycle == MemoryLifecycle.ACTIVE

    def test_stale_context_gets_pruned(self):
        record = MemoryRecord(
            content="at coffee shop", source="user",
            segment=MemorySegment.CONTEXT,
        )
        record.timestamp = time.monotonic() - 365 * 86400  # 1 year old
        decay = MemoryDecay()
        pruned = decay.prune([record])
        assert len(pruned) == 1
        assert record.lifecycle == MemoryLifecycle.PRUNED

    def test_archived_records_not_considered(self):
        record = MemoryRecord(
            content="old project", source="user",
            segment=MemorySegment.PROJECT,
        )
        record.lifecycle = MemoryLifecycle.ARCHIVED
        record.timestamp = time.monotonic() - 365 * 86400
        decay = MemoryDecay()
        pruned = decay.prune([record])
        assert len(pruned) == 0

    def test_fresh_records_not_pruned(self):
        record = MemoryRecord(
            content="at coffee shop", source="user",
            segment=MemorySegment.CONTEXT,
        )
        # Just created — score should be well above threshold
        decay = MemoryDecay()
        pruned = decay.prune([record])
        assert len(pruned) == 0


# ---------------------------------------------------------------------------
# MemoryDecay decay_and_archive
# ---------------------------------------------------------------------------

class TestMemoryDecayArchive:
    def test_old_records_get_archived(self):
        record = MemoryRecord(
            content="old context", source="user",
            segment=MemorySegment.CONTEXT,
        )
        record.timestamp = time.monotonic() - 365 * 86400
        decay = MemoryDecay()
        archived, pruned = decay.decay_and_archive([record])
        assert len(archived) == 1
        assert record.lifecycle == MemoryLifecycle.ARCHIVED

    def test_permanent_records_never_archived(self):
        record = MemoryRecord(
            content="name", source="user",
            segment=MemorySegment.IDENTITY,
        )
        record.timestamp = time.monotonic() - 36500 * 86400
        decay = MemoryDecay()
        archived, pruned = decay.decay_and_archive([record])
        assert len(archived) == 0
        assert record.lifecycle == MemoryLifecycle.ACTIVE


# ---------------------------------------------------------------------------
# BaseMemoryProvider segment extensions
# ---------------------------------------------------------------------------

class TestBaseMemoryProviderExtensions:
    def test_store_record_default_fallback(self):
        from the_agents_playbook.memory.protocol import BaseMemoryProvider, Fact

        class FakeMemory(BaseMemoryProvider):
            def __init__(self):
                self.stored: list[Fact] = []

            async def store(self, fact: Fact) -> None:
                self.stored.append(fact)

            async def recall(self, query: str, top_k: int = 5) -> list[Fact]:
                return self.stored

            async def consolidate(self) -> None:
                pass

        import asyncio
        memory = FakeMemory()
        record = MemoryRecord(
            content="test fact", source="user",
            segment=MemorySegment.PREFERENCE,
        )

        asyncio.run(memory.store_record(record))
        assert len(memory.stored) == 1
        assert memory.stored[0].content == "test fact"

    def test_recall_by_segment_default_empty(self):
        from the_agents_playbook.memory.protocol import BaseMemoryProvider, Fact

        class FakeMemory(BaseMemoryProvider):
            async def store(self, fact: Fact) -> None:
                pass
            async def recall(self, query: str, top_k: int = 5) -> list[Fact]:
                return []
            async def consolidate(self) -> None:
                pass

        import asyncio
        memory = FakeMemory()
        results = asyncio.run(memory.recall_by_segment(MemorySegment.IDENTITY))
        assert results == []

    def test_archive_default_noop(self):
        from the_agents_playbook.memory.protocol import BaseMemoryProvider, Fact

        class FakeMemory(BaseMemoryProvider):
            async def store(self, fact: Fact) -> None:
                pass
            async def recall(self, query: str, top_k: int = 5) -> list[Fact]:
                return []
            async def consolidate(self) -> None:
                pass

        import asyncio
        memory = FakeMemory()
        # Should not raise
        asyncio.run(memory.archive("some-id"))
