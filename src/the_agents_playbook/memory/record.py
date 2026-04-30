"""Memory record — a Fact extended with segment, tier, and lifecycle metadata.

A MemoryRecord is what actually gets stored when you want tiered, decaying
memory. It wraps the base Fact with all the metadata needed for segmented
recall, importance scoring, and lifecycle management.
"""

from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Any

import numpy as np

from .segments import MemorySegment, MemoryTier, SEGMENT_DEFAULTS, SegmentConfig


class MemoryLifecycle(str, Enum):
    """Lifecycle states for a memory record.

    active:   Normal — participates in recall and decay.
    archived: Manually frozen — no decay, excluded from default recall.
    pruned:   Automatically removed by decay — ready for deletion.
    """

    ACTIVE = "active"
    ARCHIVED = "archived"
    PRUNED = "pruned"


@dataclass
class MemoryRecord:
    """An extended fact with segment classification, tier, and lifecycle.

    Attributes:
        content: The text content of the memory.
        source: Where this memory came from (user, assistant, tool).
        timestamp: When the memory was created.
        embedding: Optional vector embedding for semantic search.
        tags: Optional tags for filtering.
        segment: Which category this memory belongs to.
        tier: Retention tier (inherited from segment if not overridden).
        importance: Base importance 0.0-1.0 (inherited from segment if not set).
        decay_rate: Lambda for exponential decay (inherited from segment if not set).
        lifecycle: Current lifecycle state.
        supersedes: ID of another record this one replaces (for corrections).
        access_count: Number of times this record has been recalled.
        last_accessed_at: Timestamp of last recall.
        metadata: Arbitrary additional data.
    """

    content: str
    source: str
    timestamp: float = field(default_factory=monotonic)
    embedding: np.ndarray | None = None
    tags: list[str] = field(default_factory=list)
    segment: MemorySegment = MemorySegment.KNOWLEDGE
    tier: MemoryTier | None = None
    importance: float | None = None
    decay_rate: float | None = None
    lifecycle: MemoryLifecycle = MemoryLifecycle.ACTIVE
    supersedes: str | None = None
    access_count: int = 0
    last_accessed_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Fill in tier/importance/decay_rate from segment defaults if not set."""
        defaults = SEGMENT_DEFAULTS.get(self.segment)
        if defaults is None:
            return
        if self.tier is None:
            self.tier = defaults.tier
        if self.importance is None:
            self.importance = defaults.importance
        if self.decay_rate is None:
            self.decay_rate = defaults.decay_rate

    @property
    def is_permanent(self) -> bool:
        return self.tier == MemoryTier.PERMANENT

    def record_access(self) -> None:
        """Increment access count and update last-accessed timestamp."""
        self.access_count += 1
        self.last_accessed_at = monotonic()
