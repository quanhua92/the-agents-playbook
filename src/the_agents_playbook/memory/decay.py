"""Memory decay — importance-weighted exponential decay with access boosting.

The scoring formula determines which memories are most relevant at any
given time:

    score = importance * exp(-lambda * days) * (1 + ln(1 + access_count) * 0.1)

- importance: base relevance from the segment (0.0-1.0)
- lambda: decay rate — how fast the memory fades (0 for permanent)
- days: time since creation
- access_count: frequently recalled memories get a small boost

Permanent memories (identity, corrections) have lambda=0, so they
never decay regardless of time passed.
"""

import logging
import math
from time import monotonic

from .record import MemoryLifecycle, MemoryRecord

logger = logging.getLogger(__name__)

# Threshold below which a record should be pruned.
PRUNE_THRESHOLD = 0.01


class MemoryDecay:
    """Calculate decay scores and prune/archive stale memories.

    Usage:
        decay = MemoryDecay()
        score = decay.score(record, days_since_creation=30)
        decayed = decay.decay_and_archive(records, days_elapsed=1)
        pruned = decay.prune(records)
    """

    def score(self, record: MemoryRecord, days: float) -> float:
        """Calculate the current relevance score for a memory record.

        Args:
            record: The memory record to score.
            days: Number of days since the record was created.

        Returns:
            A relevance score between 0.0 and ~1.1 (slightly above 1.0
            for frequently-accessed permanent memories due to access boost).
        """
        if record.tier is None or record.importance is None or record.decay_rate is None:
            return 0.0

        # Exponential decay: exp(-lambda * days)
        time_factor = math.exp(-record.decay_rate * days)

        # Access boost: ln(1 + count) * 0.1
        access_boost = 1.0 + math.log(1 + record.access_count) * 0.1

        return record.importance * time_factor * access_boost

    def prune(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        """Mark records below the prune threshold as PRUNED.

        Permanent records are never pruned. Only ACTIVE records are
        considered — archived records are left alone.

        Args:
            records: All memory records.

        Returns:
            The list of records that were newly pruned.
        """
        pruned: list[MemoryRecord] = []
        now = monotonic()

        for record in records:
            if record.lifecycle != MemoryLifecycle.ACTIVE:
                continue
            if record.is_permanent:
                continue

            days = (now - record.timestamp) / 86400.0
            current_score = self.score(record, days)

            if current_score < PRUNE_THRESHOLD:
                record.lifecycle = MemoryLifecycle.PRUNED
                pruned.append(record)
                logger.debug(
                    "Pruned record: %s (score=%.4f, segment=%s)",
                    record.content[:50],
                    current_score,
                    record.segment.value,
                )

        return pruned

    def decay_and_archive(
        self,
        records: list[MemoryRecord],
        days_elapsed: float = 1.0,
    ) -> tuple[list[MemoryRecord], list[MemoryRecord]]:
        """Apply decay logic and transition records between lifecycle states.

        For each active record:
        - If the score drops below the archive threshold, mark as archived.
        - If already archived and below prune threshold, mark as pruned.

        Args:
            records: All memory records.
            days_elapsed: Simulated days to advance (default 1.0).

        Returns:
            Tuple of (newly_archived, newly_pruned) record lists.
        """
        archived: list[MemoryRecord] = []
        pruned: list[MemoryRecord] = []
        now = monotonic()

        for record in records:
            if record.is_permanent:
                continue

            days = (now - record.timestamp) / 86400.0
            current_score = self.score(record, days)

            if record.lifecycle == MemoryLifecycle.ACTIVE:
                if current_score < 0.05:  # archive threshold
                    record.lifecycle = MemoryLifecycle.ARCHIVED
                    archived.append(record)
                    logger.debug(
                        "Archived record: %s (score=%.4f)",
                        record.content[:50],
                        current_score,
                    )

        # Prune from both active and archived
        pruned = self.prune(records)

        return archived, pruned
