"""Memory segments — categorize memories by type with per-segment defaults.

Not all memories are equal. A user's name (identity) should never decay,
while a temporary context ("I'm at a coffee shop") should decay within hours.
Segments give us the vocabulary to express these differences.
"""

from dataclasses import dataclass
from enum import Enum


class MemorySegment(str, Enum):
    """Categories of memory with different retention policies.

    Each segment maps to a natural category of things an agent might
    remember about a user or conversation.

    Corrections use the `supersedes` field on any record rather than
    a dedicated segment — this lets any segment carry a correction.
    """

    # Permanent — never decay
    IDENTITY = "identity"  # Name, email, pronouns

    # Long-term — very slow decay
    EXPERTISE = "expertise"  # Skills, proficiency levels
    PREFERENCE = "preference"  # Style, format preferences
    RELATIONSHIP = "relationship"  # Org, team, family
    GOAL = "goal"  # Intentions, plans

    # Medium-term — moderate decay
    FEEDBACK = "feedback"  # Agent performance signals
    PROJECT = "project"  # Active deliverables
    KNOWLEDGE = "knowledge"  # Tool/system facts

    # Short-term — fast decay
    CONTEXT = "context"  # Temporary situation


class MemoryTier(str, Enum):
    """Retention tiers controlling how aggressively memories decay.

    PERMANENT:     Never decay (identity).
    LONG_TERM:     Very slow decay (expertise, preferences, relationship, goals).
    MEDIUM_TERM:   Moderate decay (feedback, project, knowledge).
    SHORT_TERM:    Fast decay (temporary context).
    """

    PERMANENT = "permanent"
    LONG_TERM = "long_term"
    MEDIUM_TERM = "medium_term"
    SHORT_TERM = "short_term"


@dataclass(frozen=True)
class SegmentConfig:
    """Per-segment configuration controlling decay behavior.

    Attributes:
        tier: The retention tier for this segment.
        importance: Base importance score (0.0-1.0).
        decay_rate: Lambda in the exponential decay formula.
            Higher = faster decay. PERMANENT tier sets this to 0.
    """

    tier: MemoryTier
    importance: float
    decay_rate: float


# Default configuration for each segment.
# These are the "factory settings" — callers can override per-record.
SEGMENT_DEFAULTS: dict[MemorySegment, SegmentConfig] = {
    # Permanent
    MemorySegment.IDENTITY: SegmentConfig(
        tier=MemoryTier.PERMANENT,
        importance=1.0,
        decay_rate=0.0,
    ),
    # Long-term
    MemorySegment.EXPERTISE: SegmentConfig(
        tier=MemoryTier.LONG_TERM,
        importance=0.85,
        decay_rate=0.001,
    ),
    MemorySegment.PREFERENCE: SegmentConfig(
        tier=MemoryTier.LONG_TERM,
        importance=0.8,
        decay_rate=0.002,
    ),
    MemorySegment.RELATIONSHIP: SegmentConfig(
        tier=MemoryTier.LONG_TERM,
        importance=0.7,
        decay_rate=0.002,
    ),
    MemorySegment.GOAL: SegmentConfig(
        tier=MemoryTier.LONG_TERM,
        importance=0.75,
        decay_rate=0.003,
    ),
    # Medium-term
    MemorySegment.FEEDBACK: SegmentConfig(
        tier=MemoryTier.MEDIUM_TERM,
        importance=0.6,
        decay_rate=0.005,
    ),
    MemorySegment.PROJECT: SegmentConfig(
        tier=MemoryTier.MEDIUM_TERM,
        importance=0.6,
        decay_rate=0.01,
    ),
    MemorySegment.KNOWLEDGE: SegmentConfig(
        tier=MemoryTier.MEDIUM_TERM,
        importance=0.5,
        decay_rate=0.008,
    ),
    # Short-term
    MemorySegment.CONTEXT: SegmentConfig(
        tier=MemoryTier.SHORT_TERM,
        importance=0.3,
        decay_rate=0.05,
    ),
}
