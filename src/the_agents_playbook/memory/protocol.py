"""Memory protocol — contracts for storing, recalling, and consolidating facts."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

import numpy as np


@dataclass
class Fact:
    """A single piece of knowledge extracted from conversation or tool results."""

    content: str
    source: str
    timestamp: float = field(default_factory=monotonic)
    embedding: np.ndarray | None = None
    tags: list[str] = field(default_factory=list)


class EmbeddingProvider(ABC):
    """Pluggable embedding provider — converts text to numpy vectors."""

    @abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """Convert text to a numpy vector."""
        ...


class BaseMemoryProvider(ABC):
    """Abstract interface for memory storage and retrieval."""

    @abstractmethod
    async def store(self, fact: Fact) -> None:
        """Store a fact in memory."""
        ...

    @abstractmethod
    async def recall(self, query: str, top_k: int = 5) -> list[Fact]:
        """Recall the most relevant facts for a given query."""
        ...

    @abstractmethod
    async def consolidate(self) -> None:
        """Consolidate raw history into structured, indexed facts."""
        ...

    # --- Segmented memory extensions ---

    async def store_record(self, record: Any) -> None:
        """Store a MemoryRecord with segment/tier metadata.

        Default implementation falls back to store() with a plain Fact.
        Subclasses should override to use the full record metadata.
        """
        fact = Fact(
            content=record.content,
            source=record.source,
            timestamp=record.timestamp,
            embedding=record.embedding,
            tags=record.tags,
        )
        await self.store(fact)

    async def recall_by_segment(
        self, segment: Any, top_k: int = 5
    ) -> list[Any]:
        """Recall memories filtered by segment.

        Default implementation returns an empty list.
        Subclasses should override to filter records by segment.
        """
        return []

    async def archive(self, memory_id: str) -> None:
        """Archive a memory record by ID, freezing it from decay.

        Default implementation is a no-op.
        Subclasses should override to update the record's lifecycle.
        """
        pass
