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
