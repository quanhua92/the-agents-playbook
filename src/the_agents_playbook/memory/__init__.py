from .consolidation import LLMConsolidator
from .embedding_provider import OpenAIEmbeddingProvider
from .file_memory import DualFileMemory
from .protocol import BaseMemoryProvider, EmbeddingProvider, Fact
from .session import SessionPersistence
from .vector_memory import InMemoryVectorStore

__all__ = [
    "BaseMemoryProvider",
    "DualFileMemory",
    "EmbeddingProvider",
    "Fact",
    "InMemoryVectorStore",
    "LLMConsolidator",
    "OpenAIEmbeddingProvider",
    "SessionPersistence",
]
