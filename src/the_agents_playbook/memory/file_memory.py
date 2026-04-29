"""DualFileMemory — persistent memory using MEMORY.md and HISTORY.md.

MEMORY.md: long-term facts, always loaded into context (read/write structured).
HISTORY.md: append-only event log, grep-searchable (write only, line-at-a-time).
"""

import logging
from pathlib import Path
from time import monotonic
from typing import Any

from .protocol import BaseMemoryProvider, Fact

logger = logging.getLogger(__name__)

FACT_SEPARATOR = "\n---\n"
MEMORY_HEADER = "# Memory\n\nFacts extracted from conversation history.\n\n"


def _parse_facts(text: str) -> list[Fact]:
    """Parse MEMORY.md content into a list of Fact objects.

    Each fact is separated by '---'. Format per fact:
        content: <text>
        source: <origin>
        tags: <comma-separated>
    """
    facts: list[Fact] = []
    # Skip the header
    body = text.replace(MEMORY_HEADER, "", 1).strip()
    if not body:
        return facts

    blocks = body.split(FACT_SEPARATOR)
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        content = ""
        source = "unknown"
        tags: list[str] = []

        for line in block.split("\n"):
            if line.startswith("content: "):
                content = line[len("content: "):]
            elif line.startswith("source: "):
                source = line[len("source: "):]
            elif line.startswith("tags: "):
                raw = line[len("tags: "):]
                tags = [t.strip() for t in raw.split(",") if t.strip()]

        if content:
            facts.append(Fact(content=content, source=source, tags=tags))

    return facts


def _serialize_facts(facts: list[Fact]) -> str:
    """Serialize a list of Fact objects into MEMORY.md content."""
    blocks: list[str] = []
    for fact in facts:
        lines = [f"content: {fact.content}"]
        lines.append(f"source: {fact.source}")
        if fact.tags:
            lines.append(f"tags: {', '.join(fact.tags)}")
        blocks.append("\n".join(lines))

    if not blocks:
        return MEMORY_HEADER

    return MEMORY_HEADER + FACT_SEPARATOR.join(blocks) + "\n"


class DualFileMemory(BaseMemoryProvider):
    """Dual-file memory system: MEMORY.md for long-term facts, HISTORY.md for raw log.

    Usage:
        memory = DualFileMemory(directory=Path(".memory"))
        await memory.store(Fact(content="User prefers Python", source="user"))
        facts = await memory.recall("Python")
    """

    def __init__(self, directory: Path | str) -> None:
        self._directory = Path(directory)
        self._memory_path = self._directory / "MEMORY.md"
        self._history_path = self._directory / "HISTORY.md"

    def _ensure_dirs(self) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)

    # --- Read / Write MEMORY.md ---

    def _read_memory(self) -> str:
        if self._memory_path.exists():
            return self._memory_path.read_text(encoding="utf-8")
        return ""

    def _write_memory(self, content: str) -> None:
        self._ensure_dirs()
        self._memory_path.write_text(content, encoding="utf-8")

    # --- Append HISTORY.md ---

    def _append_history(self, event: str) -> None:
        self._ensure_dirs()
        with open(self._history_path, "a", encoding="utf-8") as f:
            f.write(event + "\n")

    # --- Public API ---

    async def store(self, fact: Fact) -> None:
        """Store a fact: append to HISTORY.md and update MEMORY.md."""
        self._ensure_dirs()

        # Append to history log
        history_entry = f"[{fact.source}] {fact.content}"
        self._append_history(history_entry)

        # Update MEMORY.md — read existing, add new, deduplicate by content
        existing = self.read_facts()
        content_set = {f.content for f in existing}

        if fact.content not in content_set:
            existing.append(fact)
            self._write_memory(_serialize_facts(existing))
            logger.info("Stored new fact: %s", fact.content)
        else:
            logger.debug("Fact already in MEMORY.md, skipped: %s", fact.content)

    async def store_event(self, text: str, source: str = "system") -> None:
        """Append a raw event to HISTORY.md without adding to MEMORY.md."""
        self._ensure_dirs()
        self._append_history(f"[{source}] {text}")

    def read_facts(self) -> list[Fact]:
        """Read and parse all facts from MEMORY.md (sync, no LLM needed)."""
        content = self._read_memory()
        return _parse_facts(content)

    def read_history(self) -> str:
        """Read the raw HISTORY.md content."""
        if self._history_path.exists():
            return self._history_path.read_text(encoding="utf-8")
        return ""

    async def recall(self, query: str, top_k: int = 5) -> list[Fact]:
        """Recall facts from MEMORY.md by substring matching.

        This is a simple grep-style search — no embeddings needed.
        For semantic search, use InMemoryVectorStore instead.
        """
        facts = self.read_facts()
        query_lower = query.lower()

        scored: list[tuple[Fact, int]] = []
        for fact in facts:
            if query_lower in fact.content.lower():
                scored.append((fact, fact.content.lower().count(query_lower)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [fact for fact, _ in scored[:top_k]]

    async def consolidate(self) -> None:
        """No-op for file memory. Consolidation is handled by LLMConsolidator."""
        pass

    @property
    def memory_path(self) -> Path:
        return self._memory_path

    @property
    def history_path(self) -> Path:
        return self._history_path
