"""Tests for memory/file_memory.py — DualFileMemory."""

from pathlib import Path

import pytest

from the_agents_playbook.memory import DualFileMemory, Fact
from the_agents_playbook.memory.file_memory import _parse_facts, _serialize_facts


# --- Parsing / Serialization ---


def test_parse_empty():
    assert _parse_facts("") == []
    assert _parse_facts("# Memory\n\nFacts extracted.\n\n") == []


def test_parse_single_fact():
    text = (
        "# Memory\n\n"
        "Facts extracted from conversation history.\n\n"
        "content: User prefers Python\n"
        "source: user\n"
    )
    facts = _parse_facts(text)
    assert len(facts) == 1
    assert facts[0].content == "User prefers Python"
    assert facts[0].source == "user"
    assert facts[0].tags == []


def test_parse_fact_with_tags():
    text = (
        "# Memory\n\n"
        "Facts extracted from conversation history.\n\n"
        "content: User prefers Python\n"
        "source: user\n"
        "tags: preference, language\n"
    )
    facts = _parse_facts(text)
    assert len(facts) == 1
    assert set(facts[0].tags) == {"preference", "language"}


def test_parse_multiple_facts():
    text = (
        "# Memory\n\n"
        "Facts extracted from conversation history.\n\n"
        "content: Fact A\n"
        "source: src_a\n"
        "\n---\n"
        "content: Fact B\n"
        "source: src_b\n"
    )
    facts = _parse_facts(text)
    assert len(facts) == 2
    assert facts[0].content == "Fact A"
    assert facts[1].content == "Fact B"


def test_roundtrip_serialize_parse():
    facts = [
        Fact(content="Python is great", source="user", tags=["language"]),
        Fact(content="FastAPI is used", source="project"),
    ]
    serialized = _serialize_facts(facts)
    parsed = _parse_facts(serialized)
    assert len(parsed) == 2
    assert parsed[0].content == "Python is great"
    assert parsed[0].source == "user"
    assert set(parsed[0].tags) == {"language"}
    assert parsed[1].content == "FastAPI is used"


def test_serialize_empty():
    text = _serialize_facts([])
    assert "Memory" in text
    assert "Facts" in text


# --- DualFileMemory ---


async def test_store_creates_memory_md(tmp_path: Path):
    memory = DualFileMemory(directory=tmp_path)
    await memory.store(Fact(content="test fact", source="unit"))

    assert memory.memory_path.exists()
    facts = memory.read_facts()
    assert len(facts) == 1
    assert facts[0].content == "test fact"


async def test_store_deduplicates(tmp_path: Path):
    memory = DualFileMemory(directory=tmp_path)
    await memory.store(Fact(content="same fact", source="a"))
    await memory.store(Fact(content="same fact", source="b"))

    facts = memory.read_facts()
    assert len(facts) == 1


async def test_store_appends_history(tmp_path: Path):
    memory = DualFileMemory(directory=tmp_path)
    await memory.store(Fact(content="event 1", source="user"))
    await memory.store(Fact(content="event 2", source="assistant"))

    history = memory.read_history()
    assert "event 1" in history
    assert "event 2" in history
    # Should have 2 lines (store_event not called, but store also appends)
    lines = [l for l in history.strip().split("\n") if l]
    assert len(lines) == 2


async def test_store_event_appends_only_history(tmp_path: Path):
    memory = DualFileMemory(directory=tmp_path)
    await memory.store_event("raw event text", source="system")

    history = memory.read_history()
    assert "raw event text" in history

    # Should NOT be in MEMORY.md
    facts = memory.read_facts()
    assert len(facts) == 0


async def test_recall_substring_match(tmp_path: Path):
    memory = DualFileMemory(directory=tmp_path)
    await memory.store(Fact(content="User loves Python", source="user"))
    await memory.store(Fact(content="User uses JavaScript", source="user"))
    await memory.store(Fact(content="The sky is blue", source="fact"))

    results = await memory.recall("Python")
    assert len(results) == 1
    assert "Python" in results[0].content

    results = await memory.recall("the")
    # Matches "The sky is blue" (case-insensitive)
    assert len(results) >= 1


async def test_recall_no_match(tmp_path: Path):
    memory = DualFileMemory(directory=tmp_path)
    await memory.store(Fact(content="unrelated", source="test"))

    results = await memory.recall("nonexistent")
    assert results == []


async def test_read_history_empty(tmp_path: Path):
    memory = DualFileMemory(directory=tmp_path)
    assert memory.read_history() == ""


async def test_consolidate_noop(tmp_path: Path):
    """DualFileMemory.consolidate is a no-op."""
    memory = DualFileMemory(directory=tmp_path)
    await memory.consolidate()  # Should not raise


async def test_creates_directory_on_demand(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "c"
    memory = DualFileMemory(directory=nested)
    await memory.store(Fact(content="deep", source="test"))

    assert nested.exists()
    assert memory.read_facts()[0].content == "deep"


async def test_property_paths(tmp_path: Path):
    memory = DualFileMemory(directory=tmp_path)
    assert memory.memory_path == tmp_path / "MEMORY.md"
    assert memory.history_path == tmp_path / "HISTORY.md"
