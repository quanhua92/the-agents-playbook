"""Tests for memory/session.py — SessionPersistence."""

import json
from pathlib import Path

import pytest

from the_agents_playbook.memory.session import SessionPersistence


@pytest.fixture
def session() -> SessionPersistence:
    return SessionPersistence()


MESSAGES = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"},
]


async def test_save_and_load(session: SessionPersistence, tmp_path: Path):
    path = tmp_path / "session.jsonl"
    await session.save(MESSAGES, path)

    assert path.exists()
    loaded = await session.load(path)
    assert len(loaded) == 3
    assert loaded[0]["role"] == "user"
    assert loaded[0]["content"] == "Hello"
    assert loaded[0]["timestamp"] is not None


async def test_save_adds_timestamp(session: SessionPersistence, tmp_path: Path):
    path = tmp_path / "session.jsonl"
    await session.save([{"role": "user", "content": "test"}], path)

    loaded = await session.load(path)
    assert "timestamp" in loaded[0]


async def test_load_nonexistent(session: SessionPersistence, tmp_path: Path):
    loaded = await session.load(tmp_path / "nope.jsonl")
    assert loaded == []


async def test_load_malformed_lines(session: SessionPersistence, tmp_path: Path):
    path = tmp_path / "session.jsonl"

    # Write a valid line then a malformed line then another valid line
    lines = [
        json.dumps({"role": "user", "content": "valid 1"}),
        "NOT JSON",
        json.dumps({"role": "assistant", "content": "valid 2"}),
    ]
    path.write_text("\n".join(lines) + "\n")

    loaded = await session.load(path)
    assert len(loaded) == 2
    assert loaded[0]["content"] == "valid 1"
    assert loaded[1]["content"] == "valid 2"


async def test_append(session: SessionPersistence, tmp_path: Path):
    path = tmp_path / "session.jsonl"
    await session.save(MESSAGES[:2], path)

    await session.append({"role": "assistant", "content": "New message"}, path)

    loaded = await session.load(path)
    assert len(loaded) == 3
    assert loaded[2]["content"] == "New message"


async def test_append_creates_directory(session: SessionPersistence, tmp_path: Path):
    nested = tmp_path / "a" / "b"
    path = nested / "session.jsonl"
    await session.append({"role": "user", "content": "test"}, path)

    assert nested.exists()
    loaded = await session.load(path)
    assert len(loaded) == 1


async def test_list_sessions(session: SessionPersistence, tmp_path: Path):
    await session.save(MESSAGES, tmp_path / "a.jsonl")
    await session.save(MESSAGES, tmp_path / "b.jsonl")

    # Create a non-jsonl file that should be ignored
    (tmp_path / "readme.txt").write_text("not a session")

    sessions = session.list_sessions(tmp_path)
    assert len(sessions) == 2
    assert sessions[0].name == "a.jsonl"
    assert sessions[1].name == "b.jsonl"


async def test_list_sessions_empty_dir(session: SessionPersistence, tmp_path: Path):
    sessions = session.list_sessions(tmp_path)
    assert sessions == []


async def test_list_sessions_missing_dir(session: SessionPersistence, tmp_path: Path):
    sessions = session.list_sessions(tmp_path / "nonexistent")
    assert sessions == []


async def test_preserves_existing_timestamp(session: SessionPersistence, tmp_path: Path):
    path = tmp_path / "session.jsonl"
    ts = "2024-01-01T00:00:00Z"
    await session.save([{"role": "user", "content": "test", "timestamp": ts}], path)

    loaded = await session.load(path)
    assert loaded[0]["timestamp"] == ts


async def test_empty_message_list(session: SessionPersistence, tmp_path: Path):
    path = tmp_path / "session.jsonl"
    await session.save([], path)

    loaded = await session.load(path)
    assert loaded == []
    assert path.exists()
