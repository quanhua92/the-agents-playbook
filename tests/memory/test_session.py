"""Tests for memory/session.py — SessionPersistence."""

import json
from pathlib import Path

import pytest

from the_agents_playbook.memory.session import SessionCompactor, SessionPersistence


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


# ---------------------------------------------------------------------------
# SessionCompactor tests
# ---------------------------------------------------------------------------


def make_msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


class TestSessionCompactor:
    def test_estimate_tokens_empty(self):
        assert SessionCompactor.estimate_tokens([]) == 0

    def test_estimate_tokens_basic(self):
        msgs = [make_msg("user", "Hello world")]
        tokens = SessionCompactor.estimate_tokens(msgs)
        # "user" (4) + "Hello world" (11) = 15 chars / 4 = 3.75 -> 3
        assert tokens == 3

    def test_compact_under_threshold_returns_unchanged(self):
        compactor = SessionCompactor(max_tokens=1000, keep_recent=2)
        msgs = [make_msg("user", "Hi"), make_msg("assistant", "Hello!")]
        result = compactor.compact(msgs)
        assert result == msgs

    def test_compact_fewer_messages_than_keep_recent(self):
        compactor = SessionCompactor(max_tokens=1, keep_recent=5)
        msgs = [make_msg("user", "Hi")]
        result = compactor.compact(msgs)
        assert result == msgs

    def test_compact_over_threshold_summarizes_old(self):
        compactor = SessionCompactor(max_tokens=5, keep_recent=2)
        msgs = [
            make_msg("user", "Message one"),
            make_msg("assistant", "Message two"),
            make_msg("user", "Message three"),
            make_msg("assistant", "Message four"),
            make_msg("user", "Keep this"),
            make_msg("assistant", "And this"),
        ]
        result = compactor.compact(msgs)
        # Should be: [summary] + last 2 messages = 3 total
        assert len(result) == 3
        # First message should be the summary
        assert result[0]["role"] == "user"
        assert "[Conversation summary]" in result[0]["content"]
        # Last 2 should be preserved verbatim
        assert result[1] == msgs[-2]
        assert result[2] == msgs[-1]

    def test_compact_preserves_recent_exactly(self):
        compactor = SessionCompactor(max_tokens=1, keep_recent=3)
        msgs = [
            make_msg("user", f"old msg {i}")
            for i in range(10)
        ]
        result = compactor.compact(msgs)
        assert len(result) == 4  # 1 summary + 3 recent
        assert result[-1] == msgs[-1]
        assert result[-2] == msgs[-2]
        assert result[-3] == msgs[-3]

    def test_compact_summary_contains_old_content(self):
        compactor = SessionCompactor(max_tokens=100, keep_recent=1)
        msgs = [
            make_msg("user", "Discussing Python"),
            make_msg("assistant", "Python is great"),
            make_msg("user", "Latest message"),
        ]
        result = compactor.compact(msgs)
        summary = result[0]["content"]
        assert "Python" in summary
        assert "Latest message" not in summary  # this is in the recent part

    def test_compact_truncates_long_messages(self):
        compactor = SessionCompactor(max_tokens=1, keep_recent=1)
        long_content = "x" * 1000
        msgs = [
            make_msg("user", long_content),
            make_msg("user", "keep"),
        ]
        result = compactor.compact(msgs)
        summary = result[0]["content"]
        assert "..." in summary  # truncated

    def test_compact_empty_old_messages(self):
        compactor = SessionCompactor(max_tokens=1, keep_recent=2)
        msgs = [
            make_msg("user", ""),
            make_msg("user", "keep"),
            make_msg("assistant", "this"),
        ]
        result = compactor.compact(msgs)
        # Even though first message is empty, it's old so gets summarized
        assert len(result) == 3
