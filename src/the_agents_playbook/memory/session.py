"""Session persistence — save and restore conversation state as JSONL."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SessionPersistence:
    """Save and restore conversation state as JSONL files.

    Each line is a JSON object with: role, content, timestamp.
    This enables pausing a conversation and resuming it later.

    Usage:
        session = SessionPersistence()
        await session.save(messages, Path("session.jsonl"))
        messages = await session.load(Path("session.jsonl"))
    """

    async def save(
        self,
        messages: list[dict[str, Any]],
        path: Path | str,
    ) -> None:
        """Write conversation messages to a JSONL file.

        Each message dict should have at least 'role' and 'content' keys.
        A 'timestamp' field is added if not present.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for msg in messages:
                entry = {
                    "role": msg.get("role", "unknown"),
                    "content": msg.get("content", ""),
                    "timestamp": msg.get(
                        "timestamp",
                        datetime.now(timezone.utc).isoformat(),
                    ),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info("Saved %d messages to %s", len(messages), path)

    async def load(self, path: Path | str) -> list[dict[str, Any]]:
        """Load conversation messages from a JSONL file.

        Returns a list of dicts with role, content, timestamp.
        Returns empty list if the file doesn't exist.
        """
        path = Path(path)

        if not path.exists():
            logger.warning("Session file not found: %s", path)
            return []

        messages: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed line %d in %s: %s", line_num, path, e)

        logger.info("Loaded %d messages from %s", len(messages), path)
        return messages

    async def append(
        self,
        message: dict[str, Any],
        path: Path | str,
    ) -> None:
        """Append a single message to an existing JSONL session file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "role": message.get("role", "unknown"),
            "content": message.get("content", ""),
            "timestamp": message.get(
                "timestamp",
                datetime.now(timezone.utc).isoformat(),
            ),
        }

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def list_sessions(self, directory: Path | str) -> list[Path]:
        """List all JSONL session files in a directory."""
        dir_path = Path(directory)
        if not dir_path.exists():
            return []
        return sorted(dir_path.glob("*.jsonl"))


class SessionCompactor:
    """Token-aware session compaction.

    When a conversation exceeds a token threshold, old messages are
    summarized into a single summary message to free up context space.
    Always keeps the most recent messages intact.

    Usage:
        compactor = SessionCompactor(max_tokens=4000, keep_recent=4)
        compacted = compactor.compact(messages)
    """

    # Rough token estimate: ~4 characters per token for English text.
    CHARS_PER_TOKEN: float = 4.0

    def __init__(
        self,
        max_tokens: int = 8000,
        keep_recent: int = 4,
        summarize_fn=None,
    ):
        """Initialize compactor.

        Args:
            max_tokens: Approximate token budget for the full message list.
            keep_recent: Number of most-recent messages to preserve verbatim.
            summarize_fn: Optional async callable(messages) -> str.
                If provided, old messages are summarized via this function.
                If None, old messages are simply concatenated into a summary.
        """
        self._max_tokens = max_tokens
        self._keep_recent = keep_recent
        self._summarize_fn = summarize_fn

    @staticmethod
    def estimate_tokens(messages: list[dict[str, Any]]) -> int:
        """Rough token count for a list of messages.

        Uses the ~4 chars/token heuristic. Counts role, content,
        and any other string fields.
        """
        total_chars = 0
        for msg in messages:
            for value in msg.values():
                if isinstance(value, str):
                    total_chars += len(value)
        return int(total_chars / SessionCompactor.CHARS_PER_TOKEN)

    def compact(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compact messages if they exceed the token threshold.

        If under budget, returns messages unchanged.
        If over budget, summarizes old messages and keeps recent ones intact.
        """
        if len(messages) <= self._keep_recent:
            return messages

        tokens = self.estimate_tokens(messages)
        if tokens <= self._max_tokens:
            return messages

        old = messages[: -self._keep_recent]
        recent = messages[-self._keep_recent:]

        summary_content = self._build_summary(old)
        summary_msg = {
            "role": "user",
            "content": f"[Conversation summary] {summary_content}",
        }

        logger.info(
            "Compacted session: %d messages (%d tokens) -> %d messages",
            len(messages),
            tokens,
            len(recent) + 1,
        )

        return [summary_msg] + recent

    def _build_summary(self, messages: list[dict[str, Any]]) -> str:
        """Build a summary from old messages.

        Concatenates message roles and content into a compact text block.
        """
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                # Truncate very long individual messages
                if len(content) > 500:
                    content = content[:500] + "..."
                parts.append(f"{role}: {content}")

        if not parts:
            return "(no prior messages)"

        text = "\n".join(parts)
        # Cap the summary length to avoid it being too large itself
        max_chars = int(self._max_tokens * self.CHARS_PER_TOKEN * 0.5)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...(truncated)"

        return text
