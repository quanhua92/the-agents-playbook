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
