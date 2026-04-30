"""Token-aware session compaction.

Adapted from the_agents_playbook/memory/session.py (SessionCompactor class).
When a conversation exceeds a token threshold, old messages are summarized
into a single summary message to free up context space.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SessionCompactor:
    """Token-aware session compaction.

    When a conversation exceeds a token threshold, old messages are
    summarized into a single summary message to free up context space.
    Always keeps the most recent messages intact.

    Usage:
        compactor = SessionCompactor(max_tokens=4000, keep_recent=4)
        compacted = compactor.compact(messages)
    """

    CHARS_PER_TOKEN: float = 4.0

    def __init__(
        self,
        max_tokens: int = 8000,
        keep_recent: int = 4,
        summarize_fn=None,
    ):
        self._max_tokens = max_tokens
        self._keep_recent = keep_recent
        self._summarize_fn = summarize_fn

    @staticmethod
    def estimate_tokens(messages: list[dict[str, Any]]) -> int:
        """Rough token count using the ~4 chars/token heuristic."""
        total_chars = 0
        for msg in messages:
            for value in msg.values():
                if isinstance(value, str):
                    total_chars += len(value)
        return int(total_chars / SessionCompactor.CHARS_PER_TOKEN)

    def compact(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compact messages if they exceed the token threshold."""
        if len(messages) <= self._keep_recent:
            return messages

        tokens = self.estimate_tokens(messages)
        if tokens <= self._max_tokens:
            return messages

        old = messages[:-self._keep_recent]
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

        If a summarize_fn was provided, it is called to produce an LLM-based
        summary. Otherwise falls back to simple concatenation.
        """
        if self._summarize_fn is not None:
            return self._summarize_fn(messages)

        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                if len(content) > 500:
                    content = content[:500] + "..."
                parts.append(f"{role}: {content}")

        if not parts:
            return "(no prior messages)"

        text = "\n".join(parts)
        max_chars = int(self._max_tokens * self.CHARS_PER_TOKEN * 0.5)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...(truncated)"

        return text
