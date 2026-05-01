"""LLMConsolidator — extract structured facts from raw conversation history."""

import json
import logging

from .file_memory import DualFileMemory
from .protocol import Fact

logger = logging.getLogger(__name__)

CONSOLIDATION_PROMPT = """You are a fact extraction system. Read the conversation history below and extract the most important facts.

Rules:
- Extract only factual, durable information (preferences, decisions, facts about the user/project).
- Ignore transient conversational noise (greetings, acknowledgments, repeated info).
- Output a JSON array of objects with "content" and "source" fields.
- Keep facts concise — one sentence each.
- Maximum 10 facts.

History:
{history}

Output ONLY a JSON array. No explanation, no markdown. Example:
[{{"content": "User prefers Python over JavaScript", "source": "user preference"}}, {{"content": "Project uses OpenAI API", "source": "project context"}}]"""


class LLMConsolidator:
    """Uses the LLM to extract structured facts from raw conversation history.

    Consolidation vs compaction:
    - Compaction: summarize and discard → loses intent, misses edge cases
    - Consolidation: extract and index → preserves facts in searchable form

    Usage:
        consolidator = LLMConsolidator(memory=file_memory, provider=provider)
        await consolidator.consolidate()
    """

    def __init__(
        self,
        memory: DualFileMemory,
        max_history_lines: int = 200,
    ) -> None:
        self._memory = memory
        self._max_history_lines = max_history_lines

    async def consolidate(self) -> list[Fact]:
        """Read HISTORY.md, send to LLM, extract facts, store in MEMORY.md.

        Returns the list of newly extracted facts.
        """
        history = self._memory.read_history()
        if not history.strip():
            logger.info("No history to consolidate")
            return []

        # Truncate history if it exceeds the limit
        lines = history.strip().split("\n")
        if len(lines) > self._max_history_lines:
            history = "\n".join(lines[-self._max_history_lines :])
            logger.info(
                "Truncated history to last %d lines for consolidation",
                self._max_history_lines,
            )

        prompt = CONSOLIDATION_PROMPT.format(history=history)
        extracted = await self._extract_facts_via_llm(prompt)

        if not extracted:
            logger.info("No facts extracted from consolidation")
            return []

        # Store new facts (DualFileMemory deduplicates by content)
        new_facts: list[Fact] = []
        existing = self._memory.read_facts()
        existing_content = {f.content for f in existing}

        for item in extracted:
            content = item.get("content", "").strip()
            source = item.get("source", "consolidation")
            if content and content not in existing_content:
                fact = Fact(content=content, source=source)
                await self._memory.store(fact)
                new_facts.append(fact)

        logger.info("Consolidation extracted %d new facts", len(new_facts))
        return new_facts

    async def _extract_facts_via_llm(self, prompt: str) -> list[dict[str, str]]:
        """Send the consolidation prompt to the LLM and parse the JSON response.

        Importing provider here to avoid circular imports at module level.
        """
        from the_agents_playbook.providers import (
            InputMessage,
            MessageRequest,
            OpenAIProvider,
            ResponseFormat,
        )
        from the_agents_playbook.utils.schema import flatten_json_schema

        provider = OpenAIProvider()

        fact_schema = {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "source": {"type": "string"},
            },
            "required": ["content", "source"],
            "additionalProperties": False,
        }

        # The array wrapper schema
        response_format = ResponseFormat(
            json_schema_name="FactArray",
            json_schema=flatten_json_schema(
                {
                    "type": "object",
                    "properties": {
                        "facts": {
                            "type": "array",
                            "items": fact_schema,
                        }
                    },
                    "required": ["facts"],
                    "additionalProperties": False,
                }
            ),
        )

        try:
            response = await provider.send_message(
                MessageRequest(
                    model=self._get_model(),
                    system="Extract structured facts from conversation history.",
                    messages=[InputMessage(role="user", content=prompt)],
                    response_format=response_format,
                )
            )

            raw = response.message.content
            if not raw:
                return []

            # The LLM might return the array directly or wrapped in an object
            data = json.loads(raw)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "facts" in data:
                return data["facts"]
            return []

        except Exception as e:
            logger.error("Consolidation LLM call failed: %s", e)
            return []
        finally:
            await provider.close()

    @staticmethod
    def _get_model() -> str:
        """Get the model name from settings."""
        from the_agents_playbook import settings

        return settings.openai_model
