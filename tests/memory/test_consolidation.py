"""Tests for memory/consolidation.py — LLMConsolidator."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from the_agents_playbook.memory import DualFileMemory, Fact
from the_agents_playbook.memory.consolidation import LLMConsolidator


@pytest.fixture
def memory(tmp_path: Path) -> DualFileMemory:
    return DualFileMemory(directory=tmp_path)


def _mock_llm_response(facts_json: str):
    """Create a mock MessageResponse for the consolidation LLM call."""
    from the_agents_playbook.providers.types import InputMessage, MessageResponse, OutputMessage

    return MessageResponse(
        message=OutputMessage(content=facts_json),
        stop_reason="stop",
    )


async def test_consolidation_extracts_facts(tmp_path: Path):
    """Consolidation should parse LLM JSON response and store new facts."""
    memory = DualFileMemory(directory=tmp_path)

    # Write some history
    await memory.store_event("User prefers Python", source="user")
    await memory.store_event("Project uses FastAPI", source="assistant")

    # Mock the LLM response
    mock_response = _mock_llm_response(
        '[{"content": "User prefers Python", "source": "user preference"}, '
        '{"content": "Project uses FastAPI", "source": "project stack"}]'
    )

    consolidator = LLMConsolidator(memory=memory)

    with patch(
        "the_agents_playbook.providers.OpenAIProvider"
    ) as MockProvider:
        mock_instance = AsyncMock()
        mock_instance.send_message = AsyncMock(return_value=mock_response)
        mock_instance.close = AsyncMock()
        MockProvider.return_value = mock_instance

        new_facts = await consolidator.consolidate()

    assert len(new_facts) == 2
    assert new_facts[0].content == "User prefers Python"

    # Facts should now be in MEMORY.md
    all_facts = memory.read_facts()
    assert len(all_facts) == 2


async def test_consolidation_deduplicates(tmp_path: Path):
    """Should not re-store facts already in MEMORY.md."""
    memory = DualFileMemory(directory=tmp_path)

    # Pre-populate MEMORY.md
    await memory.store(Fact(content="Already known", source="existing"))
    await memory.store_event("User mentioned Already known again", source="user")

    mock_response = _mock_llm_response(
        '[{"content": "Already known", "source": "existing"}, '
        '{"content": "Brand new fact", "source": "consolidation"}]'
    )

    consolidator = LLMConsolidator(memory=memory)

    with patch(
        "the_agents_playbook.providers.OpenAIProvider"
    ) as MockProvider:
        mock_instance = AsyncMock()
        mock_instance.send_message = AsyncMock(return_value=mock_response)
        mock_instance.close = AsyncMock()
        MockProvider.return_value = mock_instance

        new_facts = await consolidator.consolidate()

    # Only "Brand new fact" should be new
    assert len(new_facts) == 1
    assert new_facts[0].content == "Brand new fact"

    all_facts = memory.read_facts()
    assert len(all_facts) == 2


async def test_consolidation_empty_history(tmp_path: Path):
    """No history = no facts extracted."""
    memory = DualFileMemory(directory=tmp_path)

    consolidator = LLMConsolidator(memory=memory)
    new_facts = await consolidator.consolidate()

    assert new_facts == []
    assert len(memory.read_facts()) == 0


async def test_consolidation_llm_error(tmp_path: Path):
    """If the LLM call fails, should return empty list without crashing."""
    memory = DualFileMemory(directory=tmp_path)
    await memory.store_event("Some history", source="user")

    consolidator = LLMConsolidator(memory=memory)

    with patch(
        "the_agents_playbook.providers.OpenAIProvider"
    ) as MockProvider:
        mock_instance = AsyncMock()
        mock_instance.send_message = AsyncMock(side_effect=RuntimeError("API down"))
        mock_instance.close = AsyncMock()
        MockProvider.return_value = mock_instance

        new_facts = await consolidator.consolidate()

    assert new_facts == []


async def test_consolidation_truncates_long_history(tmp_path: Path):
    """History exceeding max_history_lines should be truncated."""
    memory = DualFileMemory(directory=tmp_path)

    # Write 50 lines of history
    for i in range(50):
        await memory.store_event(f"Event {i}", source="user")

    mock_response = _mock_llm_response('[{"content": "Summary fact", "source": "consolidation"}]')

    # Set a low max to force truncation
    consolidator = LLMConsolidator(memory=memory, max_history_lines=10)

    with patch(
        "the_agents_playbook.providers.OpenAIProvider"
    ) as MockProvider:
        mock_instance = AsyncMock()
        mock_instance.send_message = AsyncMock(return_value=mock_response)
        mock_instance.close = AsyncMock()
        MockProvider.return_value = mock_instance

        new_facts = await consolidator.consolidate()

    # Should still work with truncated history
    assert len(new_facts) == 1
