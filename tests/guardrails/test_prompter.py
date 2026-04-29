"""Tests for guardrails.prompter — Prompter ABC and implementations."""

from unittest.mock import AsyncMock

import pytest

from the_agents_playbook.guardrails.permissions import RiskLevel
from the_agents_playbook.guardrails.prompter import (
    DenyAllPrompter,
    Prompter,
    SilentPrompter,
    TerminalPrompter,
)


class TestTerminalPrompter:
    @pytest.fixture
    def prompter(self):
        return TerminalPrompter()

    async def test_confirm_yes(self):
        prompter = TerminalPrompter(input_fn=AsyncMock(return_value="y"))
        result = await prompter.confirm("Proceed?")
        assert result is True

    async def test_confirm_no(self):
        prompter = TerminalPrompter(input_fn=AsyncMock(return_value="n"))
        result = await prompter.confirm("Proceed?")
        assert result is False

    async def test_confirm_whitespace_yes(self):
        prompter = TerminalPrompter(input_fn=AsyncMock(return_value="  yes  "))
        result = await prompter.confirm("Proceed?")
        assert result is True

    async def test_confirm_uppercase_yes(self):
        prompter = TerminalPrompter(input_fn=AsyncMock(return_value="Y"))
        result = await prompter.confirm("Proceed?")
        assert result is True

    async def test_confirm_receives_risk_label(self):
        input_fn = AsyncMock(return_value="y")
        prompter = TerminalPrompter(input_fn=input_fn)
        await prompter.confirm("Delete?", risk=RiskLevel.DANGER)
        call_arg = input_fn.call_args[0][0]
        assert "DANGER" in call_arg


class TestSilentPrompter:
    async def test_always_approves(self):
        prompter = SilentPrompter()
        assert await prompter.confirm("anything") is True

    async def test_approves_danger(self):
        prompter = SilentPrompter()
        assert await prompter.confirm("delete all", risk=RiskLevel.DANGER) is True


class TestDenyAllPrompter:
    async def test_always_denies(self):
        prompter = DenyAllPrompter()
        assert await prompter.confirm("anything") is False

    async def test_denies_read_only(self):
        prompter = DenyAllPrompter()
        assert await prompter.confirm("read file", risk=RiskLevel.READ_ONLY) is False


class TestPrompterABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Prompter()
