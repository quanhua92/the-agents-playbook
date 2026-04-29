"""Tests for claw.repair — RepairLoop."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from the_agents_playbook.claw.repair import RepairLoop, RepairResult
from the_agents_playbook.tools.protocol import ToolResult


class TestRepairResult:
    def test_success(self):
        r = RepairResult(success=True, attempts=2, final_output="ok", error_history=["err1"])
        assert r.success is True
        assert r.attempts == 2
        assert r.final_output == "ok"
        assert len(r.error_history) == 1

    def test_failure(self):
        r = RepairResult(success=False, attempts=3, error_history=["e1", "e2", "e3"])
        assert r.success is False
        assert r.final_output is None
        assert len(r.error_history) == 3

    def test_defaults(self):
        r = RepairResult(success=True)
        assert r.attempts == 1
        assert r.error_history == []


class TestRepairLoop:
    @pytest.fixture
    def registry(self):
        return MagicMock()

    @pytest.fixture
    def loop(self, registry):
        return RepairLoop(registry, max_retries=3)

    async def test_succeeds_first_try(self, loop, registry):
        registry.dispatch = AsyncMock(return_value=ToolResult(output="done"))
        result = await loop.repair("echo", {"msg": "hi"})
        assert result.success is True
        assert result.attempts == 1
        assert result.final_output == "done"

    async def test_retries_on_error_result(self, loop, registry):
        registry.dispatch = AsyncMock(side_effect=[
            ToolResult(output="fail1", error=True),
            ToolResult(output="fail2", error=True),
            ToolResult(output="ok"),
        ])
        result = await loop.repair("tool", {})
        assert result.success is True
        assert result.attempts == 3
        assert len(result.error_history) == 2

    async def test_retries_on_exception(self, loop, registry):
        registry.dispatch = AsyncMock(side_effect=[
            RuntimeError("boom"),
            RuntimeError("bam"),
            ToolResult(output="recovered"),
        ])
        result = await loop.repair("tool", {})
        assert result.success is True
        assert result.attempts == 3

    async def test_exhausts_retries(self, loop, registry):
        registry.dispatch = AsyncMock(return_value=ToolResult(output="fail", error=True))
        result = await loop.repair("tool", {})
        assert result.success is False
        assert result.attempts == 3
        assert len(result.error_history) == 3

    def test_max_retries(self, loop):
        assert loop.max_retries == 3

    async def test_diagnose(self, loop):
        diagnosis = await loop.diagnose("shell", "permission denied")
        assert "shell" in diagnosis
        assert "permission denied" in diagnosis
