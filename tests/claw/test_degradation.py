"""Tests for claw.degradation — DegradationManager."""

import pytest

from the_agents_playbook.claw.degradation import DegradationManager, FallbackResult


class TestFallbackResult:
    def test_defaults(self):
        r = FallbackResult(handled=True, output="ok", strategy="test")
        assert r.handled is True
        assert r.output == "ok"
        assert r.strategy == "test"


class TestDegradationManager:
    @pytest.fixture
    def mgr(self):
        return DegradationManager()

    async def test_handle_tool_failure_default(self, mgr):
        result = await mgr.handle_tool_failure("shell", RuntimeError("timeout"))
        assert result.handled is True
        assert result.strategy == "tool_fallback"
        assert "shell" in result.output
        assert "timeout" in result.output

    async def test_handle_tool_failure_registered(self, mgr):
        mgr.register_tool_fallback("deploy", "Roll back deployment.")
        result = await mgr.handle_tool_failure("deploy", RuntimeError("500"))
        assert "Roll back" in result.output

    async def test_handle_llm_failure(self, mgr):
        result = await mgr.handle_llm_failure(ConnectionError("refused"))
        assert result.handled is True
        assert result.strategy == "llm_fallback"
        assert "unavailable" in result.output

    async def test_handle_context_overflow(self, mgr):
        result = await mgr.handle_context_overflow(max_tokens=8192)
        assert result.handled is True
        assert result.strategy == "context_compaction"
        assert "8192" in result.output
