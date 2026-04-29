"""Tests for guardrails.hooks — HookSystem."""

from unittest.mock import AsyncMock

import pytest

from the_agents_playbook.guardrails.hooks import (
    ON_TOOL_CALL,
    ON_TOOL_RESULT,
    ON_TURN_END,
    ON_TURN_START,
    HookSystem,
)


class TestHookSystem:
    @pytest.fixture
    def hooks(self):
        return HookSystem()

    async def test_register_and_emit(self, hooks):
        handler = AsyncMock()
        hooks.on("on_test", handler)
        await hooks.emit("on_test", key="value")
        handler.assert_called_once_with(key="value")

    async def test_multiple_handlers(self, hooks):
        h1 = AsyncMock()
        h2 = AsyncMock()
        hooks.on("on_test", h1)
        hooks.on("on_test", h2)
        await hooks.emit("on_test")
        h1.assert_called_once()
        h2.assert_called_once()

    async def test_emit_unknown_event_no_error(self, hooks):
        await hooks.emit("nonexistent")  # should not raise

    async def test_handlers_returns_empty_for_unknown(self, hooks):
        assert hooks.handlers("nonexistent") == []

    async def test_handlers_returns_registered(self, hooks):
        handler = AsyncMock()
        hooks.on("on_test", handler)
        assert handler in hooks.handlers("on_test")

    async def test_off_removes_specific_handler(self, hooks):
        h1 = AsyncMock()
        h2 = AsyncMock()
        hooks.on("on_test", h1)
        hooks.on("on_test", h2)
        hooks.off("on_test", h1)
        assert h1 not in hooks.handlers("on_test")
        assert h2 in hooks.handlers("on_test")

    async def test_off_removes_all_handlers(self, hooks):
        h1 = AsyncMock()
        h2 = AsyncMock()
        hooks.on("on_test", h1)
        hooks.on("on_test", h2)
        hooks.off("on_test")
        assert hooks.handlers("on_test") == []

    async def test_off_nonexistent_event_no_error(self, hooks):
        hooks.off("nonexistent")  # should not raise

    async def test_clear_removes_all(self, hooks):
        h1 = AsyncMock()
        hooks.on("e1", h1)
        hooks.on("e2", h1)
        hooks.clear()
        assert hooks.handlers("e1") == []
        assert hooks.handlers("e2") == []

    async def test_handler_error_doesnt_stop_others(self, hooks):
        bad = AsyncMock(side_effect=RuntimeError("boom"))
        good = AsyncMock()
        hooks.on("on_test", bad)
        hooks.on("on_test", good)
        await hooks.emit("on_test")
        good.assert_called_once()

    async def test_standard_event_constants(self):
        assert ON_TURN_START == "on_turn_start"
        assert ON_TOOL_CALL == "on_tool_call"
        assert ON_TOOL_RESULT == "on_tool_result"
        assert ON_TURN_END == "on_turn_end"

    async def test_standard_events_work(self, hooks):
        handler = AsyncMock()
        hooks.on(ON_TOOL_CALL, handler)
        await hooks.emit(ON_TOOL_CALL, tool_name="shell", arguments={"cmd": "ls"})
        handler.assert_called_once_with(tool_name="shell", arguments={"cmd": "ls"})
