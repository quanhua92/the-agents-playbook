"""Tests for workflows.hooks — WorkflowHookSystem."""

from unittest.mock import AsyncMock

import pytest

from the_agents_playbook.workflows.hooks import (
    ON_STEP_FAILURE,
    POST_STEP_EXECUTE,
    PRE_STEP_EXECUTE,
    WorkflowHookSystem,
)


class TestWorkflowHookSystem:
    @pytest.fixture
    def hooks(self):
        return WorkflowHookSystem()

    async def test_register_and_emit(self, hooks):
        handler = AsyncMock()
        hooks.on("pre_run", handler)
        await hooks.emit("pre_run", step_id="plan")
        handler.assert_called_once_with(step_id="plan")

    async def test_multiple_handlers(self, hooks):
        h1 = AsyncMock()
        h2 = AsyncMock()
        hooks.on("post_run", h1)
        hooks.on("post_run", h2)
        await hooks.emit("post_run", result="ok")
        h1.assert_called_once()
        h2.assert_called_once()

    async def test_emit_unknown_no_error(self, hooks):
        await hooks.emit("nonexistent")

    async def test_off_specific(self, hooks):
        h = AsyncMock()
        hooks.on("test", h)
        hooks.off("test", h)
        assert hooks.handlers("test") == []

    async def test_off_all(self, hooks):
        hooks.on("test", AsyncMock())
        hooks.on("test", AsyncMock())
        hooks.off("test")
        assert hooks.handlers("test") == []

    async def test_clear(self, hooks):
        hooks.on("a", AsyncMock())
        hooks.on("b", AsyncMock())
        hooks.clear()
        assert hooks.handlers("a") == []
        assert hooks.handlers("b") == []

    async def test_error_isolation(self, hooks):
        bad = AsyncMock(side_effect=RuntimeError("oops"))
        good = AsyncMock()
        hooks.on("test", bad)
        hooks.on("test", good)
        await hooks.emit("test")
        good.assert_called_once()

    async def test_standard_constants(self):
        assert PRE_STEP_EXECUTE == "pre_step_execute"
        assert POST_STEP_EXECUTE == "post_step_execute"
        assert ON_STEP_FAILURE == "on_step_failure"
