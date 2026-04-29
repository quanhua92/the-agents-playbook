"""Tests for guardrails.permissions — RiskLevel and PermissionMiddleware."""

from unittest.mock import AsyncMock

import pytest

from the_agents_playbook.guardrails.permissions import (
    PermissionMiddleware,
    RiskAnnotatedTool,
    RiskLevel,
)
from the_agents_playbook.tools.protocol import Tool, ToolResult


class StubTool(Tool):
    @property
    def name(self) -> str:
        return "stub"

    @property
    def description(self) -> str:
        return "A stub tool."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(output="ok")


class TestRiskLevel:
    def test_values(self):
        assert RiskLevel.READ_ONLY.value == "read_only"
        assert RiskLevel.WORKSPACE_WRITE.value == "workspace_write"
        assert RiskLevel.DANGER.value == "danger"

    def test_all_members(self):
        assert len(RiskLevel) == 3


class TestRiskAnnotatedTool:
    @pytest.fixture
    def wrapped(self):
        return RiskAnnotatedTool(StubTool(), RiskLevel.DANGER)

    def test_delegates_name(self, wrapped):
        assert wrapped.name == "stub"

    def test_delegates_description(self, wrapped):
        assert wrapped.description == "A stub tool."

    def test_delegates_parameters(self, wrapped):
        assert wrapped.parameters == {"type": "object", "properties": {}}

    def test_exposes_risk(self, wrapped):
        assert wrapped.risk == RiskLevel.DANGER

    def test_exposes_inner_tool(self, wrapped):
        assert isinstance(wrapped.inner_tool, StubTool)

    async def test_delegates_execute(self, wrapped):
        result = await wrapped.execute()
        assert result.output == "ok"
        assert result.error is False


class TestPermissionMiddleware:
    def test_default_auto_approve(self):
        mw = PermissionMiddleware()
        assert RiskLevel.READ_ONLY in mw._auto_approve

    def test_custom_auto_approve(self):
        mw = PermissionMiddleware(auto_approve={RiskLevel.READ_ONLY})
        assert RiskLevel.READ_ONLY in mw._auto_approve
        assert RiskLevel.WORKSPACE_WRITE not in mw._auto_approve

    def test_annotate_tool(self):
        mw = PermissionMiddleware()
        mw.annotate("shell", RiskLevel.DANGER)
        assert mw.get_risk("shell") == RiskLevel.DANGER

    def test_default_risk_is_read_only(self):
        mw = PermissionMiddleware()
        assert mw.get_risk("unknown_tool") == RiskLevel.READ_ONLY

    def test_should_prompt_for_danger(self):
        mw = PermissionMiddleware()
        mw.annotate("rm", RiskLevel.DANGER)
        assert mw.should_prompt("rm") is True

    def test_should_not_prompt_for_read_only(self):
        mw = PermissionMiddleware()
        assert mw.should_prompt("search") is False

    def test_check_sync_auto_approved(self):
        mw = PermissionMiddleware()
        assert mw.check_sync("read_file") is True  # READ_ONLY, auto-approved

    def test_check_sync_needs_prompt(self):
        mw = PermissionMiddleware()
        mw.annotate("delete", RiskLevel.DANGER)
        assert mw.check_sync("delete") is False

    def test_wrap_tool(self):
        mw = PermissionMiddleware()
        wrapped = mw.wrap_tool(StubTool(), RiskLevel.WORKSPACE_WRITE)
        assert isinstance(wrapped, RiskAnnotatedTool)
        assert wrapped.risk == RiskLevel.WORKSPACE_WRITE
