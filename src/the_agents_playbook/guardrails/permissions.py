"""Permission middleware — intercept tool calls by risk level.

Tools are annotated with a RiskLevel. The middleware intercepts dispatch
calls and prompts the user based on the risk level before allowing or
blocking execution.
"""

import logging
from enum import Enum
from typing import Any

from ..tools.protocol import Tool, ToolResult

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for tool operations.

    READ_ONLY: Auto-approve. Read files, search, query memory.
    WORKSPACE_WRITE: Prompt for approval. Create/edit files, run safe commands.
    DANGER: Require explicit confirmation. Delete, network access, deploy.
    """

    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"
    DANGER = "danger"


class RiskAnnotatedTool(Tool):
    """Wraps a Tool with a RiskLevel annotation.

    Delegates all Tool protocol methods to the wrapped tool while
    carrying the risk metadata.
    """

    def __init__(self, tool: Tool, risk: RiskLevel) -> None:
        self._tool = tool
        self._risk = risk

    @property
    def name(self) -> str:
        return self._tool.name

    @property
    def description(self) -> str:
        return self._tool.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._tool.parameters

    @property
    def risk(self) -> RiskLevel:
        return self._risk

    @property
    def inner_tool(self) -> Tool:
        return self._tool

    async def execute(self, **kwargs: Any) -> ToolResult:
        return await self._tool.execute(**kwargs)


class PermissionMiddleware:
    """Intercepts tool calls and enforces risk-based permissions.

    Usage:
        middleware = PermissionMiddleware(prompter=TerminalPrompter())
        middleware.annotate("shell", RiskLevel.DANGER)
        if await middleware.check("shell", {"command": "rm -rf /tmp"}):
            result = await registry.dispatch("shell", {"command": "rm -rf /tmp"})
    """

    def __init__(
        self,
        auto_approve: set[RiskLevel] | None = None,
    ) -> None:
        self._auto_approve = auto_approve or {RiskLevel.READ_ONLY}
        self._risk_map: dict[str, RiskLevel] = {}

    def annotate(self, tool_name: str, risk: RiskLevel) -> None:
        """Annotate a tool name with a risk level."""
        self._risk_map[tool_name] = risk
        logger.debug("Annotated tool %s with risk level %s", tool_name, risk.value)

    def get_risk(self, tool_name: str) -> RiskLevel:
        """Get the risk level for a tool. Defaults to READ_ONLY."""
        return self._risk_map.get(tool_name, RiskLevel.READ_ONLY)

    def should_prompt(self, tool_name: str) -> bool:
        """Check if a tool requires user confirmation."""
        risk = self.get_risk(tool_name)
        return risk not in self._auto_approve

    def check_sync(self, tool_name: str) -> bool:
        """Synchronously check permission (auto-approve only, no prompt).

        Returns True if the tool is in the auto-approve set.
        Returns False if user confirmation would be needed.
        """
        return not self.should_prompt(tool_name)

    def wrap_tool(self, tool: Tool, risk: RiskLevel) -> RiskAnnotatedTool:
        """Wrap a Tool instance with a risk annotation."""
        return RiskAnnotatedTool(tool, risk)
