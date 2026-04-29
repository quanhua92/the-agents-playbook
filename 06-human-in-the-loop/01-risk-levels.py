"""01-risk-levels.py — Define RiskLevel, annotate tools with risk.

Tools are classified by danger level. READ_ONLY tools auto-approve,
WORKSPACE_WRITE tools prompt the user, and DANGER tools require
explicit confirmation.
"""

from the_agents_playbook.guardrails import PermissionMiddleware, RiskLevel, RiskAnnotatedTool
from the_agents_playbook.tools import Tool, ToolResult


class ReadFileTool(Tool):
    @property
    def name(self) -> str: return "read_file"
    @property
    def description(self) -> str: return "Read a file."
    @property
    def parameters(self) -> dict: return {"type": "object", "properties": {"path": {"type": "string"}}}
    async def execute(self, **kw) -> ToolResult: return ToolResult(output="file contents")


class WriteFileTool(Tool):
    @property
    def name(self) -> str: return "write_file"
    @property
    def description(self) -> str: return "Write a file."
    @property
    def parameters(self) -> dict: return {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}
    async def execute(self, **kw) -> ToolResult: return ToolResult(output="written")


class DeleteTool(Tool):
    @property
    def name(self) -> str: return "delete"
    @property
    def description(self) -> str: return "Delete a file."
    @property
    def parameters(self) -> dict: return {"type": "object", "properties": {"path": {"type": "string"}}}
    async def execute(self, **kw) -> ToolResult: return ToolResult(output="deleted")


def main():
    # --- Risk levels ---

    print("=== Risk Levels ===")
    for level in RiskLevel:
        print(f"  {level.name:20s} → {level.value}")
    print()

    # --- Annotate tools with risk ---

    middleware = PermissionMiddleware()
    middleware.annotate("read_file", RiskLevel.READ_ONLY)
    middleware.annotate("write_file", RiskLevel.WORKSPACE_WRITE)
    middleware.annotate("delete", RiskLevel.DANGER)

    # --- Check which tools need prompting ---

    print("=== Permission Checks ===")
    for tool_name in ["read_file", "write_file", "delete"]:
        risk = middleware.get_risk(tool_name)
        needs_prompt = middleware.should_prompt(tool_name)
        print(f"  {tool_name:12s} risk={risk.value:20s} needs_prompt={needs_prompt}")
    print()

    # --- Wrap tools with risk annotations ---

    print("=== RiskAnnotatedTool ===")
    wrapped = middleware.wrap_tool(DeleteTool(), RiskLevel.DANGER)
    print(f"  Tool name:  {wrapped.name}")
    print(f"  Risk level: {wrapped.risk.value}")
    print(f"  Inner tool: {type(wrapped.inner_tool).__name__}")


if __name__ == "__main__":
    main()
