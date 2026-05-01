"""06-guarded-agent.py — Full agent with permissions, hooks, and ask-user.

Combines all guardrails into a complete example: an agent that checks
permissions before tool calls, logs every decision via hooks, and can
ask the user for clarification when uncertain.

Set MOCK_ONLY=true in .env to run without an API key (uses simulation).
"""

import asyncio

from the_agents_playbook import settings
from the_agents_playbook.guardrails import (
    ON_TOOL_CALL,
    ON_TOOL_RESULT,
    ON_TURN_START,
    HookSystem,
    PermissionMiddleware,
    RiskLevel,
    SilentPrompter,
)
from the_agents_playbook.guardrails.ask_user import AskUserQuestion
from the_agents_playbook.tools import Tool, ToolResult, ToolRegistry


class GrepTool(Tool):
    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return "Search for a pattern in files."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
        }

    async def execute(self, pattern: str = "", path: str = "", **kw) -> ToolResult:
        return ToolResult(output=f"Found '{pattern}' in {path}:42")


class EditTool(Tool):
    @property
    def name(self) -> str:
        return "edit"

    @property
    def description(self) -> str:
        return "Edit a file."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string"},
                "new": {"type": "string"},
            },
        }

    async def execute(self, **kw) -> ToolResult:
        return ToolResult(output="File edited.")


async def run_with_real_agent():
    """Run the guarded agent with a real LLM provider."""
    from the_agents_playbook.loop import Agent, AgentConfig
    from the_agents_playbook.providers import OpenAIProvider

    provider = OpenAIProvider()

    # --- Set up tools with risk annotations ---

    registry = ToolRegistry()
    registry.register(GrepTool())
    registry.register(EditTool())
    registry.register(AskUserQuestion(prompter=SilentPrompter()))

    middleware = PermissionMiddleware()
    middleware.annotate("grep", RiskLevel.READ_ONLY)
    middleware.annotate("edit", RiskLevel.WORKSPACE_WRITE)
    middleware.annotate("ask_user", RiskLevel.READ_ONLY)

    print(f"Tools: {registry.list_tools()}")
    print()

    # --- Set up hooks for observability ---

    hooks = HookSystem()
    audit_log: list[dict] = []

    async def audit(event: str, **kw):
        audit_log.append({"event": event, **kw})

    hooks.on(ON_TURN_START, lambda **kw: audit("turn_start", **kw))
    hooks.on(ON_TOOL_CALL, lambda **kw: audit("tool_call", **kw))
    hooks.on(ON_TOOL_RESULT, lambda **kw: audit("tool_result", **kw))

    # --- Run the guarded agent ---

    agent = Agent(
        provider=provider,
        registry=registry,
        config=AgentConfig(max_tool_iterations=5),
    )

    print("=== Guarded Agent (real LLM) ===")
    async for event in agent.run("Find the typo in README.md and fix it"):
        if event.type == "status":
            print(f"  [STATUS] {event.data['message']}")
        elif event.type == "tool_call":
            risk = middleware.get_risk(event.data["tool_name"])
            needs_approval = middleware.should_prompt(event.data["tool_name"])
            print(
                f"  [TOOL]   {event.data['tool_name']}({event.data['arguments']}) [risk={risk.value}, needs_approval={needs_approval}]"
            )
        elif event.type == "tool_result":
            status = "ERROR" if event.data["error"] else "OK"
            print(f"  [RESULT] [{status}] {event.data['output']}")
        elif event.type == "text":
            print(f"  [TEXT]   {event.data['text']}")
        elif event.type == "error":
            print(f"  [ERROR]  {event.data['message']}")

    # --- Show audit log ---

    print(f"\n=== Audit Log ({len(audit_log)} entries) ===")
    for entry in audit_log:
        print(f"  [{entry['event']:12s}] ", end="")
        for k, v in entry.items():
            if k != "event":
                print(f"{k}={v} ", end="")
        print()

    await agent.close()


async def run_simulation():
    """Run the guarded agent simulation (mock mode, no API key needed)."""
    # --- Set up tools with risk annotations ---

    registry = ToolRegistry()
    registry.register(GrepTool())
    registry.register(EditTool())
    registry.register(AskUserQuestion(prompter=SilentPrompter()))

    middleware = PermissionMiddleware()
    middleware.annotate("grep", RiskLevel.READ_ONLY)
    middleware.annotate("edit", RiskLevel.WORKSPACE_WRITE)
    middleware.annotate("ask_user", RiskLevel.READ_ONLY)

    print(f"Tools: {registry.list_tools()}")
    print()

    # --- Set up hooks for observability ---

    hooks = HookSystem()
    audit_log: list[dict] = []

    async def audit(event: str, **kw):
        audit_log.append({"event": event, **kw})

    hooks.on(ON_TURN_START, lambda **kw: audit("turn_start", **kw))
    hooks.on(ON_TOOL_CALL, lambda **kw: audit("tool_call", **kw))
    hooks.on(ON_TOOL_RESULT, lambda **kw: audit("tool_result", **kw))

    # --- Simulate a guarded agent turn ---

    print("=== Simulated Guarded Turn ===")

    await hooks.emit(ON_TURN_START, prompt="Fix the typo in README.md")

    # Tool call 1: grep (READ_ONLY → auto-approve)
    tool_name = "grep"
    risk = middleware.get_risk(tool_name)
    needs_approval = middleware.should_prompt(tool_name)
    print(f"\n  Tool: {tool_name} (risk={risk.value}, needs_approval={needs_approval})")

    if needs_approval:
        print("    → Would prompt user for approval")
    else:
        print("    → Auto-approved (READ_ONLY)")

    await hooks.emit(ON_TOOL_CALL, tool_name=tool_name, risk=risk.value)
    result = await registry.dispatch(
        tool_name, {"pattern": "typo", "path": "README.md"}
    )
    await hooks.emit(
        ON_TOOL_RESULT, tool_name=tool_name, output=result.output, error=result.error
    )
    print(f"    → Result: {result.output}")

    # Tool call 2: edit (WORKSPACE_WRITE → would prompt)
    tool_name = "edit"
    risk = middleware.get_risk(tool_name)
    needs_approval = middleware.should_prompt(tool_name)
    print(f"\n  Tool: {tool_name} (risk={risk.value}, needs_approval={needs_approval})")

    if needs_approval:
        print(f"    → Would prompt: [{risk.value.upper()}] Edit README.md? (y/n)")
        # In a real agent, the prompter would be called here
    else:
        print("    → Auto-approved")

    await hooks.emit(ON_TOOL_CALL, tool_name=tool_name, risk=risk.value)
    result = await registry.dispatch(
        tool_name, {"path": "README.md", "old": "typo", "new": "fix"}
    )
    await hooks.emit(
        ON_TOOL_RESULT, tool_name=tool_name, output=result.output, error=result.error
    )
    print(f"    → Result: {result.output}")

    # --- Show audit log ---

    print(f"\n=== Audit Log ({len(audit_log)} entries) ===")
    for entry in audit_log:
        print(f"  [{entry['event']:12s}] ", end="")
        for k, v in entry.items():
            if k != "event":
                print(f"{k}={v} ", end="")
        print()


async def main():
    if settings.mock_only:
        print("NOTE: Running in MOCK_ONLY mode. Set MOCK_ONLY=false to use real LLM.\n")
        await run_simulation()
    else:
        await run_with_real_agent()


if __name__ == "__main__":
    asyncio.run(main())
