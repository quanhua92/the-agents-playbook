"""02-permission-middleware.py — Intercept tool calls, prompt user for approval.

PermissionMiddleware sits between the agent and tool execution. It checks
whether a tool call needs user approval based on its risk annotation.
"""

from the_agents_playbook.guardrails import PermissionMiddleware, RiskLevel


def main():
    # --- Set up middleware with custom auto-approve policy ---

    middleware = PermissionMiddleware(
        auto_approve={RiskLevel.READ_ONLY},
    )

    # Annotate tools
    middleware.annotate("read_file", RiskLevel.READ_ONLY)
    middleware.annotate("write_file", RiskLevel.WORKSPACE_WRITE)
    middleware.annotate("delete", RiskLevel.DANGER)
    middleware.annotate("deploy", RiskLevel.DANGER)

    # --- Check permissions synchronously ---

    print("=== Sync Permission Checks ===")
    checks = [
        ("read_file", {"path": "main.py"}),
        ("write_file", {"path": "main.py", "content": "print('hi')"}),
        ("delete", {"path": "old.py"}),
        ("deploy", {"env": "production"}),
    ]

    for tool_name, args in checks:
        approved = middleware.check_sync(tool_name)
        risk = middleware.get_risk(tool_name)
        status = "AUTO-APPROVED" if approved else "NEEDS APPROVAL"
        print(f"  {tool_name:12s} {status:20s} (risk={risk.value})")
    print()

    # --- Different auto-approve policies ---

    print("=== Policy Comparison ===")
    policies = [
        ("Strict", {RiskLevel.READ_ONLY}),
        ("Moderate", {RiskLevel.READ_ONLY, RiskLevel.WORKSPACE_WRITE}),
        (
            "Permissive",
            {RiskLevel.READ_ONLY, RiskLevel.WORKSPACE_WRITE, RiskLevel.DANGER},
        ),
    ]

    for policy_name, auto_set in policies:
        mw = PermissionMiddleware(auto_approve=auto_set)
        mw.annotate("read_file", RiskLevel.READ_ONLY)
        mw.annotate("write_file", RiskLevel.WORKSPACE_WRITE)
        mw.annotate("delete", RiskLevel.DANGER)

        line = f"  {policy_name:12s}: "
        for tool in ["read_file", "write_file", "delete"]:
            ok = mw.check_sync(tool)
            line += f"{tool}={'✓' if ok else '✗'} "
        print(line)


if __name__ == "__main__":
    main()
