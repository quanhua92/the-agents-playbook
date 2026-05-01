"""03-prompter-abc.py — Terminal vs. Silent prompter, custom implementations.

The Prompter ABC decouples HOW the agent asks from THAT it asks.
Different implementations handle terminal, web, or API contexts.
"""

import asyncio
from unittest.mock import AsyncMock

from the_agents_playbook.guardrails import (
    DenyAllPrompter,
    RiskLevel,
    SilentPrompter,
    TerminalPrompter,
)


async def main():
    # --- SilentPrompter: auto-approve everything ---

    print("=== SilentPrompter ===")
    silent = SilentPrompter()
    for risk in RiskLevel:
        result = await silent.confirm(f"Action at {risk.value}?", risk=risk)
        print(f"  {risk.value:20s} → {result}")
    print()

    # --- DenyAllPrompter: auto-deny everything ---

    print("=== DenyAllPrompter ===")
    deny = DenyAllPrompter()
    for risk in RiskLevel:
        result = await deny.confirm(f"Action at {risk.value}?", risk=risk)
        print(f"  {risk.value:20s} → {result}")
    print()

    # --- TerminalPrompter with mocked input ---

    print("=== TerminalPrompter (mocked) ===")
    terminal = TerminalPrompter(input_fn=AsyncMock(side_effect=["y", "n", "Y", "nope"]))
    for i, (answer, expected) in enumerate(
        [("y", True), ("n", False), ("Y", True), ("nope", False)]
    ):
        result = await terminal.confirm(f"Question {i + 1}?")
        status = "approved" if result else "denied"
        print(
            f"  Answer '{answer:5s}' → {status} (expected: {'approved' if expected else 'denied'})"
        )
    print()

    # --- Risk labels in prompts ---

    print("=== Risk Labels in Prompts ===")
    terminal = TerminalPrompter(input_fn=AsyncMock(return_value="y"))
    for risk in [RiskLevel.READ_ONLY, RiskLevel.WORKSPACE_WRITE, RiskLevel.DANGER]:
        await terminal.confirm("Do something?", risk=risk)
        prompt_arg = terminal._input_fn.call_args[0][0]
        print(f"  {prompt_arg}")


if __name__ == "__main__":
    asyncio.run(main())
