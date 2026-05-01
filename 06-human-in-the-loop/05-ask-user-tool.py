"""05-ask-user-tool.py — Agent asks user for clarification mid-loop.

Instead of hallucinating an answer, the agent can break character and
ask the user directly using the AskUserQuestion tool.
"""

import asyncio

from the_agents_playbook.guardrails import AskUserQuestion
from the_agents_playbook.guardrails.prompter import DenyAllPrompter


async def main():
    tool = AskUserQuestion()

    # --- Tool protocol ---

    print("=== AskUserQuestion Protocol ===")
    print(f"  Name:        {tool.name}")
    print(f"  Description: {tool.description[:60]}...")
    print(f"  Parameters:  required={tool.parameters['required']}")
    print(f"  Properties:  {list(tool.parameters['properties'].keys())}")
    print()

    # --- Ask a question (SilentPrompter auto-approves) ---

    print("=== Ask Question (SilentPrompter) ===")
    result = await tool.execute(question="What is your name?")
    print(f"  Output: {result.output}")
    print(f"  Error:  {result.error}")
    print()

    # --- Ask with options ---

    print("=== Ask with Options ===")
    result = await tool.execute(
        question="Which file should I edit?",
        options=["auth.py", "user.py", "config.py"],
    )
    print(f"  Output: {result.output}")
    print()

    # --- User declines (DenyAllPrompter) ---

    print("=== User Declines (DenyAllPrompter) ===")
    deny_tool = AskUserQuestion(prompter=DenyAllPrompter())
    result = await deny_tool.execute(question="Should I deploy to production?")
    print(f"  Output: {result.output}")
    print(f"  Error:  {result.error}")
    print()

    # --- Register in a ToolRegistry ---

    print("=== Register in ToolRegistry ===")
    from the_agents_playbook.tools import ToolRegistry

    registry = ToolRegistry()
    registry.register(tool)
    specs = registry.get_specs()
    print(f"  Registered tools: {registry.list_tools()}")
    print(f"  Specs count: {len(specs)}")
    print(f"  First spec name: {specs[0].name}")


if __name__ == "__main__":
    asyncio.run(main())
