"""02-prompt-templates.py — Load SOUL.md, USER.md templates.

Demonstrates the PromptTemplate class for loading markdown files
from disk with {{variable}} substitution.
"""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from the_agents_playbook.context import PromptTemplate


async def main():
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # --- Create template files ---

        # SOUL.md — defines agent personality
        soul_path = tmpdir / "SOUL.md"
        soul_path.write_text("""# Agent Identity

You are {{name}}, {{role}}.

## Principles
- Be precise and concise
- Always explain your reasoning step by step
- Ask clarifying questions when uncertain

## Style
- Use code examples when helpful
- Prefer straightforward solutions over clever ones
""")

        # USER.md — user preferences
        user_path = tmpdir / "USER.md"
        user_path.write_text("""# User Preferences

- Preferred language: {{language}}
- Experience level: {{level}}
- Communication style: {{style}}
""")

        # AGENTS.md — agent capabilities
        agents_path = tmpdir / "AGENTS.md"
        agents_path.write_text("""# Agent Capabilities

This agent has access to the following tools:
- Shell execution (sandboxed)
- File I/O (workspace-restricted)
- Web search
- Memory (short and long-term)

Current environment: {{env}}
""")

        # --- Load and inspect templates ---

        soul = PromptTemplate(soul_path)
        print(f"Template: {soul.path.name}")
        print(f"Variables found: {soul.variables()}")
        print()

        # --- Render with variables ---

        soul_layer = soul.render(name="Claude", role="a coding assistant")
        print("=== Rendered SOUL.md ===")
        print(soul_layer.content)
        print(f"Priority: {soul_layer.priority.name}")

        # --- Render with defaults ---

        user = PromptTemplate(user_path)
        user_defaults = {
            "language": "Python",
            "level": "intermediate",
            "style": "direct",
        }

        # Defaults only
        user_layer = user.render_with_defaults(defaults=user_defaults)
        print("\n=== Rendered USER.md (defaults) ===")
        print(user_layer.content)

        # Override specific values
        user_layer = user.render_with_defaults(
            defaults=user_defaults,
            level="advanced",
            style="friendly",
        )
        print("\n=== Rendered USER.md (with overrides) ===")
        print(user_layer.content)

        # --- Missing variables stay as-is ---

        partial = soul.render(name="Assistant")
        # {{role}} is not provided, so it remains in the output
        assert "{{role}}" in partial.content
        print(f"\n✓ Missing variables preserved: '{{role}}' still in output")

        # --- Template not found raises error ---
        try:
            PromptTemplate(Path("/nonexistent/template.md"))
            assert False, "Should have raised"
        except FileNotFoundError as e:
            print(f"✓ Missing template raises: {e}")


if __name__ == "__main__":
    asyncio.run(main())
