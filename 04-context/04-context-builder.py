"""04-context-builder.py — Full builder: assemble system prompt from all layers.

Demonstrates the ContextBuilder fluent API for assembling a complete
system prompt from static, semi-stable, and dynamic layers.
"""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from the_agents_playbook.context import (
    ContextBuilder,
    ContextLayer,
    LayerPriority,
    PromptTemplate,
    inject_cwd,
    inject_date,
    inject_git_status,
)


async def main():
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a SOUL.md template
        soul_path = tmpdir / "SOUL.md"
        soul_path.write_text("""# Agent Identity

You are a helpful coding assistant.

## Rules
- Always show your work
- Use tools when needed
- Be precise""")

        # Load template
        soul = PromptTemplate(soul_path)
        soul_layer = soul.render()

        # Get dynamic metadata
        git_layer = await inject_git_status()
        date_layer = inject_date()
        cwd_layer = inject_cwd()

        # --- Build the system prompt ---

        builder = ContextBuilder(max_tokens=4096)

        builder.add_static(soul_layer)
        builder.add_static(
            ContextLayer(
                name="world_rules",
                content="Files are in Unix format. Python 3.12+.",
                priority=LayerPriority.STATIC,
                order=1,
            )
        )

        builder.add_semi_stable(
            ContextLayer(
                name="user_preferences",
                content="User prefers Python. No unnecessary comments.",
                priority=LayerPriority.SEMI_STABLE,
            )
        )

        builder.add_dynamic(git_layer)
        builder.add_dynamic(date_layer)
        builder.add_dynamic(cwd_layer)

        # --- Assemble ---

        prompt = builder.build()
        print("=== Assembled System Prompt ===")
        print(prompt)
        print()

        # --- Token budget report ---

        report = builder.build_report()
        print("=== Token Budget Report ===")
        print(f"Estimated tokens: {report['total_tokens']}")
        print(f"Budget: {report['budget']}")
        print(f"Over budget: {report['over_budget']}")
        print()
        print("Layer breakdown:")
        for layer_info in report["layer_breakdown"]:
            print(
                f"  {layer_info['priority']:12s} {layer_info['name']:20s} ~{layer_info['tokens']} tokens"
            )

        # --- Fluent API chaining ---

        prompt2 = (
            ContextBuilder()
            .add_static(soul_layer)
            .add_dynamic(date_layer)
            .add_dynamic(cwd_layer)
            .build()
        )
        assert len(prompt2) > 0
        print(f"\n✓ Fluent API produces prompt: {len(prompt2)} chars")

        # --- Token budget warning ---

        tiny_builder = ContextBuilder(max_tokens=10)
        tiny_builder.add_static(
            ContextLayer(
                name="big",
                content="x" * 100,
            )
        )
        # build() warns but doesn't raise
        tiny_builder.build()
        print("✓ Over-budget prompt still builds (logs warning)")


if __name__ == "__main__":
    asyncio.run(main())
