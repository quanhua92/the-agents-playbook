"""05-kv-cache-demo.py — Show how static-first ordering maximizes KV cache hits.

When the LLM processes a system prompt, it caches the key-value (KV) pairs
for the transformer attention layers. If the system prompt starts with
unchanged static content, the KV cache can be reused across turns — only
the dynamic suffix needs recomputation.

This demo simulates two consecutive turns and shows how much of the prompt
stays cached vs. needs recomputation.
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

        # Static content — rarely changes
        soul_path = tmpdir / "SOUL.md"
        soul_path.write_text("""# Agent Identity

You are a helpful coding assistant.
Rules: be precise, explain reasoning, use tools when needed.""")

        soul = PromptTemplate(soul_path)

        # === Turn 1 ===

        builder1 = ContextBuilder()
        builder1.add_static(soul.render())
        builder1.add_semi_stable(ContextLayer(
            name="session_context",
            content="Session started. User is working on a Python project.",
            priority=LayerPriority.SEMI_STABLE,
        ))
        builder1.add_dynamic(inject_date())
        builder1.add_dynamic(inject_cwd())
        builder1.add_dynamic(ContextLayer(
            name="working_files",
            content="Active files: main.py, utils.py",
            priority=LayerPriority.DYNAMIC,
        ))

        report1 = builder1.build_report()
        static_tokens_1 = sum(
            l["tokens"] for l in report1["layer_breakdown"]
            if l["priority"] == "STATIC"
        )
        dynamic_tokens_1 = sum(
            l["tokens"] for l in report1["layer_breakdown"]
            if l["priority"] == "DYNAMIC"
        )
        semi_tokens_1 = sum(
            l["tokens"] for l in report1["layer_breakdown"]
            if l["priority"] == "SEMI_STABLE"
        )

        print("=== Turn 1 ===")
        for info in report1["layer_breakdown"]:
            status = "cached" if info["priority"] == "STATIC" else "computed"
            print(f"  [{status:8s}] {info['priority']:12s} {info['name']:20s} ~{info['tokens']} tokens")
        print(f"  Total: ~{report1['total_tokens']} tokens")
        print()

        # === Turn 2 — static content unchanged, only dynamic changes ===

        builder2 = ContextBuilder()
        builder2.add_static(soul.render())  # Same as turn 1 → cached!
        builder2.add_semi_stable(ContextLayer(
            name="session_context",
            content="Session started. User is working on a Python project.",
            priority=LayerPriority.SEMI_STABLE,
        ))
        builder2.add_dynamic(inject_date())  # May have changed
        builder2.add_dynamic(inject_cwd())    # Same directory
        builder2.add_dynamic(ContextLayer(
            name="working_files",
            content="Active files: main.py, utils.py, test_main.py",  # Updated
            priority=LayerPriority.DYNAMIC,
        ))

        report2 = builder2.build_report()

        print("=== Turn 2 ===")
        for info in report2["layer_breakdown"]:
            if info["priority"] == "STATIC":
                status = "CACHED ✓"
            elif info["priority"] == "SEMI_STABLE":
                status = "cached ✓"
            else:
                status = "recomputed"
            print(f"  [{status:12s}] {info['priority']:12s} {info['name']:20s} ~{info['tokens']} tokens")
        print(f"  Total: ~{report2['total_tokens']} tokens")
        print()

        # === Summary ===

        static_pct = (static_tokens_1 / report2["total_tokens"] * 100) if report2["total_tokens"] else 0
        cached_total = static_tokens_1 + semi_tokens_1
        cached_pct = (cached_total / report2["total_tokens"] * 100) if report2["total_tokens"] else 0

        print("=== KV Cache Analysis ===")
        print(f"  Static tokens (always cached):   ~{static_tokens_1} ({static_pct:.0f}%)")
        print(f"  Semi-stable tokens (cached):     ~{semi_tokens_1}")
        print(f"  Total cached:                    ~{cached_total} ({cached_pct:.0f}%)")
        print(f"  Dynamic tokens (recomputed):     ~{dynamic_tokens_1}")
        print()
        print("  By placing static content FIRST, the LLM's KV cache")
        print("  reuses the prefix across turns — saving compute and latency.")


if __name__ == "__main__":
    asyncio.run(main())
