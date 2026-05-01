"""06-token-budget.py — Manage context window capacity like a financial budget.

Context windows are finite. You must reserve tokens for the response,
system overhead, and tool output. When approaching the limit, trim
lower-priority context layers first.

This example demonstrates:
- TokenBudget: reserve, release, and track utilization
- Priority-based trimming: drop DYNAMIC layers before SEMI_STABLE before STATIC
- UsageTracker: monitor cost per model across requests
"""

from the_agents_playbook.context import TokenBudget, UsageTracker
from the_agents_playbook.context.layers import ContextLayer, LayerPriority


def print_bar(label: str, current: int, maximum: int, width: int = 40) -> None:
    """Print a labeled progress bar showing usage."""
    ratio = min(current / maximum, 1.0) if maximum > 0 else 0
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    print(f"  {label:25s} [{bar}] {current:>7,} / {maximum:,} ({ratio:.0%})")


def main():
    # === Token Budget ===
    print("=== Token Budget Management ===\n")

    # Typical GPT-4o context: 128k tokens
    budget = TokenBudget(total=128_000, reserved_for_response=4096)

    print(f"Budget: {budget}")
    print(f"Available for prompt: {budget.available:,} tokens\n")

    # Reserve context layers in priority order
    print("--- Reserving Context Layers ---\n")

    layers = [
        ("System prompt", 500, LayerPriority.STATIC),
        ("Tool definitions", 2000, LayerPriority.STATIC),
        ("User preferences", 300, LayerPriority.SEMI_STABLE),
        ("Memory recall", 800, LayerPriority.SEMI_STABLE),
        ("Git status", 200, LayerPriority.DYNAMIC),
        ("Current date/time", 50, LayerPriority.DYNAMIC),
        ("Conversation history", 45_000, LayerPriority.SEMI_STABLE),
    ]

    for name, tokens, priority in layers:
        success = budget.reserve(tokens)
        status = "OK" if success else "EXCEEDED"
        print(f"  [{priority.name:12s}] {name:25s} {tokens:>6,} tokens  -> {status}")

    print(f"\nAfter reservations: {budget}\n")

    # Try to add more than fits
    print("--- Budget Exceeded ---\n")
    big_layer = budget.reserve(100_000)
    print(
        f"  Reserve 100k more tokens: {'OK' if big_layer else 'REJECTED (not enough budget)'}"
    )
    print(f"  Budget: {budget}\n")

    # Release and try again
    print("--- Release and Retry ---\n")
    budget.release(40_000)
    print("  Released 40k tokens")
    print(f"  Budget: {budget}\n")
    big_layer = budget.reserve(100_000)
    print(f"  Reserve 100k more tokens: {'OK' if big_layer else 'REJECTED'}\n")

    # === Priority-Based Trimming ===
    print("=== Priority-Based Trimming ===\n")

    budget2 = TokenBudget(total=8_000, reserved_for_response=1000)
    context_layers = [
        ContextLayer("system", "You are a helpful agent.", LayerPriority.STATIC),
        ContextLayer("tools", "tool1: ...\ntool2: ...", LayerPriority.STATIC),
        ContextLayer(
            "memory",
            "User likes dark mode. Working on auth.",
            LayerPriority.SEMI_STABLE,
        ),
        ContextLayer(
            "git_status", "On branch main, 3 files changed", LayerPriority.DYNAMIC
        ),
        ContextLayer(
            "date", "Today is Wednesday, April 30, 2026", LayerPriority.DYNAMIC
        ),
    ]

    # Try to reserve all layers — DYNAMIC ones might not fit
    print("Reserving layers by priority (trim DYNAMIC first if over budget):\n")

    # Sort by priority (STATIC first, so it gets reserved first)
    for layer in sorted(context_layers):
        tokens = len(layer.content.split())  # rough token estimate
        success = budget2.reserve(tokens)
        status = "reserved" if success else "TRIMMED (dropped)"
        print(
            f"  [{layer.priority.name:12s}] {layer.name:15s} ~{tokens:3d} tokens  -> {status}"
        )

    print(f"\nFinal budget: {budget2}\n")

    # === Usage Tracking ===
    print("=== Usage Tracking ===\n")

    tracker = UsageTracker()

    # Simulate requests across models
    tracker.record("gpt-4o", 1500, 200, "agent_loop")
    tracker.record("gpt-4o", 3000, 500, "agent_loop")
    tracker.record("gpt-4o", 800, 100, "evaluation")
    tracker.record("gpt-4o-mini", 5000, 1000, "memory_consolidation")
    tracker.record("gpt-4o-mini", 2000, 300, "evaluation")

    print("Usage by model:")
    by_model = tracker.by_model()
    for model, data in by_model.items():
        cost = data["cost"]
        cost_str = f"${cost:.4f}" if cost is not None else "unknown"
        print(
            f"  {model:20s}  "
            f"input={data['input_tokens']:>6,}  "
            f"output={data['output_tokens']:>6,}  "
            f"requests={data['request_count']}  "
            f"cost={cost_str}"
        )

    total_in, total_out, total_all = tracker.total_tokens()
    print(f"\n  Total: {total_all:,} tokens ({total_in:,} in, {total_out:,} out)")
    print(f"  Estimated cost: ${tracker.total_cost():.4f}")


if __name__ == "__main__":
    main()
