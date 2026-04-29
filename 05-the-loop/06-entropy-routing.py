"""06-entropy-routing.py — Route to clarification when entropy is high.

When the agent is uncertain about which tool to use (high entropy),
it should ask the user for clarification instead of guessing. This
prevents hallucinated tool calls and wasted tokens.

This example demonstrates the decision boundary between "confident
enough to act" and "should ask the user."
"""

from the_agents_playbook.loop import AgentConfig, score_tools, shannon_entropy


def main():
    # --- Define routing thresholds ---

    low_threshold = 0.5    # Very confident — auto-execute
    mid_threshold = 1.5    # Moderate — execute but monitor
    high_threshold = 2.0   # Uncertain — ask user

    print("=== Entropy Routing Thresholds ===")
    print(f"  Low:    H < {low_threshold}  → Auto-execute (confident)")
    print(f"  Mid:    {low_threshold} ≤ H < {high_threshold}  → Execute with monitoring")
    print(f"  High:   H ≥ {high_threshold}  → Ask user for clarification")
    print()

    # --- Test scenarios ---

    scenarios = [
        ("One clear tool", {"shell": 0.95, "search": 0.03, "memory": 0.02}),
        ("Two good options", {"shell": 0.45, "search": 0.45, "memory": 0.10}),
        ("Three-way tie", {"shell": 0.34, "search": 0.33, "memory": 0.33}),
        ("Complete uncertainty", {f"tool_{i}": 0.1 for i in range(10)}),
    ]

    print("=== Routing Decisions ===")
    for label, scores in scenarios:
        entropy = score_tools(scores)

        if entropy < low_threshold:
            route = "AUTO-EXECUTE"
            tool = max(scores, key=scores.get)
            detail = f"(tool: {tool})"
        elif entropy < high_threshold:
            route = "EXECUTE + MONITOR"
            tool = max(scores, key=scores.get)
            detail = f"(tool: {tool}, watch for errors)"
        else:
            route = "ASK USER"
            detail = "(too uncertain, get clarification)"

        bar_len = int(min(entropy / 3.0, 1.0) * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {label:25s} H={entropy:.3f} [{bar}] → {route} {detail}")

    print()

    # --- Adaptive threshold example ---

    print("=== Adaptive Thresholds ===")
    print("As the agent gains experience (success history),")
    print("it can lower the threshold and act more autonomously.")
    print()

    for step in range(1, 6):
        # Simulate increasing confidence over time
        threshold = 2.0 - (step * 0.2)
        scores = {"shell": 0.5, "search": 0.3, "memory": 0.2}
        entropy = score_tools(scores)
        route = "EXECUTE" if entropy < threshold else "ASK"

        print(
            f"  Step {step}: threshold={threshold:.1f}, "
            f"entropy={entropy:.3f} → {route}"
        )

    print()

    # --- AgentConfig integration ---

    print("=== AgentConfig Integration ===")
    config = AgentConfig(entropy_threshold=1.5)
    print(f"  entropy_threshold: {config.entropy_threshold}")
    print(f"  max_chain_length:  {config.max_chain_length}")
    print()
    print("  The Agent uses entropy_threshold to decide when to ask")
    print("  the user vs. when to proceed with tool execution.")


if __name__ == "__main__":
    main()
