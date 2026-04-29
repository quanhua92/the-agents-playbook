"""03-shannon-entropy.py — Score tool selections, detect uncertainty.

Shannon entropy measures how uncertain the agent is about which tool
to use. High entropy = the agent is unsure. Low entropy = clear choice.
The agent can use this to decide: execute the best tool, ask the user,
or continue thinking.
"""

import math

from the_agents_playbook.loop import score_tools, shannon_entropy


def main():
    # --- Basic entropy calculations ---

    print("=== Shannon Entropy ===")
    print(f"Certain (one option):      H = {shannon_entropy([1.0]):.4f} bits")
    print(f"Coin flip (50/50):         H = {shannon_entropy([0.5, 0.5]):.4f} bits")
    print(f"Three-way uniform:         H = {shannon_entropy([1/3, 1/3, 1/3]):.4f} bits")
    print(f"Eight-way uniform:         H = {shannon_entropy([1/8] * 8):.4f} bits (log2(8)=3.0)")
    print(f"Skewed (90/5/5):          H = {shannon_entropy([0.9, 0.05, 0.05]):.4f} bits")
    print(f"Very skewed (99/0.5/0.5): H = {shannon_entropy([0.99, 0.005, 0.005]):.4f} bits")
    print()

    # --- Tool scoring scenarios ---

    print("=== Tool Selection Scoring ===")

    # Scenario 1: Clear choice (one dominant tool)
    scores_1 = {"shell": 0.8, "search": 0.1, "memory": 0.1}
    entropy_1 = score_tools(scores_1)
    print(f"Clear choice:  {scores_1}")
    print(f"  → Entropy: {entropy_1:.4f} bits  (low = confident)")
    print()

    # Scenario 2: Two equally good tools
    scores_2 = {"shell": 0.5, "search": 0.5}
    entropy_2 = score_tools(scores_2)
    print(f"Two-way tie:   {scores_2}")
    print(f"  → Entropy: {entropy_2:.4f} bits  (high = uncertain)")
    print()

    # Scenario 3: Many tools, all equal
    scores_3 = {f"tool_{i}": 0.2 for i in range(5)}
    entropy_3 = score_tools(scores_3)
    print(f"Five-way tie:  {len(scores_3)} tools, all 0.2")
    print(f"  → Entropy: {entropy_3:.4f} bits  (max = log2(5) = {math.log2(5):.4f})")
    print()

    # Scenario 4: Progressive uncertainty
    print("=== Progressive Uncertainty ===")
    for n_tools in [2, 3, 4, 5, 8, 16]:
        scores = {f"tool_{i}": 1.0 for i in range(n_tools)}
        entropy = score_tools(scores)
        max_entropy = math.log2(n_tools)
        bar = "#" * int(entropy / max_entropy * 40) if max_entropy > 0 else ""
        print(f"  {n_tools:2d} tools: H={entropy:.3f}/{max_entropy:.3f} bits  {bar}")
    print()

    # --- Routing decision example ---

    print("=== Routing Decisions ===")
    threshold = 1.5

    for label, scores in [
        ("Clear winner", {"shell": 0.9, "search": 0.05, "memory": 0.05}),
        ("Moderate", {"shell": 0.5, "search": 0.3, "memory": 0.2}),
        ("Very uncertain", {"shell": 0.34, "search": 0.33, "memory": 0.33}),
    ]:
        entropy = score_tools(scores)
        decision = "EXECUTE" if entropy < threshold else "ASK USER"
        print(f"  {label:20s} H={entropy:.3f}  threshold={threshold}  → {decision}")


if __name__ == "__main__":
    main()
