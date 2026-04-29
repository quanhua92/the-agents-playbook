"""Shannon entropy scoring for measuring tool selection uncertainty."""

import math


def shannon_entropy(probabilities: list[float]) -> float:
    """Compute Shannon entropy H(p) = -Σ p_i * log2(p_i).

    Returns bits of uncertainty. Higher entropy means more uncertainty
    about which tool to select. A uniform distribution over N tools
    gives H = log2(N), while a single certain tool gives H = 0.

    Args:
        probabilities: List of probability values (must sum to ≤ 1.0).

    Returns:
        Entropy in bits. Returns 0.0 for empty or single-element lists.
    """
    if len(probabilities) <= 1:
        return 0.0

    return -sum(p * math.log2(p) for p in probabilities if p > 0)


def score_tools(tool_scores: dict[str, float]) -> float:
    """Compute entropy over tool selection scores.

    Converts raw scores to probabilities and returns the Shannon entropy.
    Higher entropy means the agent is more uncertain about which tool
    to use.

    Args:
        tool_scores: Mapping of tool name to relevance score.

    Returns:
        Entropy in bits. Returns 0.0 if all scores are zero.
    """
    if not tool_scores:
        return 0.0

    total = sum(tool_scores.values())
    if total == 0:
        return 0.0

    probs = [s / total for s in tool_scores.values()]
    return shannon_entropy(probs)
