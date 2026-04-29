"""Vector math utilities for memory similarity search.

Uses numpy only -- no heavy ML dependencies.
Copied from the_agents_playbook/utils/vectors.py.
"""

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Returns a float in [-1, 1]. Identical vectors return 1.0,
    orthogonal vectors return 0.0, opposite vectors return -1.0.
    Returns 0.0 for zero vectors.
    """
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))


def normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length. Returns zero vector if input is zero."""
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return np.zeros_like(vec)
    return vec / norm
