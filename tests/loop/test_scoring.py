"""Tests for loop.scoring — Shannon entropy and tool scoring."""

import math

import pytest

from the_agents_playbook.loop.scoring import score_tools, shannon_entropy


class TestShannonEntropy:
    def test_empty_list(self):
        assert shannon_entropy([]) == 0.0

    def test_single_element(self):
        assert shannon_entropy([1.0]) == 0.0

    def test_certain_distribution(self):
        """One tool with all probability = zero uncertainty."""
        assert shannon_entropy([1.0, 0.0, 0.0]) == 0.0

    def test_uniform_binary(self):
        """Two equally likely options: H = log2(2) = 1.0 bit."""
        result = shannon_entropy([0.5, 0.5])
        assert abs(result - 1.0) < 1e-9

    def test_uniform_three(self):
        """Three equally likely options: H = log2(3) ≈ 1.585 bits."""
        result = shannon_entropy([1.0 / 3, 1.0 / 3, 1.0 / 3])
        assert abs(result - math.log2(3)) < 1e-9

    def test_skewed_distribution(self):
        """Heavily skewed: low entropy."""
        result = shannon_entropy([0.9, 0.05, 0.05])
        assert result < 0.6  # Much less than log2(3)

    def test_ignores_zero_probabilities(self):
        """Zero probabilities don't contribute to entropy (by definition)."""
        assert shannon_entropy([0.5, 0.5, 0.0, 0.0]) == pytest.approx(1.0)

    def test_maximum_entropy(self):
        """Maximum entropy for N items is log2(N)."""
        n = 8
        p = 1.0 / n
        result = shannon_entropy([p] * n)
        assert abs(result - math.log2(n)) < 1e-9


class TestScoreTools:
    def test_empty_dict(self):
        assert score_tools({}) == 0.0

    def test_all_zero_scores(self):
        assert score_tools({"a": 0.0, "b": 0.0}) == 0.0

    def test_single_tool(self):
        assert score_tools({"shell": 1.0}) == 0.0

    def test_two_tools_equal_scores(self):
        """Equal scores → uniform distribution → 1 bit entropy."""
        result = score_tools({"shell": 0.5, "search": 0.5})
        assert abs(result - 1.0) < 1e-9

    def test_many_tools_uniform(self):
        """N tools with equal scores → log2(N) bits."""
        n = 5
        scores = {f"tool_{i}": 1.0 for i in range(n)}
        result = score_tools(scores)
        assert abs(result - math.log2(n)) < 1e-9

    def test_skewed_scores(self):
        """One dominant tool → low entropy."""
        result = score_tools({"shell": 0.9, "search": 0.05, "memory": 0.05})
        assert result < 0.6

    def test_scores_not_normalized(self):
        """Scores are normalized internally before computing entropy."""
        result1 = score_tools({"a": 3.0, "b": 3.0})
        result2 = score_tools({"a": 1.0, "b": 1.0})
        assert abs(result1 - result2) < 1e-9
