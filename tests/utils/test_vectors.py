"""Tests for utils/vectors.py — cosine_similarity and normalize."""

import numpy as np
import pytest

from the_agents_playbook.utils.vectors import cosine_similarity, normalize


def test_identical_vectors():
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-6)


def test_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_opposite_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)


def test_similar_vectors():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.9, 0.1, 0.0])
    sim = cosine_similarity(a, b)
    assert 0.9 < sim < 1.0


def test_zero_vector():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 2.0])
    assert cosine_similarity(a, b) == 0.0


def test_both_zero_vectors():
    a = np.array([0.0, 0.0])
    b = np.array([0.0, 0.0])
    assert cosine_similarity(a, b) == 0.0


def test_returns_float():
    a = np.array([1.0])
    b = np.array([2.0])
    assert isinstance(cosine_similarity(a, b), float)


def test_normalize_unit_length():
    vec = np.array([3.0, 4.0])
    result = normalize(vec)
    norm = np.linalg.norm(result)
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_normalize_preserves_direction():
    vec = np.array([3.0, 4.0])
    result = normalize(vec)
    assert result[0] / result[1] == pytest.approx(3.0 / 4.0, abs=1e-6)


def test_normalize_zero_vector():
    vec = np.array([0.0, 0.0, 0.0])
    result = normalize(vec)
    assert np.all(result == 0.0)


def test_normalize_already_unit():
    vec = np.array([1.0, 0.0])
    result = normalize(vec)
    assert np.allclose(result, vec)
