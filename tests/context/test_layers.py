"""Tests for context/layers.py — ContextLayer and LayerPriority."""

import pytest

from the_agents_playbook.context.layers import ContextLayer, LayerPriority


# ---------------------------------------------------------------------------
# LayerPriority
# ---------------------------------------------------------------------------


def test_priority_ordering():
    assert LayerPriority.STATIC < LayerPriority.SEMI_STABLE < LayerPriority.DYNAMIC


def test_priority_values():
    assert LayerPriority.STATIC == 0
    assert LayerPriority.SEMI_STABLE == 1
    assert LayerPriority.DYNAMIC == 2


# ---------------------------------------------------------------------------
# ContextLayer creation
# ---------------------------------------------------------------------------


def test_layer_defaults():
    layer = ContextLayer(name="test", content="hello")
    assert layer.name == "test"
    assert layer.content == "hello"
    assert layer.priority == LayerPriority.STATIC
    assert layer.order == 0
    assert layer.metadata == {}


def test_layer_explicit_fields():
    layer = ContextLayer(
        name="rules",
        content="Be precise",
        priority=LayerPriority.DYNAMIC,
        order=5,
        metadata={"source": "git"},
    )
    assert layer.priority == LayerPriority.DYNAMIC
    assert layer.order == 5
    assert layer.metadata == {"source": "git"}


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


def test_sort_by_priority():
    static = ContextLayer(name="s", content="", priority=LayerPriority.STATIC)
    dynamic = ContextLayer(name="d", content="", priority=LayerPriority.DYNAMIC)
    semi = ContextLayer(name="ss", content="", priority=LayerPriority.SEMI_STABLE)

    result = sorted([dynamic, static, semi])
    assert result[0].name == "s"
    assert result[1].name == "ss"
    assert result[2].name == "d"


def test_sort_by_order_within_priority():
    a = ContextLayer(name="a", content="", priority=LayerPriority.STATIC, order=2)
    b = ContextLayer(name="b", content="", priority=LayerPriority.STATIC, order=0)
    c = ContextLayer(name="c", content="", priority=LayerPriority.STATIC, order=1)

    result = sorted([a, b, c])
    assert [l.name for l in result] == ["b", "c", "a"]


def test_sort_mixed_priority_and_order():
    layers = [
        ContextLayer(name="d1", content="", priority=LayerPriority.DYNAMIC, order=0),
        ContextLayer(name="s1", content="", priority=LayerPriority.STATIC, order=1),
        ContextLayer(name="ss1", content="", priority=LayerPriority.SEMI_STABLE, order=1),
        ContextLayer(name="s0", content="", priority=LayerPriority.STATIC, order=0),
        ContextLayer(name="d0", content="", priority=LayerPriority.DYNAMIC, order=0),
        ContextLayer(name="ss0", content="", priority=LayerPriority.SEMI_STABLE, order=0),
    ]
    result = sorted(layers)
    names = [l.name for l in result]
    assert names == ["s0", "s1", "ss0", "ss1", "d1", "d0"]


# ---------------------------------------------------------------------------
# Comparison operators
# ---------------------------------------------------------------------------


def test_lt_operator():
    static = ContextLayer(name="s", content="", priority=LayerPriority.STATIC)
    dynamic = ContextLayer(name="d", content="", priority=LayerPriority.DYNAMIC)
    assert static < dynamic
    assert not (dynamic < static)


def test_lt_same_priority_uses_order():
    a = ContextLayer(name="a", content="", priority=LayerPriority.STATIC, order=0)
    b = ContextLayer(name="b", content="", priority=LayerPriority.STATIC, order=1)
    assert a < b
    assert not (b < a)


def test_lt_returns_not_implemented_for_other_types():
    layer = ContextLayer(name="a", content="")
    assert layer.__lt__("not a layer") is NotImplemented


def test_lt_same_priority_same_order():
    a = ContextLayer(name="a", content="", priority=LayerPriority.STATIC, order=0)
    b = ContextLayer(name="b", content="", priority=LayerPriority.STATIC, order=0)
    assert not (a < b)
    assert not (b < a)
