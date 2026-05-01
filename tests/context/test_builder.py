"""Tests for context/builder.py — ContextBuilder."""

from the_agents_playbook.context.builder import ContextBuilder
from the_agents_playbook.context.layers import ContextLayer, LayerPriority


# ---------------------------------------------------------------------------
# Fluent API
# ---------------------------------------------------------------------------


def test_add_static_sets_priority():
    builder = ContextBuilder()
    layer = ContextLayer(name="rules", content="Be precise")
    builder.add_static(layer)
    assert layer.priority == LayerPriority.STATIC


def test_add_semi_stable_sets_priority():
    builder = ContextBuilder()
    layer = ContextLayer(name="memory", content="User likes Python")
    builder.add_semi_stable(layer)
    assert layer.priority == LayerPriority.SEMI_STABLE


def test_add_dynamic_sets_priority():
    builder = ContextBuilder()
    layer = ContextLayer(name="date", content="Today is Monday")
    builder.add_dynamic(layer)
    assert layer.priority == LayerPriority.DYNAMIC


def test_add_preserves_existing_priority():
    builder = ContextBuilder()
    layer = ContextLayer(name="x", content="y", priority=LayerPriority.DYNAMIC)
    builder.add(layer)
    assert layer.priority == LayerPriority.DYNAMIC


def test_fluent_returns_self():
    builder = ContextBuilder()
    result = builder.add_static(ContextLayer(name="a", content=""))
    assert result is builder


def test_clear():
    builder = ContextBuilder()
    builder.add_static(ContextLayer(name="a", content="x"))
    builder.add_dynamic(ContextLayer(name="b", content="y"))
    assert len(builder._layers) == 2
    builder.clear()
    assert len(builder._layers) == 0


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


def test_layers_sorted_by_priority():
    builder = ContextBuilder()
    builder.add_dynamic(ContextLayer(name="d", content=""))
    builder.add_static(ContextLayer(name="s", content=""))
    builder.add_semi_stable(ContextLayer(name="ss", content=""))

    layers = builder.layers
    assert [lyr.name for lyr in layers] == ["s", "ss", "d"]


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------


def test_build_assembles_content():
    builder = ContextBuilder()
    builder.add_static(ContextLayer(name="a", content="First"))
    builder.add_dynamic(ContextLayer(name="b", content="Second"))
    result = builder.build()
    assert result == "First\n\nSecond"


def test_build_empty_layers():
    builder = ContextBuilder()
    assert builder.build() == ""


def test_build_skips_empty_layers():
    builder = ContextBuilder()
    builder.add_static(ContextLayer(name="a", content="Has content"))
    builder.add_static(ContextLayer(name="b", content="   "))
    builder.add_static(ContextLayer(name="c", content=""))
    result = builder.build()
    assert result == "Has content"


def test_build_sorts_before_assembling():
    builder = ContextBuilder()
    builder.add_dynamic(ContextLayer(name="d", content="Dynamic"))
    builder.add_static(ContextLayer(name="s", content="Static"))
    result = builder.build()
    assert result.startswith("Static")
    assert result.endswith("Dynamic")


def test_build_uses_add_preserves_priority():
    builder = ContextBuilder()
    builder.add(
        ContextLayer(name="d", content="Dynamic", priority=LayerPriority.DYNAMIC)
    )
    builder.add(ContextLayer(name="s", content="Static", priority=LayerPriority.STATIC))
    result = builder.build()
    assert result.startswith("Static")


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def test_estimated_tokens():
    builder = ContextBuilder()
    # 100 chars ≈ 25 tokens
    builder.add_static(ContextLayer(name="a", content="x" * 100))
    assert builder.estimated_tokens() == 25


def test_estimated_tokens_empty():
    builder = ContextBuilder()
    assert builder.estimated_tokens() == 0


def test_token_budget_remaining():
    builder = ContextBuilder(max_tokens=100)
    builder.add_static(ContextLayer(name="a", content="x" * 100))
    assert builder.token_budget_remaining() == 75  # 100 - 25


def test_over_budget_logged_as_warning(caplog):
    builder = ContextBuilder(max_tokens=5)
    builder.add_static(ContextLayer(name="a", content="x" * 100))
    builder.build()
    assert "exceeds budget" in caplog.text


# ---------------------------------------------------------------------------
# build_report()
# ---------------------------------------------------------------------------


def test_build_report_structure():
    builder = ContextBuilder(max_tokens=100)
    builder.add_static(ContextLayer(name="rules", content="Be precise."))
    builder.add_dynamic(ContextLayer(name="date", content="2025-01-01"))

    report = builder.build_report()

    assert "total_tokens" in report
    assert "budget" in report
    assert "over_budget" in report
    assert "layer_breakdown" in report
    assert report["budget"] == 100
    assert len(report["layer_breakdown"]) == 2
    assert report["layer_breakdown"][0]["name"] == "rules"
    assert report["layer_breakdown"][0]["priority"] == "STATIC"
    assert report["layer_breakdown"][1]["name"] == "date"
    assert report["layer_breakdown"][1]["priority"] == "DYNAMIC"


def test_build_report_over_budget_flag():
    builder = ContextBuilder(max_tokens=1)
    builder.add_static(ContextLayer(name="a", content="x" * 100))
    report = builder.build_report()
    assert report["over_budget"] is True
