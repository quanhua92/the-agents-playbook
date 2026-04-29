"""Tests for context/templates.py — PromptTemplate."""

from pathlib import Path

import pytest

from the_agents_playbook.context.templates import PromptTemplate
from the_agents_playbook.context.layers import LayerPriority


@pytest.fixture
def template_dir(tmp_path: Path) -> Path:
    return tmp_path


def _write_template(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


class TestPromptTemplate:
    def test_load_from_file(self, template_dir: Path):
        path = _write_template(template_dir / "SOUL.md", "# Soul\nYou are {{name}}.")
        t = PromptTemplate(path)
        assert t.path.name == "SOUL.md"
        assert "{{name}}" in t.raw

    def test_file_not_found(self, template_dir: Path):
        with pytest.raises(FileNotFoundError, match="Template not found"):
            PromptTemplate(template_dir / "missing.md")

    def test_variables_extraction(self, template_dir: Path):
        path = _write_template(
            template_dir / "tpl.md",
            "Hello {{name}}, you are {{role}}. {{name}} appears twice.",
        )
        t = PromptTemplate(path)
        vars = t.variables()
        assert vars == ["name", "role"]  # deduplicated, order preserved

    def test_variables_no_placeholders(self, template_dir: Path):
        path = _write_template(template_dir / "plain.md", "No variables here.")
        t = PromptTemplate(path)
        assert t.variables() == []

    def test_render_replaces_variables(self, template_dir: Path):
        path = _write_template(template_dir / "tpl.md", "Hi {{name}}!")
        t = PromptTemplate(path)
        layer = t.render(name="Alice")
        assert layer.content == "Hi Alice!"

    def test_render_missing_variables_preserved(self, template_dir: Path):
        path = _write_template(template_dir / "tpl.md", "Hi {{name}}, role is {{role}}.")
        t = PromptTemplate(path)
        layer = t.render(name="Alice")
        assert layer.content == "Hi Alice, role is {{role}}."

    def test_render_returns_static_layer(self, template_dir: Path):
        path = _write_template(template_dir / "tpl.md", "content")
        t = PromptTemplate(path)
        layer = t.render()
        assert layer.priority == LayerPriority.STATIC
        assert layer.name == "tpl"
        assert layer.content == "content"

    def test_render_with_defaults(self, template_dir: Path):
        path = _write_template(
            template_dir / "tpl.md",
            "{{lang}} expert, level {{level}}",
        )
        t = PromptTemplate(path)
        layer = t.render_with_defaults(
            defaults={"lang": "Python", "level": "junior"},
            level="senior",
        )
        assert layer.content == "Python expert, level senior"

    def test_render_defaults_only(self, template_dir: Path):
        path = _write_template(template_dir / "tpl.md", "{{a}} {{b}}")
        t = PromptTemplate(path)
        layer = t.render_with_defaults(defaults={"a": "1", "b": "2"})
        assert layer.content == "1 2"

    def test_raw_property(self, template_dir: Path):
        content = "# Title\nBody with {{var}}"
        path = _write_template(template_dir / "tpl.md", content)
        t = PromptTemplate(path)
        assert t.raw == content

    def test_render_multiple_variables(self, template_dir: Path):
        path = _write_template(
            template_dir / "tpl.md",
            "{{a}}-{{b}}-{{c}}-{{a}}",
        )
        t = PromptTemplate(path)
        layer = t.render(a="X", b="Y", c="Z")
        assert layer.content == "X-Y-Z-X"
