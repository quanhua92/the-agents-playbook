"""Prompt templates — load .md files from disk and return ContextLayer instances.

Templates support {{variable}} substitution and always produce STATIC priority layers.
Common templates: SOUL.md (agent personality), USER.md (user preferences),
AGENTS.md (agent capabilities).
"""

import re
from pathlib import Path

from .layers import ContextLayer, LayerPriority

_VARIABLE_PATTERN = re.compile(r"\{\{(\w+)\}\}")


class PromptTemplate:
    """Load a markdown template file and render it with variable substitution.

    Usage:
        soul = PromptTemplate(Path("SOUL.md"))
        layer = soul.render(name="Assistant", role="coding helper")
        # layer.content has {{name}} and {{role}} replaced
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Template not found: {self._path}")
        self._raw = self._path.read_text(encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    @property
    def raw(self) -> str:
        """The raw template content before variable substitution."""
        return self._raw

    def variables(self) -> list[str]:
        """Return the list of {{variable}} names found in the template."""
        return list(dict.fromkeys(_VARIABLE_PATTERN.findall(self._raw)))

    def render(self, **variables: str) -> ContextLayer:
        """Substitute {{variable}} placeholders and return a STATIC ContextLayer.

        Missing variables are left as-is (not replaced).
        """
        content = self._raw
        for key, value in variables.items():
            content = content.replace(f"{{{{{key}}}}}", value)

        return ContextLayer(
            name=self._path.stem,
            content=content,
            priority=LayerPriority.STATIC,
        )

    def render_with_defaults(
        self, defaults: dict[str, str], **overrides: str
    ) -> ContextLayer:
        """Render with defaults merged with explicit overrides.

        Override values take precedence over defaults.
        """
        merged = {**defaults, **overrides}
        return self.render(**merged)
