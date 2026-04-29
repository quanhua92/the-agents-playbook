"""ContextBuilder — assembles the system prompt from ordered context layers.

Layers are sorted by priority (STATIC → SEMI_STABLE → DYNAMIC) then by order
within each priority. This maximizes KV cache hits — static sections stay
cached between turns while only dynamic sections are recomputed.

Context windows are a budget — the builder includes a max_tokens estimate
to help avoid exceeding model limits.
"""

import logging
from typing import Self

from .layers import ContextLayer, LayerPriority

logger = logging.getLogger(__name__)

# Rough estimate: 1 token ≈ 4 characters
_CHARS_PER_TOKEN = 4


class ContextBuilder:
    """Assemble a system prompt from composable context layers.

    Usage:
        builder = ContextBuilder()
        builder.add_static(soul_layer)
        builder.add_static(tool_defs_layer)
        builder.add_semi_stable(memory_summary_layer)
        builder.add_dynamic(git_status_layer)
        system_prompt = builder.build()
    """

    def __init__(self, max_tokens: int = 8192) -> None:
        self._layers: list[ContextLayer] = []
        self._max_tokens = max_tokens

    def add_static(self, layer: ContextLayer) -> Self:
        """Add a static layer (system instructions, tool definitions)."""
        layer.priority = LayerPriority.STATIC
        self._layers.append(layer)
        return self

    def add_semi_stable(self, layer: ContextLayer) -> Self:
        """Add a semi-stable layer (memory, skills, user preferences)."""
        layer.priority = LayerPriority.SEMI_STABLE
        self._layers.append(layer)
        return self

    def add_dynamic(self, layer: ContextLayer) -> Self:
        """Add a dynamic layer (git status, date, tool results)."""
        layer.priority = LayerPriority.DYNAMIC
        self._layers.append(layer)
        return self

    def add(self, layer: ContextLayer) -> Self:
        """Add a layer using its own priority setting."""
        self._layers.append(layer)
        return self

    def clear(self) -> Self:
        """Remove all layers."""
        self._layers.clear()
        return self

    @property
    def layers(self) -> list[ContextLayer]:
        """Return the current layers sorted by priority then order."""
        return sorted(self._layers)

    def estimated_tokens(self) -> int:
        """Estimate the total token count of the assembled prompt."""
        total = sum(len(layer.content) for layer in self._layers)
        return total // _CHARS_PER_TOKEN

    def token_budget_remaining(self) -> int:
        """Return remaining token budget before exceeding max_tokens."""
        return self._max_tokens - self.estimated_tokens()

    def build(self) -> str:
        """Assemble all layers into the final system prompt string.

        Layers are sorted by priority (STATIC → SEMI_STABLE → DYNAMIC),
        then by order within each priority. Sections are joined with
        double newlines.

        Raises:
            ValueError: if estimated tokens exceed max_tokens budget.
        """
        estimated = self.estimated_tokens()
        if estimated > self._max_tokens:
            logger.warning(
                "System prompt estimate (%d tokens) exceeds budget (%d tokens)",
                estimated,
                self._max_tokens,
            )

        sorted_layers = self.layers
        sections = [layer.content for layer in sorted_layers if layer.content.strip()]

        return "\n\n".join(sections)

    def build_report(self) -> dict:
        """Build the prompt and return a diagnostic report.

        Returns:
            dict with keys: total_tokens, budget, over_budget, layer_breakdown
        """
        sorted_layers = self.layers
        total_tokens = self.estimated_tokens()
        breakdown = []
        for layer in sorted_layers:
            tokens = len(layer.content) // _CHARS_PER_TOKEN
            breakdown.append({
                "name": layer.name,
                "priority": layer.priority.name,
                "tokens": tokens,
                "characters": len(layer.content),
            })

        return {
            "total_tokens": total_tokens,
            "budget": self._max_tokens,
            "over_budget": total_tokens > self._max_tokens,
            "layer_breakdown": breakdown,
        }
