"""Context layers — ordered sections for system prompt assembly.

Layers are sorted by priority first (STATIC → SEMI_STABLE → DYNAMIC),
then by order within each priority. This maximizes KV cache hits because
static sections rarely change between turns.
"""

from dataclasses import dataclass, field
from enum import IntEnum


class LayerPriority(IntEnum):
    """Context layer priority — lower values are assembled first in the prompt.

    STATIC:      System instructions, tool definitions, world rules (rarely changes)
    SEMI_STABLE: Memory summaries, skill descriptions, user preferences (changes per session)
    DYNAMIC:     Git status, date, active tool results (changes every turn)
    """

    STATIC = 0
    SEMI_STABLE = 1
    DYNAMIC = 2


@dataclass
class ContextLayer:
    """A single section of the system prompt.

    Layers with lower priority values appear earlier in the assembled prompt.
    Within the same priority, layers with lower order appear first.
    """

    name: str
    content: str
    priority: LayerPriority = LayerPriority.STATIC
    order: int = 0
    metadata: dict = field(default_factory=dict)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ContextLayer):
            return NotImplemented
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.order < other.order
