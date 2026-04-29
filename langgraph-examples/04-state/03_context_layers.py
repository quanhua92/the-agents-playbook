"""03_context_layers.py -- System message composition mirrors ContextBuilder.

In the root project:
  builder = ContextBuilder()
  builder.add_static(ContextLayer(name="identity", content="You are..."))
  builder.add_dynamic(ContextLayer(name="date", content=f"Today: {date}"))
  system_prompt = builder.build()  # sorted by priority, joined with \n\n

In LangGraph:
  Same concept, but output is a SystemMessage injected as the first message
  in the messages list. Layers are sorted by priority (STATIC -> SEMI_STABLE -> DYNAMIC)
  to maximize KV cache hits.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum

from langchain_core.messages import SystemMessage


class LayerPriority(IntEnum):
    STATIC = 0       # System instructions, world rules (rarely changes)
    SEMI_STABLE = 1  # Memory, preferences (changes per session)
    DYNAMIC = 2      # Date, git status, runtime context (every turn)


@dataclass
class ContextLayer:
    """A single section of the system prompt."""
    name: str
    content: str
    priority: LayerPriority = LayerPriority.STATIC
    order: int = 0

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ContextLayer):
            return NotImplemented
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.order < other.order


class LangGraphContextBuilder:
    """Builds a SystemMessage from composable context layers.

    Mirrors root's ContextBuilder but outputs a langchain SystemMessage.
    """

    def __init__(self, max_chars: int = 32768):
        self._layers: list[ContextLayer] = []

    def add_static(self, layer: ContextLayer) -> "LangGraphContextBuilder":
        layer.priority = LayerPriority.STATIC
        self._layers.append(layer)
        return self

    def add_semi_stable(self, layer: ContextLayer) -> "LangGraphContextBuilder":
        layer.priority = LayerPriority.SEMI_STABLE
        self._layers.append(layer)
        return self

    def add_dynamic(self, layer: ContextLayer) -> "LangGraphContextBuilder":
        layer.priority = LayerPriority.DYNAMIC
        self._layers.append(layer)
        return self

    def build_message(self) -> SystemMessage:
        sorted_layers = sorted(self._layers)
        sections = [l.content for l in sorted_layers if l.content.strip()]
        return SystemMessage(content="\n\n".join(sections))

    def build_report(self) -> list[dict]:
        sorted_layers = sorted(self._layers)
        return [
            {
                "name": l.name,
                "priority": l.priority.name,
                "chars": len(l.content),
            }
            for l in sorted_layers
        ]


def main():
    builder = LangGraphContextBuilder()

    builder.add_static(ContextLayer(
        name="identity",
        content="You are a helpful coding assistant.",
    ))
    builder.add_static(ContextLayer(
        name="rules",
        content="Always show your work. Use Python 3.12+.",
        order=1,
    ))
    builder.add_semi_stable(ContextLayer(
        name="preferences",
        content="User prefers concise answers with code examples.",
    ))
    builder.add_dynamic(ContextLayer(
        name="runtime",
        content=f"Current date: {datetime.now().strftime('%Y-%m-%d')}",
    ))

    system_msg = builder.build_message()

    print("=== Context Layer Report ===")
    for layer in builder.build_report():
        print(f"  [{layer['priority']:12s}] {layer['name']:15s} ({layer['chars']} chars)")

    print(f"\n=== Assembled System Message ({len(system_msg.content)} chars) ===")
    print(system_msg.content)

    print("\n=== Root Comparison ===")
    print("Root: ContextBuilder.build() -> str (system prompt)")
    print("Here: LangGraphContextBuilder.build_message() -> SystemMessage")
    print("Same layering concept, different output type.")
    print("Inject as first message: [SystemMessage, ...other_messages]")


if __name__ == "__main__":
    main()
