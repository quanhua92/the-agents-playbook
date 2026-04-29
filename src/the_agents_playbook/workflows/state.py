"""WorkflowState — shared context, history, and global memory for workflows.

WorkflowState is the "shared clipboard" that steps use to communicate.
Each step reads from and writes to shared_context. The history log
captures every step execution for post-mortem analysis.
"""

from dataclasses import dataclass, field
from typing import Any

from ..memory.protocol import BaseMemoryProvider


@dataclass
class WorkflowState:
    """Shared state for a workflow execution.

    Attributes:
        history: Log of every step executed (for post-mortem analysis).
        shared_context: Key-value store for cross-step communication.
        global_memory: Optional persistent memory across the entire workflow.
    """

    history: list[Any] = field(default_factory=list)  # list[StepResult] but avoid circular import
    shared_context: dict[str, Any] = field(default_factory=dict)
    global_memory: BaseMemoryProvider | None = None

    def add_result(self, result: Any) -> None:
        """Append a step result to the history log."""
        self.history.append(result)

    def get_context(self, key: str, default: Any = None) -> Any:
        """Read a value from shared context."""
        return self.shared_context.get(key, default)

    def set_context(self, key: str, value: Any) -> None:
        """Write a value to shared context."""
        self.shared_context[key] = value

    def merge_context(self, updates: dict[str, Any]) -> None:
        """Merge a dict of updates into shared context."""
        self.shared_context.update(updates)

    def clear_context(self) -> None:
        """Remove all entries from shared context."""
        self.shared_context.clear()

    def successful_steps(self) -> list[Any]:
        """Return history entries that succeeded."""
        return [r for r in self.history if getattr(r, "success", False)]

    def failed_steps(self) -> list[Any]:
        """Return history entries that failed."""
        return [r for r in self.history if not getattr(r, "success", True)]
