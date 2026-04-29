"""Workflow protocol — BaseStep ABC, StepResult, and WorkflowEvent types.

These are the core contracts for the workflow orchestration system.
Steps are the nodes in a DAG; StepResult is the output of running one;
WorkflowEvent is the streaming contract for workflow execution.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

# Forward reference for WorkflowState
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .state import WorkflowState


@dataclass
class StepResult:
    """Outcome of executing a single workflow step.

    Attributes:
        step_id: ID of the step that produced this result.
        success: Whether the step completed successfully.
        output_data: The primary value created (e.g., a Plan or code).
        summary: Prose summary for the next step's context.
        updates: Key-value pairs to merge into WorkflowState.shared_context.
        error: Set when success is False.
    """

    step_id: str
    success: bool
    output_data: Any = None
    summary: str = ""
    updates: dict[str, Any] = field(default_factory=dict)
    error: Exception | None = None


@dataclass
class WorkflowEvent:
    """A single event yielded by the workflow runner.

    Attributes:
        type: Event category (step lifecycle or workflow lifecycle).
        data: Associated payload.
    """

    type: Literal[
        "step_started",
        "step_completed",
        "step_failed",
        "workflow_completed",
        "workflow_failed",
    ]
    data: dict[str, Any] = field(default_factory=dict)


class BaseStep(ABC):
    """Abstract base for a workflow step.

    Each step has an ID, optional dependencies, and a run() method
    that receives input data and the shared WorkflowState.

    Subclasses implement the run() method to define step-specific logic.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for this step."""
        ...

    @property
    def dependencies(self) -> list[str]:
        """IDs of steps that must complete before this one runs."""
        return []

    @abstractmethod
    async def run(self, input_data: Any, state: "WorkflowState") -> StepResult:
        """Execute this step.

        Args:
            input_data: Input for this step (e.g., user prompt, previous output).
            state: Shared workflow state for cross-step communication.

        Returns:
            StepResult with the outcome of this step.
        """
        ...
