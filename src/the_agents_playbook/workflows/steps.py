"""Workflow steps — PlanStep (read-only) and BuildStep (write after approval).

Plan-and-Execute pattern: the agent commits to a plan first, the user
approves it, then write tools are enabled for implementation.
"""

import logging
from typing import Any

from .protocol import BaseStep, StepResult

logger = logging.getLogger(__name__)


class PlanStep(BaseStep):
    """A step that produces a plan using read-only tools.

    The plan is saved to WorkflowState.shared_context under the "plan" key.
    Downstream BuildStep consumes this plan for implementation.

    Usage:
        step = PlanStep(
            step_id="plan",
            plan_instructions="Create a plan for fixing the auth bug.",
        )
    """

    def __init__(self, step_id: str, plan_instructions: str = "") -> None:
        self._id = step_id
        self._plan_instructions = plan_instructions

    @property
    def id(self) -> str:
        return self._id

    @property
    def plan_instructions(self) -> str:
        return self._plan_instructions

    async def run(self, input_data: Any, state: "WorkflowState") -> StepResult:  # noqa: ANN401,F821
        """Generate a plan and store it in shared context.

        In a real implementation, this would invoke an Agent with read-only
        tools. Here it stores the plan instructions as the output.
        """
        plan = self._plan_instructions or str(input_data)

        # Store plan in shared context for BuildStep to consume
        state.shared_context["plan"] = plan

        return StepResult(
            step_id=self._id,
            success=True,
            output_data=plan,
            summary=f"Created plan: {plan[:100]}",
            updates={"plan": plan},
        )


class BuildStep(BaseStep):
    """A step that implements a plan using write tools.

    Consumes the "plan" from WorkflowState.shared_context. Only runs
    after PlanStep has produced an approved plan.

    Usage:
        step = BuildStep(
            step_id="build",
            dependencies=["plan"],
            build_instructions="Implement the approved plan.",
        )
    """

    def __init__(
        self,
        step_id: str,
        build_instructions: str = "",
        dependencies: list[str] | None = None,
    ) -> None:
        self._id = step_id
        self._build_instructions = build_instructions
        self._dependencies = dependencies or []

    @property
    def id(self) -> str:
        return self._id

    @property
    def dependencies(self) -> list[str]:
        return self._dependencies

    @property
    def build_instructions(self) -> str:
        return self._build_instructions

    async def run(self, input_data: Any, state: "WorkflowState") -> StepResult:  # noqa: ANN401,F821
        """Implement the plan from shared context.

        In a real implementation, this would invoke an Agent with write
        tools enabled. Here it acknowledges the plan.
        """
        plan = state.shared_context.get("plan", "")
        if not plan:
            return StepResult(
                step_id=self._id,
                success=False,
                error=RuntimeError("No plan found in shared context"),
            )

        build_output = self._build_instructions or f"Implemented plan: {plan[:80]}"

        return StepResult(
            step_id=self._id,
            success=True,
            output_data=build_output,
            summary=f"Built: {build_output[:100]}",
            updates={"build_output": build_output},
        )
