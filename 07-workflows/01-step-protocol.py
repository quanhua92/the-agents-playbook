"""01-step-protocol.py — Define BaseStep, StepResult, run a single step.

WorkflowStep is the node in a workflow DAG. Each step has an ID, optional
dependencies on other steps, and a run() method that produces a StepResult.
"""

import asyncio

from the_agents_playbook.workflows.protocol import BaseStep, StepResult, WorkflowEvent
from the_agents_playbook.workflows.state import WorkflowState


class GreetingStep(BaseStep):
    """A simple step that generates a greeting."""

    def __init__(self, name: str = "greet"):
        self._id = name

    @property
    def id(self) -> str:
        return self._id

    async def run(self, input_data, state: WorkflowState):
        name = input_data or "World"
        greeting = f"Hello, {name}!"
        return StepResult(
            step_id=self._id,
            success=True,
            output_data=greeting,
            summary=f"Greeted {name}",
            updates={"greeting": greeting},
        )


async def main():
    # --- StepResult anatomy ---

    print("=== StepResult ===")
    result = StepResult(
        step_id="demo",
        success=True,
        output_data="hello",
        summary="Said hello",
        updates={"key": "value"},
    )
    print(f"  step_id:    {result.step_id}")
    print(f"  success:    {result.success}")
    print(f"  output:     {result.output_data}")
    print(f"  summary:    {result.summary}")
    print(f"  updates:    {result.updates}")
    print()

    # --- WorkflowEvent types ---

    print("=== WorkflowEvent Types ===")
    for event_type in [
        "step_started",
        "step_completed",
        "step_failed",
        "workflow_completed",
        "workflow_failed",
    ]:
        event = WorkflowEvent(type=event_type, data={"step_id": "demo"})
        print(f"  {event_type:25s} → data={event.data}")
    print()

    # --- Run a single step ---

    print("=== Run GreetingStep ===")
    step = GreetingStep("greet")
    state = WorkflowState()
    result = await step.run("Alice", state)

    print(f"  Success: {result.success}")
    print(f"  Output:  {result.output_data}")
    print(f"  State:   {state.shared_context}")


if __name__ == "__main__":
    asyncio.run(main())
