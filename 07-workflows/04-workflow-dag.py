"""04-workflow-dag.py — Multi-step DAG with dependency ordering.

Workflow validates the DAG (no cycles, no missing deps), computes
topological batches, and executes steps in the correct order.
"""

import asyncio

from the_agents_playbook.workflows.protocol import BaseStep, StepResult
from the_agents_playbook.workflows.state import WorkflowState
from the_agents_playbook.workflows.workflow import Workflow


class DemoStep(BaseStep):
    def __init__(self, step_id: str, deps: list[str] | None = None, sleep_time: float = 0):
        self._id = step_id
        self._deps = deps or []
        self._sleep = sleep_time

    @property
    def id(self) -> str:
        return self._id

    @property
    def dependencies(self) -> list[str]:
        return self._deps

    async def run(self, input_data, state):
        if self._sleep:
            await asyncio.sleep(self._sleep)
        return StepResult(
            step_id=self._id,
            success=True,
            summary=f"{self._id} done",
            updates={f"{self._id}_result": "ok"},
        )


async def main():
    # --- Build a 3-step DAG ---

    print("=== DAG: research → plan → implement ===")
    wf = Workflow(steps=[
        DemoStep("research"),
        DemoStep("plan", deps=["research"]),
        DemoStep("implement", deps=["plan"]),
    ])

    errors = wf.validate()
    print(f"  Validation: {'PASS' if not errors else errors}")

    batches = wf._execution_order()
    for i, batch in enumerate(batches):
        print(f"  Batch {i}: {batch}")
    print()

    # --- Execute ---

    print("=== Execution ===")
    async for event in wf.run("Build a feature"):
        if event.type == "step_started":
            print(f"  [START] {event.data['step_id']}")
        elif event.type == "step_completed":
            print(f"  [DONE]  {event.data['step_id']}: {event.data.get('summary', '')}")
        elif event.type == "workflow_completed":
            print(f"  [FINISH] completed={event.data['steps_completed']}, failed={event.data['steps_failed']}")
    print()

    # --- Validation catches errors ---

    print("=== Validation: Missing Dependency ===")
    bad_wf = Workflow(steps=[DemoStep("s1", deps=["missing"])])
    errors = bad_wf.validate()
    print(f"  Errors: {errors}")
    print()

    print("=== Validation: Cycle Detected ===")
    cycle_wf = Workflow(steps=[
        DemoStep("a", deps=["c"]),
        DemoStep("b", deps=["a"]),
        DemoStep("c", deps=["b"]),
    ])
    errors = cycle_wf.validate()
    print(f"  Errors: {errors}")


if __name__ == "__main__":
    asyncio.run(main())
