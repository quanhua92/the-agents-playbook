"""05-workflow-hooks.py — Pre/post execution hooks for audit and validation.

WorkflowHookSystem fires at key workflow lifecycle points, enabling
custom middleware without modifying step or workflow code.
"""

import asyncio
from unittest.mock import AsyncMock

from the_agents_playbook.workflows.hooks import (
    ON_STEP_FAILURE,
    POST_STEP_EXECUTE,
    PRE_STEP_EXECUTE,
    WorkflowHookSystem,
)
from the_agents_playbook.workflows.protocol import BaseStep, StepResult
from the_agents_playbook.workflows.state import WorkflowState
from the_agents_playbook.workflows.workflow import Workflow


class SimpleStep(BaseStep):
    def __init__(self, step_id: str):
        self._id = step_id

    @property
    def id(self) -> str:
        return self._id

    async def run(self, input_data, state):
        return StepResult(step_id=self._id, success=True, summary=f"{self._id} completed")


async def main():
    # --- Set up hooks ---

    hooks = WorkflowHookSystem()
    audit_log: list[dict] = []

    async def audit(event_name: str, **kwargs):
        audit_log.append({"event": event_name, **kwargs})

    hooks.on(PRE_STEP_EXECUTE, lambda **kw: audit("pre", **kw))
    hooks.on(POST_STEP_EXECUTE, lambda **kw: audit("post", **kw))
    hooks.on(ON_STEP_FAILURE, lambda **kw: audit("failure", **kw))
    print("Registered 3 workflow hooks")
    print()

    # --- Run workflow with hooks ---

    print("=== Workflow with Hooks ===")
    wf = Workflow(steps=[
        SimpleStep("research"),
        SimpleStep("plan"),
    ])
    wf.set_hooks(hooks)

    async for event in wf.run("test"):
        if event.type in ("step_started", "step_completed", "workflow_completed"):
            print(f"  [{event.type}] {event.data}")
    print()

    # --- Audit log ---

    print("=== Audit Log ===")
    for entry in audit_log:
        print(f"  [{entry['event']:8s}] step_id={entry.get('step_id', '—')}")
    print()

    # --- Hook for validation ---

    print("=== Pre-Execution Validation ===")
    validated_steps = set()

    async def validate_step(**kwargs):
        sid = kwargs.get("step_id", "")
        validated_steps.add(sid)
        print(f"  Validating step: {sid}")

    hooks.clear()
    hooks.on(PRE_STEP_EXECUTE, validate_step)
    wf.set_hooks(hooks)

    async for _ in wf.run("test"):
        pass
    print(f"  Validated steps: {sorted(validated_steps)}")


if __name__ == "__main__":
    asyncio.run(main())
