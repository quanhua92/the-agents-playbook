"""Workflow DAG runner — validates, orders, and executes workflow steps.

The Workflow class takes a list of BaseStep instances, validates the DAG
(no cycles, no missing dependencies), then executes steps respecting
dependency ordering. Independent steps run concurrently via asyncio.gather.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from .hooks import ON_STEP_FAILURE, POST_STEP_EXECUTE, PRE_STEP_EXECUTE
from .protocol import BaseStep, StepResult, WorkflowEvent
from .state import WorkflowState

logger = logging.getLogger(__name__)


class Workflow:
    """Orchestrates multiple steps into a deterministic DAG.

    Usage:
        workflow = Workflow(steps=[plan_step, build_step])
        errors = workflow.validate()
        if errors:
            print(errors)
        else:
            async for event in workflow.run("Fix the bug"):
                print(event)
    """

    def __init__(
        self,
        steps: list[BaseStep],
        state: WorkflowState | None = None,
        on_step_failure: str = "abort",
    ) -> None:
        self._steps = steps
        self._state = state or WorkflowState()
        self._on_step_failure = on_step_failure  # "abort" or "skip"
        self._hooks = None  # set via set_hooks()

    @property
    def state(self) -> WorkflowState:
        return self._state

    @property
    def steps(self) -> list[BaseStep]:
        return self._steps

    def set_hooks(self, hooks: Any) -> None:
        """Set a WorkflowHookSystem for lifecycle events."""
        self._hooks = hooks

    def validate(self) -> list[str]:
        """Validate the workflow DAG.

        Checks for:
        - Missing dependency references
        - Cycles (via topological sort)

        Returns:
            List of validation error strings. Empty = valid.
        """
        errors: list[str] = []
        step_ids = {s.id for s in self._steps}

        # Check for missing dependencies
        for step in self._steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    errors.append(f"Step '{step.id}' depends on '{dep}' which does not exist")

        # Check for cycles via Kahn's algorithm
        in_degree: dict[str, int] = {s.id: 0 for s in self._steps}
        adjacency: dict[str, list[str]] = {s.id: [] for s in self._steps}

        for step in self._steps:
            for dep in step.dependencies:
                if dep in step_ids:
                    adjacency[dep].append(step.id)
                    in_degree[step.id] += 1

        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        visited_count = 0

        while queue:
            node = queue.pop(0)
            visited_count += 1
            for neighbor in adjacency.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited_count != len(self._steps):
            errors.append("Workflow contains a cycle")

        return errors

    def _execution_order(self) -> list[list[str]]:
        """Compute execution batches using topological sort.

        Returns a list of batches where each batch contains step IDs
        that can run concurrently (no dependencies between them).
        """
        step_ids = {s.id for s in self._steps}
        in_degree: dict[str, int] = {s.id: 0 for s in self._steps}
        adjacency: dict[str, list[str]] = {s.id: [] for s in self._steps}

        for step in self._steps:
            for dep in step.dependencies:
                if dep in step_ids:
                    adjacency[dep].append(step.id)
                    in_degree[step.id] += 1

        batches: list[list[str]] = []
        remaining = set(s.id for s in self._steps)

        while remaining:
            batch = [sid for sid in remaining if in_degree[sid] == 0]
            if not batch:
                break
            batches.append(sorted(batch))
            for sid in batch:
                remaining.remove(sid)
                for neighbor in adjacency.get(sid, []):
                    in_degree[neighbor] -= 1

        return batches

    async def _execute_step(self, step: BaseStep, input_data: Any) -> StepResult:
        """Execute a single step, record result, fire hooks.

        Returns the StepResult (events are yielded by the caller).
        """
        if self._hooks:
            await self._hooks.emit(PRE_STEP_EXECUTE, step_id=step.id, state=self._state)

        try:
            result = await step.run(input_data, self._state)
        except Exception as exc:
            result = StepResult(
                step_id=step.id,
                success=False,
                error=exc,
            )

        self._state.add_result(result)

        if result.updates:
            self._state.merge_context(result.updates)

        if self._hooks:
            await self._hooks.emit(
                POST_STEP_EXECUTE,
                step_id=step.id,
                result=result,
                state=self._state,
            )

        if not result.success and self._hooks:
            await self._hooks.emit(
                ON_STEP_FAILURE,
                step_id=step.id,
                error=result.error,
            )

        return result

    async def run(self, initial_input: Any = None) -> AsyncGenerator[WorkflowEvent, None]:
        """Execute the workflow step by step, yielding events.

        Steps are executed in topological order. Independent steps
        within the same batch run concurrently via asyncio.gather.

        Args:
            initial_input: Optional input data for the first steps.

        Yields:
            WorkflowEvent objects for each step lifecycle event.
        """
        errors = self.validate()
        if errors:
            yield WorkflowEvent(type="workflow_failed", data={"errors": errors})
            return

        failed_steps: set[str] = set()
        batches = self._execution_order()

        for batch in batches:
            ready = []
            for sid in batch:
                step = next(s for s in self._steps if s.id == sid)
                deps_failed = any(d in failed_steps for d in step.dependencies)
                if deps_failed:
                    logger.info("Skipping step %s: dependency failed", sid)
                    failed_steps.add(sid)
                    yield WorkflowEvent(
                        type="step_failed",
                        data={"step_id": sid, "error": "dependency failed"},
                    )
                else:
                    ready.append(step)

            if not ready:
                continue

            # Execute all steps in the batch concurrently
            coros = [self._execute_step(step, initial_input) for step in ready]
            results = await asyncio.gather(*coros, return_exceptions=True)

            for step, step_result in zip(ready, results):
                if isinstance(step_result, Exception):
                    step_result = StepResult(
                        step_id=step.id,
                        success=False,
                        error=step_result,
                    )
                    self._state.add_result(step_result)

                yield WorkflowEvent(type="step_started", data={"step_id": step.id})

                if step_result.success:
                    yield WorkflowEvent(
                        type="step_completed",
                        data={"step_id": step.id, "summary": step_result.summary},
                    )
                else:
                    yield WorkflowEvent(
                        type="step_failed",
                        data={"step_id": step.id, "error": str(step_result.error)},
                    )
                    if self._on_step_failure == "abort":
                        yield WorkflowEvent(
                            type="workflow_failed",
                            data={"error": f"Step '{step.id}' failed: {step_result.error}"},
                        )
                        return
                    failed_steps.add(step.id)

        yield WorkflowEvent(
            type="workflow_completed",
            data={
                "steps_completed": len(self._state.successful_steps()),
                "steps_failed": len(self._state.failed_steps()),
            },
        )
