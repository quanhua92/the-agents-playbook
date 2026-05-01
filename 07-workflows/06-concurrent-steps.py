"""06-concurrent-steps.py — Independent steps running via asyncio.gather.

When steps have no dependency edges between them, the workflow
scheduler runs them concurrently for better throughput.
"""

import asyncio
import time

from the_agents_playbook.workflows.protocol import BaseStep, StepResult
from the_agents_playbook.workflows.workflow import Workflow


class SlowStep(BaseStep):
    """Simulates a slow step (0.1s each)."""

    def __init__(
        self, step_id: str, deps: list[str] | None = None, duration: float = 0.1
    ):
        self._id = step_id
        self._deps = deps or []
        self._duration = duration

    @property
    def id(self) -> str:
        return self._id

    @property
    def dependencies(self) -> list[str]:
        return self._deps

    async def run(self, input_data, state):
        await asyncio.sleep(self._duration)
        return StepResult(step_id=self._id, success=True, summary=f"{self._id} done")


async def main():
    # --- Sequential DAG (3 steps, 3 batches) ---

    print("=== Sequential DAG ===")
    seq_wf = Workflow(
        steps=[
            SlowStep("a"),
            SlowStep("b", deps=["a"]),
            SlowStep("c", deps=["b"]),
        ]
    )
    batches = seq_wf._execution_order()
    print(f"  Batches: {batches}")
    print("  (Each step waits for previous)")

    start = time.monotonic()
    async for event in seq_wf.run("test"):
        pass
    elapsed = time.monotonic() - start
    print(f"  Total time: {elapsed:.2f}s (expected ~0.3s)")
    print()

    # --- Concurrent DAG (2 parallel, then 1) ---

    print("=== Concurrent DAG ===")
    par_wf = Workflow(
        steps=[
            SlowStep("research_a"),
            SlowStep("research_b"),
            SlowStep("synthesize", deps=["research_a", "research_b"]),
        ]
    )
    batches = par_wf._execution_order()
    print(f"  Batches: {batches}")
    print("  (research_a and research_b run in parallel)")

    start = time.monotonic()
    async for event in par_wf.run("test"):
        pass
    elapsed = time.monotonic() - start
    print(
        f"  Total time: {elapsed:.2f}s (expected ~0.2s — saved 0.1s from concurrency)"
    )
    print()

    # --- Full parallel (all independent) ---

    print("=== Fully Parallel DAG ===")
    full_wf = Workflow(
        steps=[
            SlowStep("task_1"),
            SlowStep("task_2"),
            SlowStep("task_3"),
        ]
    )
    batches = full_wf._execution_order()
    print(f"  Batches: {batches}")

    start = time.monotonic()
    async for event in full_wf.run("test"):
        pass
    elapsed = time.monotonic() - start
    print(f"  Total time: {elapsed:.2f}s (expected ~0.1s — all 3 ran at once)")


if __name__ == "__main__":
    asyncio.run(main())
