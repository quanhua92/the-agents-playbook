"""03-workflow-state.py — Shared context, history, global memory.

WorkflowState is the shared clipboard that steps use to communicate.
Data written by one step is available to all subsequent steps.
"""

from the_agents_playbook.workflows.protocol import StepResult
from the_agents_playbook.workflows.state import WorkflowState


def main():
    state = WorkflowState()

    # --- Shared context ---

    print("=== Shared Context ===")
    state.set_context("plan", "Fix auth bug")
    state.set_context("user", "Alice")
    state.merge_context({"priority": "high", "files": ["auth.py", "user.py"]})

    print(f"  plan:     {state.get_context('plan')}")
    print(f"  user:     {state.get_context('user')}")
    print(f"  priority: {state.get_context('priority')}")
    print(f"  files:    {state.get_context('files')}")
    print(f"  missing:  {state.get_context('missing', 'N/A')}")
    print()

    # --- History ---

    print("=== History ===")
    state.add_result(StepResult(step_id="plan", success=True, summary="Created plan"))
    state.add_result(StepResult(step_id="research", success=True, summary="Found bug"))
    state.add_result(StepResult(step_id="implement", success=True, summary="Fixed bug"))
    state.add_result(
        StepResult(step_id="test", success=False, error=AssertionError("test failed"))
    )

    print(f"  Total entries: {len(state.history)}")
    for entry in state.history:
        status = "OK" if entry.success else "FAIL"
        print(f"    [{status}] {entry.step_id}: {entry.summary}")
    print()

    # --- Query history ---

    print("=== History Queries ===")
    print(f"  Successful: {len(state.successful_steps())}")
    print(f"  Failed:     {len(state.failed_steps())}")
    print()

    # --- Clear context ---

    print("=== Clear Context ===")
    state.clear_context()
    print(f"  After clear: {state.shared_context}")


if __name__ == "__main__":
    main()
