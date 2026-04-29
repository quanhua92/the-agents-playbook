"""02-plan-and-build.py — PlanStep (read-only) then BuildStep (write).

The plan-and-execute pattern: the agent commits to a plan first,
the plan is stored in shared context, then BuildStep consumes it
to guide implementation.
"""

import asyncio

from the_agents_playbook.workflows.state import WorkflowState
from the_agents_playbook.workflows.steps import BuildStep, PlanStep


async def main():
    # --- Plan phase ---

    print("=== Plan Phase ===")
    plan_step = PlanStep(
        step_id="plan",
        plan_instructions="1. Read auth.py\n2. Find the bug\n3. Fix it\n4. Write tests",
    )
    state = WorkflowState()
    plan_result = await plan_step.run("Fix the auth bug", state)

    print(f"  Success: {plan_result.success}")
    print(f"  Output:  {plan_result.output_data}")
    print(f"  Context: {state.shared_context.get('plan', '')[:60]}...")
    print()

    # --- Build phase (consumes plan from state) ---

    print("=== Build Phase ===")
    build_step = BuildStep(
        step_id="build",
        dependencies=["plan"],
        build_instructions="Implemented fixes for auth.py",
    )
    build_result = await build_step.run(None, state)

    print(f"  Success: {build_result.success}")
    print(f"  Output:  {build_result.output_data}")
    print()

    # --- Build without plan fails ---

    print("=== Build Without Plan ===")
    empty_state = WorkflowState()
    orphan_build = BuildStep(step_id="build_orphan")
    fail_result = await orphan_build.run(None, empty_state)

    print(f"  Success: {fail_result.success}")
    print(f"  Error:   {fail_result.error}")
    print()

    # --- History (recorded when using Workflow runner, or manually) ---

    print("=== Workflow History ===")
    state.add_result(plan_result)
    state.add_result(build_result)
    print(f"  Total steps:  {len(state.history)}")
    print(f"  Successful:   {len(state.successful_steps())}")
    print(f"  Failed:       {len(state.failed_steps())}")


if __name__ == "__main__":
    asyncio.run(main())
