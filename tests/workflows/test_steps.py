"""Tests for workflows.steps — PlanStep and BuildStep."""

from the_agents_playbook.workflows.steps import BuildStep, PlanStep
from the_agents_playbook.workflows.state import WorkflowState


class TestPlanStep:
    def test_properties(self):
        step = PlanStep(step_id="plan", plan_instructions="Fix auth bug")
        assert step.id == "plan"
        assert step.dependencies == []
        assert step.plan_instructions == "Fix auth bug"

    async def test_run_stores_plan_in_context(self):
        step = PlanStep(step_id="plan", plan_instructions="Create a plan to fix auth")
        state = WorkflowState()
        result = await step.run("Fix auth", state)

        assert result.success is True
        assert result.step_id == "plan"
        assert "plan" in result.updates
        assert state.shared_context["plan"] == "Create a plan to fix auth"

    async def test_run_uses_input_when_no_instructions(self):
        step = PlanStep(step_id="plan")
        state = WorkflowState()
        result = await step.run("User wants X", state)

        assert result.success is True
        assert state.shared_context["plan"] == "User wants X"


class TestBuildStep:
    def test_properties(self):
        step = BuildStep(
            step_id="build", dependencies=["plan"], build_instructions="Implement"
        )
        assert step.id == "build"
        assert step.dependencies == ["plan"]
        assert step.build_instructions == "Implement"

    def test_default_no_dependencies(self):
        step = BuildStep(step_id="build")
        assert step.dependencies == []

    async def test_run_consumes_plan_from_state(self):
        plan = PlanStep(step_id="plan", plan_instructions="Plan: fix auth")
        state = WorkflowState()
        await plan.run("Fix auth", state)

        build = BuildStep(
            step_id="build",
            dependencies=["plan"],
            build_instructions="Implement the fix",
        )
        result = await build.run(None, state)

        assert result.success is True
        assert result.step_id == "build"

    async def test_run_fails_without_plan(self):
        build = BuildStep(step_id="build", dependencies=["plan"])
        state = WorkflowState()
        result = await build.run(None, state)

        assert result.success is False
        assert result.error is not None
