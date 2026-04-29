"""Tests for workflows.protocol — BaseStep, StepResult, WorkflowEvent."""

from the_agents_playbook.workflows.protocol import BaseStep, StepResult, WorkflowEvent
from the_agents_playbook.workflows.state import WorkflowState


class MockStep(BaseStep):
    def __init__(self, step_id: str, dependencies: list[str] | None = None):
        self._id = step_id
        self._deps = dependencies or []

    @property
    def id(self) -> str:
        return self._id

    @property
    def dependencies(self) -> list[str]:
        return self._deps

    async def run(self, input_data, state):
        return StepResult(step_id=self._id, success=True, output_data="done")


class TestStepResult:
    def test_defaults(self):
        r = StepResult(step_id="s1", success=True)
        assert r.step_id == "s1"
        assert r.success is True
        assert r.output_data is None
        assert r.summary == ""
        assert r.updates == {}
        assert r.error is None

    def test_with_data(self):
        r = StepResult(step_id="s1", success=True, output_data={"key": "val"}, summary="ok", updates={"x": 1})
        assert r.output_data == {"key": "val"}
        assert r.summary == "ok"
        assert r.updates == {"x": 1}

    def test_with_error(self):
        r = StepResult(step_id="s1", success=False, error=RuntimeError("boom"))
        assert r.success is False
        assert isinstance(r.error, RuntimeError)


class TestWorkflowEvent:
    def test_step_started(self):
        e = WorkflowEvent(type="step_started", data={"step_id": "plan"})
        assert e.type == "step_started"
        assert e.data["step_id"] == "plan"

    def test_step_completed(self):
        e = WorkflowEvent(type="step_completed", data={"step_id": "plan", "summary": "ok"})
        assert e.type == "step_completed"

    def test_workflow_completed(self):
        e = WorkflowEvent(type="workflow_completed", data={"steps_completed": 2, "steps_failed": 0})
        assert e.type == "workflow_completed"

    def test_workflow_failed(self):
        e = WorkflowEvent(type="workflow_failed", data={"errors": ["cycle"]})
        assert e.type == "workflow_failed"

    def test_default_data(self):
        e = WorkflowEvent(type="step_started")
        assert e.data == {}


class TestBaseStep:
    async def test_subclass_must_implement(self):
        step = MockStep("s1")
        assert step.id == "s1"
        assert step.dependencies == []
        result = await step.run(None, WorkflowState())
        assert result.success is True

    async def test_dependencies(self):
        step = MockStep("s2", dependencies=["s1"])
        assert step.dependencies == ["s1"]
