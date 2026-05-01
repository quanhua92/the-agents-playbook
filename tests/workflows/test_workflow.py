"""Tests for workflows.workflow — DAG validation, execution, concurrency."""

from the_agents_playbook.workflows.protocol import BaseStep, StepResult
from the_agents_playbook.workflows.workflow import Workflow


class CountStep(BaseStep):
    """A test step that tracks execution order."""

    def __init__(
        self, step_id: str, dependencies: list[str] | None = None, fail: bool = False
    ):
        self._id = step_id
        self._deps = dependencies or []
        self._fail = fail
        self.executed = False

    @property
    def id(self) -> str:
        return self._id

    @property
    def dependencies(self) -> list[str]:
        return self._deps

    async def run(self, input_data, state):
        self.executed = True
        if self._fail:
            return StepResult(
                step_id=self._id,
                success=False,
                error=RuntimeError(f"{self._id} failed"),
            )
        return StepResult(
            step_id=self._id,
            success=True,
            output_data=f"{self._id}_output",
            updates={f"{self._id}_result": f"{self._id}_output"},
        )


class TestValidation:
    def test_valid_workflow(self):
        s1 = CountStep("s1")
        s2 = CountStep("s2", dependencies=["s1"])
        wf = Workflow(steps=[s1, s2])
        assert wf.validate() == []

    def test_missing_dependency(self):
        s1 = CountStep("s1", dependencies=["nonexistent"])
        wf = Workflow(steps=[s1])
        errors = wf.validate()
        assert any("nonexistent" in e for e in errors)

    def test_cycle_detected(self):
        s1 = CountStep("s1", dependencies=["s2"])
        s2 = CountStep("s2", dependencies=["s1"])
        wf = Workflow(steps=[s1, s2])
        errors = wf.validate()
        assert any("cycle" in e.lower() for e in errors)

    def test_three_step_chain(self):
        s1 = CountStep("s1")
        s2 = CountStep("s2", dependencies=["s1"])
        s3 = CountStep("s3", dependencies=["s2"])
        wf = Workflow(steps=[s1, s2, s3])
        assert wf.validate() == []

    def test_empty_workflow(self):
        wf = Workflow(steps=[])
        assert wf.validate() == []


class TestExecutionOrder:
    def test_linear_chain(self):
        s1 = CountStep("s1")
        s2 = CountStep("s2", dependencies=["s1"])
        s3 = CountStep("s3", dependencies=["s2"])
        wf = Workflow(steps=[s1, s2, s3])
        batches = wf._execution_order()
        assert batches == [["s1"], ["s2"], ["s3"]]

    def test_parallel_steps(self):
        s1 = CountStep("s1")
        s2 = CountStep("s2")
        s3 = CountStep("s3", dependencies=["s1", "s2"])
        wf = Workflow(steps=[s1, s2, s3])
        batches = wf._execution_order()
        # s1 and s2 have no deps, can run in parallel
        assert set(batches[0]) == {"s1", "s2"}
        assert batches[1] == ["s3"]


class TestRun:
    async def test_simple_run(self):
        s1 = CountStep("s1")
        wf = Workflow(steps=[s1])
        events = []
        async for event in wf.run("test"):
            events.append(event)

        types = [e.type for e in events]
        assert "step_started" in types
        assert "step_completed" in types
        assert "workflow_completed" in types

    async def test_two_step_chain(self):
        s1 = CountStep("s1")
        s2 = CountStep("s2", dependencies=["s1"])
        wf = Workflow(steps=[s1, s2])
        events = []
        async for event in wf.run("test"):
            events.append(event)

        assert s1.executed
        assert s2.executed
        completed = [e for e in events if e.type == "step_completed"]
        assert len(completed) == 2

    async def test_shared_context_flows(self):
        s1 = CountStep("s1")
        s2 = CountStep("s2", dependencies=["s1"])
        wf = Workflow(steps=[s1, s2])
        async for _ in wf.run("test"):
            pass
        assert "s1_result" in wf.state.shared_context

    async def test_step_failure_aborts(self):
        s1 = CountStep("s1", fail=True)
        wf = Workflow(steps=[s1])
        events = []
        async for event in wf.run("test"):
            events.append(event)

        assert any(e.type == "workflow_failed" for e in events)

    async def test_step_failure_skips_dependent(self):
        s1 = CountStep("s1", fail=True)
        s2 = CountStep("s2", dependencies=["s1"])
        wf = Workflow(steps=[s1, s2], on_step_failure="skip")
        events = []
        async for event in wf.run("test"):
            events.append(event)

        # s2 should be skipped (dependency failed)
        failed = [e for e in events if e.type == "step_failed"]
        assert len(failed) >= 1

    async def test_validation_failure_yields_error(self):
        s1 = CountStep("s1", dependencies=["missing"])
        wf = Workflow(steps=[s1])
        events = []
        async for event in wf.run("test"):
            events.append(event)

        assert events[0].type == "workflow_failed"

    async def test_parallel_steps_both_execute(self):
        s1 = CountStep("s1")
        s2 = CountStep("s2")
        wf = Workflow(steps=[s1, s2])
        async for _ in wf.run("test"):
            pass
        assert s1.executed
        assert s2.executed
