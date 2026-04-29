"""Tests for workflows.state — WorkflowState."""

from the_agents_playbook.workflows.protocol import StepResult
from the_agents_playbook.workflows.state import WorkflowState


class TestWorkflowState:
    def test_defaults(self):
        state = WorkflowState()
        assert state.history == []
        assert state.shared_context == {}
        assert state.global_memory is None

    def test_add_result(self):
        state = WorkflowState()
        result = StepResult(step_id="s1", success=True)
        state.add_result(result)
        assert len(state.history) == 1
        assert state.history[0].step_id == "s1"

    def test_get_set_context(self):
        state = WorkflowState()
        state.set_context("plan", "fix auth")
        assert state.get_context("plan") == "fix auth"
        assert state.get_context("missing") is None
        assert state.get_context("missing", "default") == "default"

    def test_merge_context(self):
        state = WorkflowState()
        state.set_context("a", 1)
        state.merge_context({"b": 2, "c": 3})
        assert state.shared_context == {"a": 1, "b": 2, "c": 3}

    def test_clear_context(self):
        state = WorkflowState()
        state.set_context("x", "y")
        state.clear_context()
        assert state.shared_context == {}

    def test_successful_steps(self):
        state = WorkflowState()
        state.add_result(StepResult(step_id="s1", success=True))
        state.add_result(StepResult(step_id="s2", success=False))
        state.add_result(StepResult(step_id="s3", success=True))
        assert len(state.successful_steps()) == 2

    def test_failed_steps(self):
        state = WorkflowState()
        state.add_result(StepResult(step_id="s1", success=True))
        state.add_result(StepResult(step_id="s2", success=False))
        assert len(state.failed_steps()) == 1
