"""Tests for loop.protocol — AgentEvent and TurnResult."""

from the_agents_playbook.loop.protocol import AgentEvent, TurnResult


class TestAgentEvent:
    def test_text_event_has_default_data(self):
        event = AgentEvent(type="text")
        assert event.data["text"] == ""

    def test_text_event_with_data(self):
        event = AgentEvent(type="text", data={"text": "hello"})
        assert event.data["text"] == "hello"

    def test_tool_call_event_defaults(self):
        event = AgentEvent(type="tool_call")
        assert event.data["tool_name"] == ""
        assert event.data["arguments"] == {}

    def test_tool_call_event_with_data(self):
        event = AgentEvent(
            type="tool_call",
            data={"tool_name": "shell", "arguments": {"command": "ls"}},
        )
        assert event.data["tool_name"] == "shell"
        assert event.data["arguments"] == {"command": "ls"}

    def test_tool_result_event_defaults(self):
        event = AgentEvent(type="tool_result")
        assert event.data["output"] == ""
        assert event.data["error"] is False

    def test_tool_result_event_with_error(self):
        event = AgentEvent(type="tool_result", data={"output": "failed", "error": True})
        assert event.data["error"] is True

    def test_status_event_defaults(self):
        event = AgentEvent(type="status")
        assert event.data["message"] == ""

    def test_status_event_with_message(self):
        event = AgentEvent(type="status", data={"message": "Thinking..."})
        assert event.data["message"] == "Thinking..."

    def test_error_event_defaults(self):
        event = AgentEvent(type="error")
        assert event.data["message"] == ""

    def test_error_event_with_message(self):
        event = AgentEvent(type="error", data={"message": "Provider timeout"})
        assert event.data["message"] == "Provider timeout"

    def test_all_event_types(self):
        for t in ("text", "tool_call", "tool_result", "status", "error"):
            event = AgentEvent(type=t)
            assert event.type == t


class TestTurnResult:
    def test_defaults(self):
        result = TurnResult()
        assert result.events == []
        assert result.tool_calls_made == 0
        assert result.final_response is None
        assert result.error is None

    def test_with_events(self):
        events = [
            AgentEvent(type="status", data={"message": "thinking"}),
            AgentEvent(type="text", data={"text": "done"}),
        ]
        result = TurnResult(events=events, tool_calls_made=2, final_response="done")
        assert len(result.events) == 2
        assert result.tool_calls_made == 2
        assert result.final_response == "done"

    def test_with_error(self):
        result = TurnResult(error="something broke")
        assert result.error == "something broke"
        assert result.final_response is None
