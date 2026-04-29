"""01-agent-events.py — Define AgentEvent and TurnResult, stream events.

AgentEvent is the streaming contract for the agent loop. Each event carries
a type (text, tool_call, tool_result, status, error) and associated data.
TurnResult collects events from a single ReAct turn for diagnostics.
"""

from the_agents_playbook.loop import AgentEvent, TurnResult


def main():
    # --- Create events of each type ---

    text_event = AgentEvent(type="text", data={"text": "Hello, world!"})
    print(f"Text event:    type={text_event.type!r}, text={text_event.data['text']!r}")

    tool_call_event = AgentEvent(
        type="tool_call",
        data={"tool_name": "shell", "arguments": {"command": "ls"}},
    )
    print(
        f"Tool call event: type={tool_call_event.type!r}, "
        f"tool={tool_call_event.data['tool_name']!r}, "
        f"args={tool_call_event.data['arguments']}"
    )

    tool_result_event = AgentEvent(
        type="tool_result",
        data={"output": "file1.py\nfile2.py", "error": False},
    )
    print(
        f"Tool result:    type={tool_result_event.type!r}, "
        f"output={tool_result_event.data['output'][:20]!r}..., "
        f"error={tool_result_event.data['error']}"
    )

    status_event = AgentEvent(type="status", data={"message": "Thinking..."})
    print(f"Status event:  type={status_event.type!r}, msg={status_event.data['message']!r}")

    error_event = AgentEvent(type="error", data={"message": "Provider timeout"})
    print(f"Error event:   type={error_event.type!r}, msg={error_event.data['message']!r}")

    print()

    # --- Events have safe defaults ---

    default_event = AgentEvent(type="text")
    print(f"Default text event: {default_event.data}")  # {"text": ""}

    default_tool_call = AgentEvent(type="tool_call")
    print(f"Default tool_call:  {default_tool_call.data}")  # {"tool_name": "", "arguments": {}}

    print()

    # --- TurnResult collects events ---

    turn = TurnResult(
        events=[status_event, text_event, tool_call_event],
        tool_calls_made=1,
        final_response="Hello, world!",
    )
    print(f"TurnResult:")
    print(f"  Events:         {len(turn.events)}")
    print(f"  Tool calls:     {turn.tool_calls_made}")
    print(f"  Final response: {turn.final_response}")
    print(f"  Error:          {turn.error}")

    # --- TurnResult with error ---

    error_turn = TurnResult(
        events=[error_event],
        error="Provider timeout",
    )
    print(f"\nError TurnResult: error={error_turn.error!r}")


if __name__ == "__main__":
    main()
