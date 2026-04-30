# Streaming Agent Loop

## Problem

The existing `Agent.run()` calls `provider.send_message()` which waits for the entire LLM response before returning. Users see 5-10 seconds of silence, then all output at once. For interactive agents, this feels broken — even a simple response should start appearing immediately.

## Solution

A new `run_streaming()` method on the `Agent` class in `loop/agent.py` uses `provider.stream()` instead of `send_message()`. It yields token-by-token text deltas in real-time while buffering tool call chunks until complete.

### How streaming differs from the existing run()

```
run()           → send_message() → waits → yields complete text event
run_streaming() → stream()       → yields text_delta immediately → buffers tool calls
```

### The event flow

For each LLM call in the ReAct loop, `run_streaming()`:

1. Opens a stream to the provider
2. For each `ResponseChunk` from the stream:
   - **Text chunk** (`delta_text`): yields `AgentEvent(type="text_delta")` immediately
   - **Tool call chunk** (`tool_call_id`, `tool_call_name`, `tool_call_arguments`): buffers into a per-call dictionary
3. When the stream finishes:
   - If tool calls were buffered → parse arguments, dispatch each tool, feed results back, continue the loop
   - If no tool calls → yield the complete text as `AgentEvent(type="text")`, return

### New event type

`AgentEvent.type` now includes `"text_delta"` alongside the existing types:

```
"text"        → complete text response (same as before)
"text_delta"  → single token chunk during streaming (new)
"tool_call"   → complete tool call with parsed arguments
"tool_result" → tool execution result
"status"      → iteration progress message
"error"       → error message
```

### Why tool calls need buffering

Modern streaming APIs interleave text and tool call chunks in a single response. For example:

```
chunk 1: delta_text = "Let me look that up"
chunk 2: delta_text = " for you"
chunk 3: tool_call_id = "call_abc", tool_call_name = "search"
chunk 4: tool_call_arguments = '{"qu'
chunk 5: tool_call_arguments = 'ery": "weather"}'
chunk 6: finish = true
```

The tool call arguments arrive as fragments. You can't dispatch until you have the complete JSON. `run_streaming()` accumulates argument fragments per `tool_call_id`, then parses and dispatches once the stream ends.

### Code comparison

```python
# Non-streaming (existing, unchanged)
async for event in agent.run("What is the weather?"):
    # Waits until entire response is ready
    print(event)

# Streaming (new)
async for event in agent.run_streaming("What is the weather?"):
    if event.type == "text_delta":
        print(event.data["text"], end="", flush=True)  # appears token-by-token
    elif event.type == "tool_call":
        print(f"\n[calling {event.data['tool_name']}]")
    elif event.type == "text":
        print(event.data["text"])  # complete final text
```

### Backward compatibility

`run()` is completely untouched. `run_streaming()` is a separate method. All existing code continues to work without changes.

## Code Reference

- `src/the_agents_playbook/loop/agent.py` — `Agent.run_streaming()` method
- `src/the_agents_playbook/loop/protocol.py` — `AgentEvent` with `"text_delta"` type (line 15)

## Playground Example

- `05-the-loop/07-streaming-agent.py` — token-by-token agent output with tool call buffering

## LangGraph Example

- `langgraph-examples/05-the-agent/05_streaming.py` — uses `.astream_events()` on `create_react_agent`
