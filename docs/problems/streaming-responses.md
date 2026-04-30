# Streaming Responses

## Problem

A non-streaming LLM call blocks for 5-15 seconds. The user sees nothing during this time — just a spinning cursor. For long responses, this is unacceptable. Streaming also matters for tool calls: modern APIs interleave text deltas and tool call chunks in a single response. You can't just "read the full response and parse."

## Solution

Use Server-Sent Events (SSE) to parse the HTTP stream into typed chunks. The provider handles SSE parsing; the consumer processes chunks incrementally.

### ResponseChunk Protocol

`ResponseChunk` (`src/the_agents_playbook/providers/types.py:222`) represents a single chunk from the API:

- `delta_text`: text token delta
- `delta_reasoning`: chain-of-thought token delta
- `tool_call_id`: which tool call this chunk belongs to
- `tool_call_name`: function name (first chunk only)
- `tool_call_arguments`: partial JSON arguments (accumulated across chunks)
- `stop_reason`: reason the stream ended (`"end_turn"`, `"tool_calls"`, etc.)
- `finish`: `True` on the `[DONE]` sentinel

`StreamUsage` (`types.py:234`) carries final token counts reported at stream end.

### BaseProvider.stream()

`BaseProvider.stream()` (`src/the_agents_playbook/providers/base.py:76`) opens an HTTP streaming connection via `httpx.AsyncClient.stream()`, then iterates `response.aiter_lines()` and parses each SSE line through `_parse_sse_line()` (`base.py:97`). The SSE parser strips `data:` prefixes, handles `[DONE]` sentinels, and delegates to `_parse_stream_chunk()` (abstract, provider-specific).

Each concrete provider (OpenAI, Anthropic) implements `_parse_stream_chunk()` to map provider-specific JSON to `ResponseChunk` objects.

### Tool Call Buffering in the Agent Loop

In the streaming agent loop (`src/the_agents_playbook/loop/agent.py`), tool call chunks accumulate in a buffer keyed by `tool_call_id`. Only when the tool call is complete (a chunk arrives with the same `tool_call_id` but no new `delta_text`) are the arguments parsed and dispatched.

## Code Reference

- `src/the_agents_playbook/providers/types.py` — `ResponseChunk` (line 222), `StreamUsage` (line 234)
- `src/the_agents_playbook/providers/base.py` — `BaseProvider.stream()` (line 76), `_parse_sse_line()` (line 97)
- `src/the_agents_playbook/providers/openai.py` — OpenAI SSE chunk parsing
- `src/the_agents_playbook/providers/anthropic.py` — Anthropic SSE chunk parsing

## Playground Example

- `01-basic-calls/06-streaming.py` — SSE parsing, chunk-by-chunk token display

## LangGraph Example

- `langgraph-examples/01-basic-calls/04_streaming.py` — LangGraph streaming with `.astream()`
