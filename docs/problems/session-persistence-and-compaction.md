# Session Persistence and Compaction

## Problem

Conversations grow until they exceed the context window. After 30 minutes of debugging, the message history might be 20,000 tokens — leaving no room for the system prompt or new messages. When you hit the limit, the API returns a `CONTEXT_TOO_LONG` error. You need to persist sessions across restarts and compress old turns to stay within budget.

## Solution

JSONL-based session persistence for save/load, and token-aware compaction to compress old messages into summaries while keeping recent turns intact.

### SessionPersistence

`SessionPersistence` (`src/the_agents_playbook/memory/session.py:12`) saves and restores conversations as JSONL files:

- `save(messages, path)` (`session.py:24`) — writes each message as a JSON line with `role`, `content`, `timestamp`
- `load(path)` (`session.py:51`) — reads JSONL back into a list of dicts, skips malformed lines
- `append(message, path)` (`session.py:77`) — appends a single message to an existing session
- `list_sessions(directory)` (`session.py:98`) — finds all `.jsonl` files in a directory

JSONL format: one JSON object per line. Append-friendly (no need to rewrite the whole file). Each message gets an ISO 8601 timestamp if not already present.

### SessionCompactor

`SessionCompactor` (`src/the_agents_playbook/memory/session.py:106`) compresses long conversations:

Token estimation (`session.py:141`): `len(text) / 4` — roughly 4 characters per English token.

Compaction logic (`session.py:154`):
1. If messages are under the token budget, return unchanged
2. Split into `old` (everything except last N) and `recent` (last N messages)
3. Build a summary from old messages by concatenating `role: content` lines
4. Return `[summary_message] + recent`

The summary is capped at 50% of the token budget to avoid being too large itself. Individual messages are truncated at 500 characters.

The `keep_recent` parameter (default 4) ensures the most recent turns are always preserved verbatim — you don't want to summarize what the user just said.

Optionally accepts a `summarize_fn` callback for LLM-based summarization instead of naive concatenation.

## Code Reference

- `src/the_agents_playbook/memory/session.py` — `SessionPersistence` (line 12), `SessionCompactor` (line 106)

## Playground Example

- `03-memory/05-session-persistence.py` — save/load sessions as JSONL, compact long conversations

## LangGraph Examples

- `langgraph-examples/03-memory/03_long_context.py` — handling long context windows in LangGraph
- `langgraph-examples/shared/compactor.py` — shared compaction utility for LangGraph examples
