# Tool Caching and MCP Bridge

## Problem

Agents make the same tool calls repeatedly across conversations. A web search for "Python 3.12 release date" returns the same result every time. Each call wastes tokens (sending the same request to the LLM) and latency (waiting for the tool). Meanwhile, the agent ecosystem is fragmenting — some tools live in the agent's codebase, others in external MCP (Model Context Protocol) servers. How do you cache results and bridge to external tool servers?

## Solution

Two complementary systems: `ToolResultCache` for in-memory TTL caching of tool results, and `MCPBridge` for connecting to external MCP servers.

### ToolResultCache

`ToolResultCache` (`src/the_agents_playbook/tools/cache.py:14`) caches tool results keyed by `(tool_name, SHA-256(arguments))`:

- `get(tool_name, arguments) -> ToolResult | None` — returns cached result or `None` on miss/expiry
- `set(tool_name, arguments, result, ttl)` — stores with per-entry TTL override
- `evict_expired()` — removes all stale entries, returns count evicted

Cache key generation (`cache.py:30`): `f"{tool_name}:{sha256(json(arguments, sort_keys=True))[:16]}"`. TTL tracked via `monotonic()` timestamps. Default TTL is 60 seconds.

The cache is a plain `dict` — no Redis, no disk persistence. Simple enough to use anywhere, fast enough for single-process agents.

### MCPBridge

`MCPBridge` (`src/the_agents_playbook/tools/mcp.py:55`) connects to external MCP servers over stdio:

1. `start()` (`mcp.py:123`) — launches the server as a subprocess, performs the MCP handshake (`initialize` → `initialized` notification), discovers tools via `tools/list`
2. `get_tools() -> list[Tool]` (`mcp.py:172`) — returns discovered tools as `Tool` instances, ready for `ToolRegistry.register()`
3. `stop()` (`mcp.py:176`) — terminates the subprocess

Internal protocol: JSON-RPC 2.0 over stdin/stdout (`_send_request()`, `mcp.py:78`). Each MCP tool is wrapped as a `_MCPTool` (a `Tool` subclass) that delegates execution to `tools/call` on the server.

Tools from an MCP bridge are indistinguishable from native tools — they implement the same `Tool` ABC and can be registered in the same `ToolRegistry`.

## Code Reference

- `src/the_agents_playbook/tools/cache.py` — `ToolResultCache` with TTL-based eviction
- `src/the_agents_playbook/tools/mcp.py` — `MCPBridge`, `_MCPTool`, `MCPConnectionError`
- `src/the_agents_playbook/tools/protocol.py` — `Tool` ABC (the contract both native and MCP tools implement)

## Playground Examples

- `02-tools/05-tool-caching.py` — TTL cache for tool results, cache hit/miss, eviction
- `02-tools/06-mcp-bridge.py` — connect to external MCP server, discover and use tools
