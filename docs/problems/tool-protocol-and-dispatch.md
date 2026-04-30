# Tool Protocol and Dispatch

## Problem

An agent needs to call tools — search the web, read files, run shell commands. But tools have different signatures, error modes, and schemas. How do you define a uniform interface? How does the LLM's JSON function-call get validated and routed to the right Python function? How do you handle malformed arguments, missing tools, and timeouts?

## Solution

Three-layer architecture: `Tool` ABC defines the contract, `ToolRegistry` manages discovery, and `ToolDispatcher` handles parsing, validation, and execution.

### Tool ABC

`Tool` (`src/the_agents_playbook/tools/protocol.py:17`) is the contract every tool implements:

- `name` (property): unique string identifier, sent to the LLM
- `description` (property): what the tool does, sent to the LLM for selection
- `parameters` (property): JSON Schema dict describing accepted arguments
- `execute(**kwargs) -> ToolResult`: the actual implementation

`ToolResult` (`protocol.py:8`) wraps the outcome with `output: str`, `error: bool`, and `metadata: dict`.

### ToolRegistry

`ToolRegistry` (`src/the_agents_playbook/tools/registry.py:20`) stores tools by name and generates `ToolSpec` objects for the LLM request:

- `register(tool)` — add a tool
- `get(name)` — lookup, raises `ToolNotFoundError` (`registry.py:12`) if missing
- `get_specs()` — returns `list[ToolSpec]` ready for `MessageRequest.tools`
- `dispatch(tool_name, arguments)` — lookup and execute

### ToolDispatcher

`ToolDispatcher` (`src/the_agents_playbook/tools/dispatcher.py:31`) bridges the LLM's JSON output to the registry:

1. `parse_arguments()` (`dispatcher.py:48`) — parses the JSON string from the LLM into a dict
2. `validate_arguments()` (`dispatcher.py:64`) — checks required properties, rejects unknowns, basic type checking
3. `dispatch_one()` (`dispatcher.py:95`) — parse + validate + execute, catches all errors into `ToolResult(error=True)`
4. `dispatch_all()` (`dispatcher.py:125`) — batch dispatch for multiple parallel tool calls

Error handling: `ToolNotFoundError`, `ToolArgumentError` (`dispatcher.py:13`), `ToolTimeoutError` (`dispatcher.py:22`). All are caught by `dispatch_one()` and converted to `ToolResult` with `error=True` so the agent loop can feed the error back to the LLM for self-repair.

## Code Reference

- `src/the_agents_playbook/tools/protocol.py` — `ToolResult` (line 8), `Tool` ABC (line 17)
- `src/the_agents_playbook/tools/registry.py` — `ToolRegistry` (line 20), `ToolNotFoundError` (line 12)
- `src/the_agents_playbook/tools/dispatcher.py` — `ToolDispatcher` (line 31), `ToolArgumentError` (line 13)

## Playground Examples

- `02-tools/01-tool-protocol.py` — define a custom tool implementing the `Tool` ABC
- `02-tools/02-tool-registry.py` — register tools, generate specs, dispatch
- `02-tools/03-tool-dispatcher.py` — argument parsing, validation, error handling
- `02-tools/04-built-in-tools.py` — built-in tool implementations (shell, file read, etc.)

## LangGraph Examples

- `langgraph-examples/02-tools/01_tool_protocol.py` — tool protocol with LangGraph
- `langgraph-examples/02-tools/02_tool_node.py` — `ToolNode` for automatic dispatch
- `langgraph-examples/02-tools/03_bound_tools.py` — binding tools to specific models
