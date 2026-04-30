# Structured Output and Tool Choice

## Problem

LLMs return free-form text, but agents need structured data. A function-calling agent needs the LLM to emit valid JSON matching a specific schema. A classification agent needs the model to always call a tool (not free-text). Without structured output, you're stuck parsing fragile regex patterns from natural language.

## Solution

Use `response_format` for JSON schema enforcement and `tool_choice` to control whether the LLM calls tools or returns text.

### Response Format

`ResponseFormat` (`src/the_agents_playbook/providers/types.py:173`) controls output shape:

- `type`: `"json_object"` (legacy) or `"json_schema"` (structured)
- `json_schema_name`: name for the schema (e.g., `"FactArray"`)
- `json_schema`: the JSON Schema dict
- `strict`: enforce schema strictly (default `True`)

When `response_format` is set on `MessageRequest` (`types.py:208`), the provider includes it in the API body. The LLM must return valid JSON matching the schema.

### Tool Choice

`ToolChoice` (`src/the_agents_playbook/providers/types.py:182`) controls tool-calling behavior:

- `type="auto"`: LLM decides whether to call a tool or return text
- `type="required"`: LLM must call at least one tool
- `type="function"` with `function_name`: LLM must call a specific function

The `to_api_dict()` method (`types.py:188`) converts to the API-specific format. Set via `MessageRequest.tool_choice` (`types.py:207`).

### ToolSpec for Function Definitions

`ToolSpec` (`src/the_agents_playbook/providers/types.py:155`) defines a function the LLM can call. Each spec has `name`, `description`, and `parameters` (JSON Schema). The `to_api_dict()` method (`types.py:162`) formats for the OpenAI API. The `ToolRegistry.get_specs()` method (`src/the_agents_playbook/tools/registry.py:48`) auto-generates specs from registered tools.

## Code Reference

- `src/the_agents_playbook/providers/types.py` — `ResponseFormat` (line 173), `ToolChoice` (line 182), `ToolSpec` (line 155), `MessageRequest` (line 199)
- `src/the_agents_playbook/tools/registry.py` — `ToolRegistry.get_specs()` (line 48)

## Playground Examples

- `01-basic-calls/03-structured-output.py` — Pydantic models, JSON Schema, response format
- `01-basic-calls/04-tool-choice.py` — forcing tool calls with `tool_choice=required`

## LangGraph Example

- `langgraph-examples/01-basic-calls/02_structured_output.py` — structured output with `with_structured_output`
