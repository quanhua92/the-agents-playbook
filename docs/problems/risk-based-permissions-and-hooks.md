# Risk-Based Permissions and Hooks

## Problem

Agents can execute dangerous operations: delete files, send emails, deploy code. Without guardrails, a misconfigured agent can cause real damage. But blocking everything makes the agent useless. You need to classify actions by risk level, auto-approve safe operations, prompt for dangerous ones, and observe all agent behavior via event hooks.

## Solution

Three-layer safety system: `RiskLevel` classifies tool operations, `PermissionMiddleware` enforces permissions, and `HookSystem` provides observability.

### RiskLevel

`RiskLevel` (`src/the_agents_playbook/guardrails/permissions.py:17`) is an enum with three tiers:

- `READ_ONLY`: auto-approve — read files, search, query memory
- `WORKSPACE_WRITE`: prompt for approval — create/edit files, run safe commands
- `DANGER`: require explicit confirmation — delete, network access, deploy

`RiskAnnotatedTool` (`permissions.py:30`) wraps a `Tool` with a `RiskLevel`. It delegates all `Tool` protocol methods to the inner tool while carrying the risk metadata.

### PermissionMiddleware

`PermissionMiddleware` (`src/the_agents_playbook/guardrails/permissions.py:65`) enforces risk-based permissions:

- `annotate(tool_name, risk)` — assign a risk level to a tool name
- `should_prompt(tool_name)` — returns `True` if the tool needs user confirmation
- `check_sync(tool_name)` — synchronous check (auto-approve only)
- `wrap_tool(tool, risk)` — wrap a `Tool` instance as a `RiskAnnotatedTool`

Default: only `READ_ONLY` tools are auto-approved. Everything else requires confirmation.

### Prompter ABC

`Prompter` (`src/the_agents_playbook/guardrails/prompter.py:13`) decouples the permission UI from agent logic:

- `TerminalPrompter` (`prompter.py:34`) — reads from stdin, displays risk label
- `SilentPrompter` (`prompter.py:56`) — auto-approves everything (tests, headless mode)
- `DenyAllPrompter` (`prompter.py:63`) — auto-denies everything (testing refusal behavior)

The agent loop doesn't know *how* the user is prompted — only *that* it needs to ask.

### HookSystem

`HookSystem` (`src/the_agents_playbook/guardrails/hooks.py:22`) provides event-driven observability:

Standard events: `ON_TURN_START`, `ON_TOOL_CALL`, `ON_TOOL_RESULT`, `ON_TURN_END`.

- `on(event, fn)` — register a handler
- `emit(event, **kwargs)` — fire all handlers for an event
- `off(event, fn)` — remove a handler

Handlers run in registration order. Errors in one handler don't prevent subsequent handlers from running (`hooks.py:81-85`).

### AskUserQuestion Tool

`AskUserQuestion` (`src/the_agents_playbook/guardrails/ask_user.py:14`) is a `Tool` that lets the agent ask the user for clarification. Instead of hallucinating an answer, the agent calls this tool. The prompter implementation controls how the question is presented.

## Code Reference

- `src/the_agents_playbook/guardrails/permissions.py` — `RiskLevel` (line 17), `RiskAnnotatedTool` (line 30), `PermissionMiddleware` (line 65)
- `src/the_agents_playbook/guardrails/prompter.py` — `Prompter` (line 13), `TerminalPrompter` (line 34)
- `src/the_agents_playbook/guardrails/hooks.py` — `HookSystem` (line 22), standard events (line 16-19)
- `src/the_agents_playbook/guardrails/ask_user.py` — `AskUserQuestion` (line 14)

## Playground Examples

- `06-human-in-the-loop/01-risk-levels.py` — risk level classification and annotation
- `06-human-in-the-loop/02-permission-middleware.py` — permission checks, auto-approve vs prompt
- `06-human-in-the-loop/03-prompter-abc.py` — Prompter ABC, terminal/silent/deny-all implementations
- `06-human-in-the-loop/04-hook-system.py` — event hooks for observability
- `06-human-in-the-loop/05-ask-user-tool.py` — agent asks user instead of guessing
- `06-human-in-the-loop/06-guarded-agent.py` — full guarded agent with all safety layers
