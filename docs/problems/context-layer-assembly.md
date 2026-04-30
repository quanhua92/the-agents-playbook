# Context Layer Assembly

## Problem

System prompts mix multiple concerns: static instructions ("You are a coding assistant"), semi-stable context (user preferences, memory summaries), and dynamic data (current date, git status, active tool results). If you concatenate them naively, changing the git status invalidates the KV cache for the entire system prompt. If you put dynamic content first, the model's attention drifts.

## Solution

Assemble the system prompt from ordered layers with priority levels. Static layers go first (KV cache hit on every turn), dynamic layers go last (recomputed each turn).

### LayerPriority

`LayerPriority` (`src/the_agents_playbook/context/layers.py:12`) is an `IntEnum`:

- `STATIC = 0`: System instructions, tool definitions, world rules (rarely changes)
- `SEMI_STABLE = 1`: Memory summaries, skill descriptions, user preferences (changes per session)
- `DYNAMIC = 2`: Git status, date, active tool results (changes every turn)

### ContextLayer

`ContextLayer` (`layers.py:25`) is a single section of the system prompt:

```python
name: str           # Identifier for diagnostics
content: str        # The actual text
priority: int       # LayerPriority value
order: int          # Sort key within same priority
metadata: dict      # Arbitrary metadata
```

Implements `__lt__` for sorting: first by priority, then by order within the same priority.

### ContextBuilder

`ContextBuilder` (`src/the_agents_playbook/context/builder.py:22`) assembles layers into the final system prompt:

- `add_static(layer)` — shortcut that sets `priority = STATIC`
- `add_semi_stable(layer)` — shortcut that sets `priority = SEMI_STABLE`
- `add_dynamic(layer)` — shortcut that sets `priority = DYNAMIC`
- `build()` (`builder.py:80`) — sorts all layers, joins sections with `\n\n`
- `build_report()` (`builder.py:103`) — returns diagnostic breakdown (tokens per layer, over-budget flag)
- `estimated_tokens()` (`builder.py:71`) — `len(all_content) / 4`
- `token_budget_remaining()` (`builder.py:76`) — budget minus estimated tokens

### PromptTemplate

`PromptTemplate` (`src/the_agents_playbook/context/templates.py:16`) loads `.md` files from disk with `{{variable}}` substitution. Always produces `STATIC` priority layers. Common templates: `SOUL.md` (personality), `USER.md` (preferences), `AGENTS.md` (capabilities). Supports `render_with_defaults()` for merging defaults with overrides.

### Metadata Injection

`inject_date()`, `inject_cwd()`, and `inject_git_status()` (`src/the_agents_playbook/context/metadata.py:17,32,43`) generate DYNAMIC layers from the environment. `inject_git_status()` runs `git status --porcelain` and `git log --oneline -5`, gracefully degrading when not inside a git repo.

## Code Reference

- `src/the_agents_playbook/context/layers.py` — `LayerPriority` (line 12), `ContextLayer` (line 25)
- `src/the_agents_playbook/context/builder.py` — `ContextBuilder` (line 22)
- `src/the_agents_playbook/context/templates.py` — `PromptTemplate` (line 16)
- `src/the_agents_playbook/context/metadata.py` — `inject_date()`, `inject_cwd()`, `inject_git_status()`

## Playground Examples

- `04-context/01-context-layers.py` — LayerPriority ordering, layer comparison
- `04-context/02-prompt-templates.py` — load .md templates, variable substitution
- `04-context/03-metadata-injection.py` — date, CWD, git status as context layers
- `04-context/04-context-builder.py` — assemble system prompt from multiple layers
- `04-context/05-kv-cache-demo.py` — demonstrate KV cache efficiency from static-first ordering

## LangGraph Example

- `langgraph-examples/04-state/03_context_layers.py` — context layers in LangGraph state management
