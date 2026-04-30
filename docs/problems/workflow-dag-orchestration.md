# Workflow DAG Orchestration

## Problem

Complex tasks require multi-step pipelines with dependencies. "Fix the bug" means: read the file, understand the code, write a test, run the test, commit. Steps have ordering constraints (can't run the test before writing it) but some can run in parallel (read file + understand code). You need to define, validate, and execute these workflows reliably.

## Solution

Define steps as nodes in a DAG (Directed Acyclic Graph), validate with topological sort, and execute in dependency order. Independent steps run concurrently via `asyncio.gather`.

### BaseStep ABC

`BaseStep` (`src/the_agents_playbook/workflows/protocol.py:58`) defines the contract:

- `id` (property): unique step identifier
- `dependencies` (property): list of step IDs that must complete first
- `run(input_data, state) -> StepResult`: execute the step

`StepResult` (`protocol.py:19`) carries the outcome: `step_id`, `success`, `output_data`, `summary`, `updates` (key-value pairs merged into shared state), `error`.

`WorkflowEvent` (`protocol.py:40`) is the streaming contract: `step_started`, `step_completed`, `step_failed`, `workflow_completed`, `workflow_failed`.

### Workflow DAG Runner

`Workflow` (`src/the_agents_playbook/workflows/workflow.py:20`) orchestrates execution:

**Validation** (`workflow.py:56`):
1. Check all dependency references exist
2. Detect cycles via Kahn's algorithm (topological sort)
3. Returns list of error strings (empty = valid)

**Execution Order** (`workflow.py:101`):
Computes batches using topological sort. Each batch contains step IDs that can run concurrently. Example: if A has no deps and B depends on A, batch 1 = [A], batch 2 = [B]. If C also has no deps, batch 1 = [A, C], batch 2 = [B].

**Execution** (`workflow.py:171`):
1. Validate the DAG
2. For each batch: filter out steps with failed dependencies, execute remaining concurrently via `asyncio.gather`
3. On step failure: either abort entire workflow or skip and continue (`on_step_failure` config)
4. Yield `WorkflowEvent` for each lifecycle transition

### WorkflowState

`WorkflowState` (`src/the_agents_playbook/workflows/state.py`) provides shared state across steps. Steps can read from and write to `shared_context` via `StepResult.updates`. Methods: `add_result()`, `successful_steps()`, `failed_steps()`.

### Workflow Hooks

Step lifecycle hooks (`src/the_agents_playbook/workflows/hooks.py`): `PRE_STEP_EXECUTE`, `POST_STEP_EXECUTE`, `ON_STEP_FAILURE`. Fired by the workflow runner at each step transition (`workflow.py:137-167`).

## Code Reference

- `src/the_agents_playbook/workflows/protocol.py` — `BaseStep` (line 58), `StepResult` (line 19), `WorkflowEvent` (line 40)
- `src/the_agents_playbook/workflows/workflow.py` — `Workflow` (line 20) with `validate()`, `_execution_order()`, `run()`
- `src/the_agents_playbook/workflows/state.py` — `WorkflowState`
- `src/the_agents_playbook/workflows/steps.py` — step implementations
- `src/the_agents_playbook/workflows/hooks.py` — step lifecycle hooks

## Playground Examples

- `07-workflows/01-step-protocol.py` — define custom workflow steps
- `07-workflows/02-plan-and-build.py` — plan → build workflow
- `07-workflows/03-workflow-state.py` — shared state across steps
- `07-workflows/04-workflow-dag.py` — DAG validation, cycle detection, execution ordering
- `07-workflows/05-workflow-hooks.py` — step lifecycle hooks
- `07-workflows/06-concurrent-steps.py` — parallel execution of independent steps

## LangGraph Examples

- `langgraph-examples/07-workflows/01_plan_execute.py` — plan-and-execute pattern in LangGraph
- `langgraph-examples/07-workflows/02_parallel_nodes.py` — parallel node execution with `Send()`
