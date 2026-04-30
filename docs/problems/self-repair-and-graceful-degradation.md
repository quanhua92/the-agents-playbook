# Self-Repair and Graceful Degradation

## Problem

Tools fail. APIs go down. Context windows overflow. An agent that crashes on the first error is useless in production. But retrying blindly can make things worse (hammering a downed API). The agent needs to read errors, diagnose them, attempt repair, and if all else fails, degrade gracefully — continuing to operate in a limited capacity rather than dying.

## Solution

Three complementary systems: `RepairLoop` for automatic retry on tool failure, `DegradationManager` for graceful fallback, and `SelfReviewer` for the agent to audit its own code.

### RepairLoop

`RepairLoop` (`src/the_agents_playbook/claw/repair.py:34`) wraps tool dispatch with retry logic:

- `repair(tool_name, arguments) -> RepairResult` (`repair.py:50`) — executes a tool up to `max_retries` times
- On `ToolResult(error=True)` or exception: logs the error, increments attempt counter, continues
- On success: returns `RepairResult(success=True, attempts, final_output, error_history)`
- After exhausting retries: returns `RepairResult(success=False)` with full error history

`RepairResult` (`repair.py:18`) captures: success, attempts count, final_output, and error_history (all error messages from each attempt).

`diagnose(tool_name, error)` (`repair.py:93`) produces a diagnosis string. The current implementation is a template; a real implementation would invoke the LLM to analyze the error and suggest a fix.

### DegradationManager

`DegradationManager` (`src/the_agents_playbook/claw/degradation.py:31`) handles failure modes gracefully:

- `handle_tool_failure(tool_name, error)` (`degradation.py:47`) — returns a fallback message for a failed tool. Per-tool fallback messages can be registered via `register_tool_fallback()`.
- `handle_llm_failure(error)` (`degradation.py:69`) — returns "LLM temporarily unavailable, please retry"
- `handle_context_overflow(max_tokens)` (`degradation.py:86`) — suggests compacting history or reducing input

`FallbackResult` (`degradation.py:17`) captures: handled (bool), output (user-facing message), strategy (which fallback was used).

### SelfReviewer

`SelfReviewer` (`src/the_agents_playbook/claw/self_review.py:69`) reads source files and produces quality reports:

- `review_file(file_path)` (`self_review.py:88`) — reads a file, runs heuristic analysis, returns findings
- `review_directory(pattern)` (`self_review.py:161`) — reviews all files matching a glob pattern

Heuristic checks (`self_review.py:125`):
- Lines longer than 120 characters (suggestion)
- Bare `except:` clauses (issue)
- TODO/FIXME markers (info)

`ReviewFinding` (`self_review.py:17`) captures: file_path, line_range, severity ("info"/"suggestion"/"issue"), description. `SelfReviewReport` (`self_review.py:34`) aggregates findings and computes a `clawability_score`: 1.0 = no issues, 0.0 = many issues. Formula: `max(0.0, 1.0 - (issue_count * 0.1))`.

The concept of "clawability" means the code must be simple enough for the agent itself to reason about and improve. Overly complex code can't self-repair.

## Code Reference

- `src/the_agents_playbook/claw/repair.py` — `RepairLoop` (line 34), `RepairResult` (line 18)
- `src/the_agents_playbook/claw/degradation.py` — `DegradationManager` (line 31), `FallbackResult` (line 17)
- `src/the_agents_playbook/claw/self_review.py` — `SelfReviewer` (line 69), `ReviewFinding` (line 17), `SelfReviewReport` (line 34)

## Playground Examples

- `08-the-claw/01-self-repair.py` — repair loop with retry on tool failure
- `08-the-claw/02-graceful-degradation.py` — fallback behavior when tools/API fail
- `08-the-claw/04-self-review.py` — agent reviews its own source code
- `08-the-claw/05-improvement-loop.py` — continuous improvement cycle
- `08-the-claw/06-claw-score.py` — clawability scoring

## LangGraph Examples

- `langgraph-examples/08-the-claw/01_error_edges.py` — error handling with conditional edges
- `langgraph-examples/08-the-claw/02_retry_loop.py` — retry loop pattern in a LangGraph graph
