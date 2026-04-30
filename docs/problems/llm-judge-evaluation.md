# LLM-as-Judge Evaluation

## Problem

`EvaluationHarness.evaluate()` was a placeholder. The comment at line 116 of the original implementation said:

```python
# Placeholder — real implementation would run the agent
```

Without actually running agents and scoring their output, you can't measure improvement or detect regressions. Substring matching works for deterministic tasks ("the answer must contain 42") but fails for open-ended tasks like "write a summary" where there's no single correct answer.

## Solution

Two new classes provide real agent execution and two complementary scoring modes:

| Class | File | Role |
|---|---|---|
| `LLMJudge` | `claw/llm_judge.py` | Sends task + result + rubric to a second LLM, parses structured scores |
| `AgentEvaluator` | `claw/agent_evaluator.py` | Runs an actual Agent, collects events, scores the result |
| `EvalConfig` | `claw/agent_evaluator.py` | Configuration for mode, expected substring, judge criteria |

### Scoring modes

**Deterministic** — binary pass/fail via substring matching:

```python
config = EvalConfig(
    mode="deterministic",
    expected_substring="42",
)
result = await evaluator.evaluate("What is 6*7?", config)
# result.success = True  (if "42" appears in the response)
# result.score = 1.0
```

**LLM-as-judge** — nuanced per-criterion scoring via a second LLM call:

```python
config = EvalConfig(
    mode="llm_judge",
    judge_criteria={
        "accuracy": "Are the facts correct?",
        "completeness": "Does it cover the key points?",
        "clarity": "Is the response well-structured?",
    },
    judge_provider=my_provider,
    judge_model="gpt-4o",
)
result = await evaluator.evaluate("Summarize the article", config)
# result.score = 0.85 (weighted average of criterion scores)
```

### How LLMJudge works

1. Constructs a system prompt instructing the LLM to return JSON with `scores`, `reasoning`, and `overall`
2. Sends the task, agent output, and rubric as a user message
3. Parses the JSON response, handling markdown code blocks and missing fields
4. Falls back to computing `overall` as the average of criterion scores if not provided
5. Without a provider, returns a mock result (0.5 on all criteria) for testing

### How AgentEvaluator works

1. Calls `agent.run(task)` and collects all events
2. Extracts tool calls, final response text, and errors from the event stream
3. Applies the scoring mode (deterministic or LLM-judge)
4. Returns a `BenchmarkResult` with success, score, tool_calls, tokens_used, and latency

### EvaluationHarness integration

`EvaluationHarness` now accepts an optional `agent` parameter. When provided, `evaluate()` runs the agent for real:

```python
# With agent — runs it and scores
harness = EvaluationHarness(agent=my_agent)
result = await harness.evaluate("What is 2+2?", expected="4")

# Without agent — backward compatible, uses pre-computed score
harness = EvaluationHarness()
result = await harness.evaluate("task", score=0.8)
```

### JudgeResult structure

```python
@dataclass
class JudgeResult:
    scores: dict[str, float]     # {"accuracy": 0.9, "clarity": 0.7}
    reasoning: str              # "Good accuracy, but could be clearer"
    overall: float               # 0.8
    raw_response: str            # Original LLM output for debugging
```

## Code Reference

- `src/the_agents_playbook/claw/llm_judge.py` — `LLMJudge`, `JudgeResult`
- `src/the_agents_playbook/claw/agent_evaluator.py` — `AgentEvaluator`, `EvalConfig`, `AgentRunResult`
- `src/the_agents_playbook/claw/evaluation.py` — `EvaluationHarness` (updated with real agent execution)

## LangGraph Example

- `langgraph-examples/08-the-claw/04_llm_judge.py` — run a ReAct agent on tasks, use second LLM call as judge
