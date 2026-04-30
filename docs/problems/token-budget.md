# Token Budget Management

## Problem

A 128k context window does not mean you can use all 128k tokens for the prompt. You must reserve tokens for the LLM's response, system overhead, and potential tool output. Without budgeting, it's easy to exceed the limit and get a context-too-long error, or waste capacity by not knowing how much is available.

There was also no mechanism to track token consumption across requests or estimate cost per model.

## Solution

Two classes in `context/token_budget.py` manage capacity and cost:

| Class | Role |
|---|---|
| `TokenBudget` | Reserve/release token slots within a context window, track utilization |
| `UsageTracker` | Record per-request token consumption, estimate cost by model |

A thin re-export in `providers/usage.py` makes `UsageTracker` available from the providers module.

### TokenBudget

Think of it like a financial budget: you have a total (128k), mandatory expenses (response reservation), and discretionary spending (prompt layers).

```python
from the_agents_playbook.context import TokenBudget

budget = TokenBudget(total=128_000, reserved_for_response=4_096)

budget.reserve(50_000)   # True — 73,904 available
budget.reserve(30_000)   # True — 43,904 available
budget.reserve(100_000)  # False — would exceed budget
budget.available()        # 43,904

budget.release(10_000)   # Now 53,904 available
budget.utilization()     # 0.533 (53.3% of total budget used)
budget.summary()
# {'total': 128000, 'reserved': 68000, 'available': 59904,
#  'utilization': 0.531, 'reservation_count': 3}
```

### Priority-based trimming

When `reserve()` returns False, you need to trim. The existing `LayerPriority` system defines the order:

```
STATIC      (priority 0) — system prompt, tool definitions → trim last
SEMI_STABLE (priority 1) — memory summaries, user preferences
DYNAMIC     (priority 2) — git status, date, active tool results → trim first
```

Drop DYNAMIC layers first, then SEMI_STABLE, never STATIC.

### UsageTracker

Records every LLM request and can estimate cost:

```python
from the_agents_playbook.context import UsageTracker

tracker = UsageTracker()

tracker.record("gpt-4o", 1500, 200, "agent_loop")
tracker.record("gpt-4o", 3000, 500, "agent_loop")
tracker.record("gpt-4o-mini", 5000, 1000, "evaluation")

tracker.total_tokens()   # (4500, 700, 5200)
tracker.total_cost()      # 0.0231

by_model = tracker.by_model()
# {"gpt-4o": {"input_tokens": 4500, "output_tokens": 700,
#            "request_count": 2, "cost": 0.02125},
#  "gpt-4o-mini": {"input_tokens": 5000, "output_tokens": 1000,
#                    "request_count": 1, "cost": 0.00165}}
```

### Built-in pricing

Cost estimates use `MODEL_PRICING` — a dict of `(input_cost_per_1M, output_cost_per_1M)` tuples for common models:

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| gpt-4o | 2.50 | 10.00 |
| gpt-4o-mini | 0.15 | 0.60 |
| claude-sonnet-4 | 3.00 | 15.00 |
| claude-haiku-4 | 0.80 | 4.00 |

Custom pricing can be passed to the constructor:

```python
tracker = UsageTracker(custom_pricing={"my-model": (1.0, 3.0)})
```

## Code Reference

- `src/the_agents_playbook/context/token_budget.py` — `TokenBudget`, `UsageTracker`, `UsageRecord`, `MODEL_PRICING`
- `src/the_agents_playbook/providers/usage.py` — re-exports `UsageRecord` and `UsageTracker` from context

## Playground Example

- `04-context/06-token-budget.py` — build context that auto-trims when approaching budget
