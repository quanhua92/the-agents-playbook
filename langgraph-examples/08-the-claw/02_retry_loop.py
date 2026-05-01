"""02_retry_loop.py -- Retry pattern with conditional edges and backoff.

In the root project:
  class RepairLoop:
      async def repair(self, tool_name, arguments) -> RepairResult:
          for attempt in range(1, self._max_retries + 1):
              result = await self._registry.dispatch(tool_name, arguments)
              if not result.error:
                  return RepairResult(success=True, attempts=attempt, ...)
          return RepairResult(success=False, ...)

In LangGraph, retries are modeled as a graph loop:
  attempt -> should_retry -> (retry: loop back | done | fail)

The graph itself is the retry mechanism. Exponential backoff can be
added inside the attempt node.
"""

import time
from typing import Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class RetryState(TypedDict):
    attempt: int
    max_attempts: int
    success: bool
    errors: list[str]
    result: str


def attempt_operation(state: RetryState) -> dict:
    """Simulate an operation that fails before succeeding."""
    attempt = state["attempt"]

    if attempt < 2:
        # Exponential backoff (using time.sleep for sync context)
        delay = min(0.1 * (2**attempt), 1.0)
        time.sleep(delay)
        error_msg = f"Attempt {attempt + 1}: connection refused (waited {delay:.1f}s)"
        print(f"  [ATTEMPT] {error_msg}")
        return {
            "attempt": attempt + 1,
            "errors": state["errors"] + [error_msg],
        }

    print(f"  [ATTEMPT] Attempt {attempt + 1}: success!")
    return {
        "attempt": attempt + 1,
        "success": True,
        "result": "Data retrieved successfully",
    }


def should_retry(state: RetryState) -> Literal["retry", "fail", "done"]:
    if state["success"]:
        return "done"
    if state["attempt"] >= state["max_attempts"]:
        return "fail"
    return "retry"


def fail_node(state: RetryState) -> dict:
    errors = "\n".join(f"  - {e}" for e in state["errors"])
    return {"result": f"Failed after {state['attempt']} attempts:\n{errors}"}


def done_node(state: RetryState) -> dict:
    return {"result": state["result"]}


def main():
    graph = StateGraph(RetryState)
    graph.add_node("attempt", attempt_operation)
    graph.add_node("fail", fail_node)
    graph.add_node("done", done_node)

    graph.add_edge(START, "attempt")
    graph.add_conditional_edges(
        "attempt",
        should_retry,
        {"retry": "attempt", "fail": "fail", "done": "done"},
    )
    graph.add_edge("fail", END)
    graph.add_edge("done", END)

    app = graph.compile()

    print("=== Retry Loop ===\n")
    start = time.monotonic()
    result = app.invoke(
        {
            "attempt": 0,
            "max_attempts": 5,
            "success": False,
            "errors": [],
            "result": "",
        }
    )
    elapsed = time.monotonic() - start

    print("\n=== Result ===")
    print(result["result"])
    print(f"\nAttempts: {result['attempt']}")
    print(f"Total time: {elapsed:.2f}s (includes backoff delays)")

    print("\n=== Exhaust Retries Scenario ===\n")
    # Lower max so it actually fails
    result2 = app.invoke(
        {
            "attempt": 0,
            "max_attempts": 1,  # Only 1 attempt -> will fail
            "success": False,
            "errors": [],
            "result": "",
        }
    )
    print(f"Result: {result2['result']}")

    print("\n=== Root Comparison ===")
    print("Root RepairLoop:     imperative retry wrapper (try/except/for loop)")
    print(
        "LangGraph:           graph edge loop (attempt -> conditional -> retry/done/fail)"
    )
    print("Same outcome, different paradigm.")


if __name__ == "__main__":
    main()
