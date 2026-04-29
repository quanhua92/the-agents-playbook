"""02_parallel_nodes.py -- Parallel execution with Send() API.

In the root project:
  batches = workflow._execution_order()  # topological sort
  for batch in batches:
      coros = [self._execute_step(step, input) for step in batch]
      results = await asyncio.gather(*coros)  # concurrent within batch

In LangGraph:
  def route_to_items(state) -> list[Send]:
      return [Send("process", {"item": i}) for i in state["items"]]

The Send() API dynamically creates parallel branches from START.
Each branch runs independently with its own state, then results
are aggregated back via the reducer.

Note: In LangGraph v1.0+, Send is used with add_conditional_edges
from START (or a node) to fan out to multiple parallel invocations.
"""

import time
from operator import add
from typing import Annotated

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


class ParallelState(dict):
    items: list[str]
    results: Annotated[list[str], add]


def route_to_items(state: ParallelState) -> list[Send]:
    """Fan-out: create one Send per item for parallel processing."""
    items = state["items"]
    print(f"  [FAN-OUT] Creating {len(items)} parallel branches")
    return [Send("process", {"item": item}) for item in items]


def process(state: dict) -> dict:
    """Process a single item. Each invocation runs independently."""
    item = state["item"]
    start = time.monotonic()
    time.sleep(0.2)
    elapsed = time.monotonic() - start
    print(f"    [PROCESS] '{item}' done in {elapsed:.2f}s")
    return {"results": [f"Processed: {item.upper()} ({elapsed:.2f}s)"]}


def main():
    graph = StateGraph(ParallelState)
    graph.add_node("process", process)

    # Fan-out from START: each item gets its own parallel invocation
    graph.add_conditional_edges(START, route_to_items, ["process"])
    graph.add_edge("process", END)

    app = graph.compile()

    items = ["research_a", "research_b", "research_c", "research_d"]

    print("=== Parallel Execution ===\n")
    start = time.monotonic()
    result = app.invoke({"items": items, "results": []})
    elapsed = time.monotonic() - start

    print(f"\n=== Results ===")
    for r in result["results"]:
        print(f"  {r}")

    print(f"\n=== Performance ===")
    print(f"Items:         {len(items)}")
    print(f"Parallel time: {elapsed:.2f}s")
    print(f"Sequential:    {len(items) * 0.2:.2f}s")
    speedup = len(items) * 0.2 / elapsed if elapsed > 0 else 0
    print(f"Speedup:       {speedup:.1f}x")

    print("\n=== Root Comparison ===")
    print("Root: Kahn's algorithm for batches + asyncio.gather")
    print("Here: Send() from START -> parallel node invocations")
    print("Send() is more flexible: branches created at runtime from state")


if __name__ == "__main__":
    main()
