"""03_evaluation.py -- Evaluation harness adapted for LangGraph agents.

In the root project:
  harness = EvaluationHarness()
  result = await harness.evaluate("Fix auth bug", expected="bug on line 42")
  suite = await harness.run_suite([task1, task2, task3])

Here we run a real LangGraph agent on each task, measure latency,
count tool calls, and check outputs against expected values.

The BenchmarkResult and SuiteResult dataclasses are adapted from
the root's EvaluationHarness.
"""

import time
from dataclasses import dataclass, field

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from shared import get_openai_llm


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@tool
def lookup_fact(topic: str) -> str:
    """Look up a fact about a topic."""
    facts = {
        "capital of france": "Paris",
        "speed of light": "299,792,458 m/s",
        "pi": "3.14159...",
    }
    return facts.get(topic.lower(), f"No fact found for '{topic}'")


@dataclass
class BenchmarkResult:
    task: str
    expected: str
    actual: str
    success: bool
    score: float = 1.0
    latency_seconds: float = 0.0
    tool_calls: int = 0


@dataclass
class SuiteResult:
    results: list[BenchmarkResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    @property
    def avg_latency(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_seconds for r in self.results) / len(self.results)

    @property
    def total_tool_calls(self) -> int:
        return sum(r.tool_calls for r in self.results)


def evaluate_agent(agent, task: str, expected: str) -> BenchmarkResult:
    """Run agent on a single task and compare output to expected."""
    start = time.monotonic()
    result = agent.invoke({"messages": [("user", task)]})
    elapsed = time.monotonic() - start

    final = result["messages"][-1].content

    # Count tool calls
    tool_count = sum(
        len(msg.tool_calls)
        for msg in result["messages"]
        if hasattr(msg, "tool_calls") and msg.tool_calls
    )

    # Simple substring match for evaluation
    success = expected.lower() in final.lower()

    return BenchmarkResult(
        task=task,
        expected=expected,
        actual=final,
        success=success,
        score=1.0 if success else 0.0,
        latency_seconds=elapsed,
        tool_calls=tool_count,
    )


def main():
    llm = get_openai_llm()
    agent = create_react_agent(llm, [calculate, lookup_fact])

    tasks = [
        {"task": "What is 2 + 3?", "expected": "5"},
        {"task": "Calculate 10 * 4", "expected": "40"},
        {"task": "What is the capital of France?", "expected": "paris"},
        {"task": "Look up the speed of light", "expected": "299"},
    ]

    suite = SuiteResult()

    print("=== Evaluation Suite ===\n")

    for task_def in tasks:
        result = evaluate_agent(agent, task_def["task"], task_def["expected"])
        suite.results.append(result)
        status = "PASS" if result.success else "FAIL"
        print(
            f"  [{status}] {result.task[:40]:40s} "
            f"tools={result.tool_calls} "
            f"time={result.latency_seconds:.2f}s"
        )
        if not result.success:
            print(f"         Expected '{result.expected}' in: {result.actual[:80]}...")

    print(f"\n=== Suite Summary ===")
    print(f"Total tasks:      {len(suite.results)}")
    print(f"Passed:           {sum(1 for r in suite.results if r.success)}")
    print(f"Failed:           {sum(1 for r in suite.results if not r.success)}")
    print(f"Pass rate:        {suite.pass_rate:.0%}")
    print(f"Avg latency:      {suite.avg_latency:.2f}s")
    print(f"Total tool calls: {suite.total_tool_calls}")

    print("\n=== Root Comparison ===")
    print("Root EvaluationHarness: placeholder (doesn't run real agents)")
    print("Here: runs real LangGraph agent, measures latency + tool calls")
    print("Both: BenchmarkResult + SuiteResult dataclasses for tracking")


if __name__ == "__main__":
    main()
