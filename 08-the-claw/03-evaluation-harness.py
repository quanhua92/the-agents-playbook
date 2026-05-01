"""03-evaluation-harness.py — Run benchmarks, measure agent performance.

EvaluationHarness runs tasks and tracks success, scores, tokens,
and latency. SuiteResult aggregates metrics across a benchmark.
"""

import asyncio

from the_agents_playbook.claw.evaluation import EvaluationHarness


async def main():
    # --- Single evaluation ---

    print("=== Single Evaluation ===")
    harness = EvaluationHarness()
    result = await harness.evaluate(
        "Read auth.py and list bugs", expected="line 42 bug"
    )
    print(f"  Task:       {result.task}")
    print(f"  Success:    {result.success}")
    print(f"  Score:      {result.score}")
    print(f"  Latency:    {result.latency_seconds:.4f}s")
    print()

    # --- Evaluation with custom score ---

    print("=== Custom Score ===")
    result = await harness.evaluate("Fix the typo", score=0.65)
    print(f"  Score: {result.score}")
    print()

    # --- Benchmark suite ---

    print("=== Benchmark Suite ===")
    tasks = [
        {"task": "Fix auth bug in login.py", "score": 0.9},
        {"task": "Add unit tests for parser", "score": 0.8},
        {"task": "Refactor API endpoint", "score": 1.0},
        {"task": "Debug memory leak", "score": 0.4},
        {"task": "Update README", "score": 0.7},
    ]
    suite = await harness.run_suite(tasks)

    print(f"  Total:      {suite.total_tasks} tasks")
    print(f"  Passed:     {suite.passed}")
    print(f"  Failed:     {suite.failed}")
    print(f"  Pass rate:  {suite.pass_rate:.1%}")
    print(f"  Avg score:  {suite.avg_score:.2f}")
    print(f"  Avg latency: {suite.avg_latency:.4f}s")
    print()

    # --- Compare two suites ---

    print("=== Suite Comparison ===")
    suite_a = await EvaluationHarness().run_suite(
        [
            {"task": "t1", "score": 0.6},
            {"task": "t2", "score": 0.7},
        ]
    )
    suite_b = await EvaluationHarness().run_suite(
        [
            {"task": "t1", "score": 0.9},
            {"task": "t2", "score": 0.95},
        ]
    )
    print(f"  Before: avg_score = {suite_a.avg_score:.2f}")
    print(f"  After:  avg_score = {suite_b.avg_score:.2f}")
    improvement = suite_b.avg_score - suite_a.avg_score
    print(f"  Improvement: +{improvement:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
