"""Evaluation harness — automated benchmarks to measure agent performance.

The EvaluationHarness runs single tasks or benchmark suites, tracking
success, tool calls, tokens, and latency. Results can be compared
over time to measure improvement.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of evaluating a single task.

    Attributes:
        task: The task description or prompt.
        success: Whether the task was completed successfully.
        score: Numeric score (0.0–1.0) for the result quality.
        tool_calls: Number of tool calls made.
        tokens_used: Total tokens consumed.
        latency_seconds: Wall-clock time for the evaluation.
        error: Error message if evaluation failed.
    """

    task: str
    success: bool
    score: float = 1.0
    tool_calls: int = 0
    tokens_used: int = 0
    latency_seconds: float = 0.0
    error: str | None = None


@dataclass
class SuiteResult:
    """Aggregated results from a benchmark suite."""

    results: list[BenchmarkResult] = field(default_factory=list)
    total_tasks: int = 0
    passed: int = 0
    failed: int = 0
    total_latency: float = 0.0

    def __post_init__(self) -> None:
        for r in self.results:
            self.total_tasks += 1
            self.passed += 1 if r.success else 0
            self.failed += 0 if r.success else 1
            self.total_latency += r.latency_seconds

    @property
    def pass_rate(self) -> float:
        """Fraction of tasks that passed (0.0–1.0)."""
        if self.total_tasks == 0:
            return 0.0
        return self.passed / self.total_tasks

    @property
    def avg_score(self) -> float:
        """Average score across all tasks."""
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def avg_latency(self) -> float:
        """Average latency per task in seconds."""
        if not self.results:
            return 0.0
        return self.total_latency / len(self.results)


class EvaluationHarness:
    """Run and track agent evaluations.

    Usage:
        harness = EvaluationHarness()
        result = await harness.evaluate("Read auth.py and find bugs", expected="bug on line 42")
        suite = await harness.run_suite([task1, task2, task3])
    """

    def __init__(self) -> None:
        self._results: list[BenchmarkResult] = []

    @property
    def results(self) -> list[BenchmarkResult]:
        return list(self._results)

    async def evaluate(
        self,
        task: str,
        expected: str | None = None,
        score: float | None = None,
    ) -> BenchmarkResult:
        """Evaluate a single task.

        In a real implementation, this runs the agent on the task and
        compares the output against expected. Here it returns a
        placeholder result.

        Args:
            task: The task description.
            expected: Optional expected output for scoring.
            score: Optional pre-computed score.

        Returns:
            BenchmarkResult for this task.
        """
        start = time.monotonic()
        # Placeholder — real implementation would run the agent
        result = BenchmarkResult(
            task=task,
            success=True,
            score=score if score is not None else 1.0,
            tool_calls=0,
            tokens_used=0,
            latency_seconds=time.monotonic() - start,
        )
        self._results.append(result)
        logger.info("Evaluated task: %s (score=%.2f)", task[:50], result.score)
        return result

    async def run_suite(self, tasks: list[dict[str, Any]]) -> SuiteResult:
        """Run a suite of evaluation tasks.

        Args:
            tasks: List of dicts with keys "task", "expected" (optional), "score" (optional).

        Returns:
            SuiteResult with aggregated metrics.
        """
        suite = SuiteResult(total_tasks=len(tasks))

        for task_def in tasks:
            task_text = task_def.get("task", "")
            expected = task_def.get("expected")
            score = task_def.get("score")

            result = await self.evaluate(task_text, expected=expected, score=score)
            suite.results.append(result)
            suite.total_latency += result.latency_seconds

            if result.success:
                suite.passed += 1
            else:
                suite.failed += 1

        logger.info(
            "Suite complete: %d/%d passed (avg score=%.2f)",
            suite.passed, suite.total_tasks, suite.avg_score,
        )
        return suite
