"""Tests for claw.evaluation — BenchmarkResult, EvaluationHarness."""

import pytest

from the_agents_playbook.claw.evaluation import (
    BenchmarkResult,
    EvaluationHarness,
    SuiteResult,
)


class TestBenchmarkResult:
    def test_defaults(self):
        r = BenchmarkResult(task="test", success=True)
        assert r.task == "test"
        assert r.success is True
        assert r.score == 1.0
        assert r.tool_calls == 0
        assert r.tokens_used == 0
        assert r.error is None

    def test_with_data(self):
        r = BenchmarkResult(
            task="fix bug",
            success=False,
            score=0.5,
            tool_calls=3,
            tokens_used=500,
            latency_seconds=2.5,
            error="timeout",
        )
        assert r.success is False
        assert r.score == 0.5
        assert r.latency_seconds == 2.5


class TestSuiteResult:
    def test_empty_suite(self):
        suite = SuiteResult()
        assert suite.pass_rate == 0.0
        assert suite.avg_score == 0.0
        assert suite.avg_latency == 0.0

    def test_all_pass(self):
        suite = SuiteResult(
            results=[
                BenchmarkResult(
                    task="t1", success=True, score=1.0, latency_seconds=1.0
                ),
                BenchmarkResult(
                    task="t2", success=True, score=0.8, latency_seconds=2.0
                ),
            ]
        )
        assert suite.passed == 2
        assert suite.failed == 0
        assert suite.pass_rate == 1.0
        assert suite.avg_score == 0.9
        assert suite.avg_latency == 1.5

    def test_mixed_results(self):
        suite = SuiteResult(
            results=[
                BenchmarkResult(task="t1", success=True, score=1.0),
                BenchmarkResult(task="t2", success=False, score=0.0),
                BenchmarkResult(task="t3", success=True, score=0.5),
            ]
        )
        assert suite.passed == 2
        assert suite.failed == 1
        assert suite.pass_rate == pytest.approx(2 / 3, abs=0.01)


class TestEvaluationHarness:
    async def test_evaluate_single(self):
        harness = EvaluationHarness()
        result = await harness.evaluate("Read auth.py", expected="bug on line 42")
        assert result.success is True
        assert len(harness.results) == 1

    async def test_evaluate_with_score(self):
        harness = EvaluationHarness()
        result = await harness.evaluate("task", score=0.75)
        assert result.score == 0.75

    async def test_run_suite(self):
        harness = EvaluationHarness()
        tasks = [
            {"task": "t1", "score": 1.0},
            {"task": "t2", "score": 0.8},
            {"task": "t3", "score": 0.6},
        ]
        suite = await harness.run_suite(tasks)
        assert suite.total_tasks == 3
        assert suite.passed == 3
        assert suite.avg_score == pytest.approx(0.8)
