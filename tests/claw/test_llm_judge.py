"""Tests for LLM-as-Judge evaluation."""

import json
import pytest

from the_agents_playbook.claw.llm_judge import JudgeResult, LLMJudge


class TestJudgeResult:
    def test_defaults(self):
        result = JudgeResult()
        assert result.scores == {}
        assert result.reasoning == ""
        assert result.overall == 0.0
        assert result.raw_response == ""

    def test_with_data(self):
        result = JudgeResult(
            scores={"accuracy": 0.9, "clarity": 0.7},
            reasoning="Good overall",
            overall=0.8,
        )
        assert len(result.scores) == 2
        assert result.overall == 0.8


class TestLLMJudge:
    def test_init_no_provider(self):
        judge = LLMJudge()
        assert judge.provider is None

    def test_init_with_provider(self):
        mock = object()
        judge = LLMJudge(provider=mock, model="gpt-4o-mini")
        assert judge.provider is mock

    def test_mock_judge(self):
        """Without a provider, judge returns mock scores."""
        judge = LLMJudge()

        result = judge._mock_judge(
            task="test task",
            result="test result",
            criteria={"accuracy": "Is it accurate?"},
        )

        assert result.overall == 0.5
        assert "accuracy" in result.scores
        assert result.scores["accuracy"] == 0.5

    def test_parse_valid_json(self):
        judge = LLMJudge()
        criteria = {"accuracy": "Is it correct?", "clarity": "Is it clear?"}

        raw = json.dumps({
            "scores": {"accuracy": 0.9, "clarity": 0.8},
            "reasoning": "Both good",
            "overall": 0.85,
        })

        result = judge._parse_response(raw, criteria)

        assert result.scores["accuracy"] == 0.9
        assert result.scores["clarity"] == 0.8
        assert result.overall == 0.85
        assert result.reasoning == "Both good"

    def test_parse_json_in_markdown_block(self):
        judge = LLMJudge()
        criteria = {"accuracy": "correct?"}

        raw = '```json\n{"scores": {"accuracy": 0.7}, "reasoning": "ok", "overall": 0.7}\n```'

        result = judge._parse_response(raw, criteria)
        assert result.scores["accuracy"] == 0.7

    def test_parse_invalid_json(self):
        judge = LLMJudge()
        criteria = {"accuracy": "correct?"}

        result = judge._parse_response("not json at all", criteria)
        assert result.overall == 0.0
        assert result.reasoning == "Failed to parse judge response"

    def test_parse_empty_response(self):
        judge = LLMJudge()
        criteria = {"accuracy": "correct?"}

        result = judge._parse_response("", criteria)
        assert result.reasoning == "No response from judge"

    def test_parse_missing_overall_computes_average(self):
        judge = LLMJudge()
        criteria = {"a": "first", "b": "second"}

        raw = json.dumps({
            "scores": {"a": 0.8, "b": 0.6},
            "reasoning": "decent",
        })

        result = judge._parse_response(raw, criteria)
        assert result.overall == pytest.approx(0.7)

    def test_parse_missing_criteria_scored_as_zero(self):
        judge = LLMJudge()
        criteria = {"a": "first", "b": "second"}

        raw = json.dumps({
            "scores": {"a": 0.9},  # 'b' missing
            "reasoning": "partial",
            "overall": 0.5,
        })

        result = judge._parse_response(raw, criteria)
        assert result.scores["a"] == 0.9
        assert result.scores["b"] == 0.0
        assert result.overall == 0.5


class TestAgentEvaluator:
    async def test_deterministic_pass(self):
        from the_agents_playbook.claw.agent_evaluator import (
            AgentEvaluator, AgentRunResult, EvalConfig,
        )

        # Create a mock agent that yields a text event
        class FakeEvent:
            def __init__(self, type, data):
                self.type = type
                self.data = data

        class FakeAgent:
            async def run(self, prompt):
                yield FakeEvent("status", {"message": "thinking"})
                yield FakeEvent("text", {"text": "The answer is 42"})

        evaluator = AgentEvaluator(FakeAgent())
        result = await evaluator.evaluate(
            task="What is the meaning of life?",
            config=EvalConfig(
                mode="deterministic",
                expected_substring="42",
            ),
        )

        assert result.success is True
        assert result.score == 1.0

    async def test_deterministic_fail(self):
        from the_agents_playbook.claw.agent_evaluator import AgentEvaluator, EvalConfig

        class FakeEvent:
            def __init__(self, type, data):
                self.type = type
                self.data = data

        class FakeAgent:
            async def run(self, prompt):
                yield FakeEvent("text", {"text": "I don't know"})

        evaluator = AgentEvaluator(FakeAgent())
        result = await evaluator.evaluate(
            task="What is 2+2?",
            config=EvalConfig(
                mode="deterministic",
                expected_substring="4",
            ),
        )

        assert result.success is False
        assert result.score == 0.0

    async def test_no_expected_passes_with_any_response(self):
        from the_agents_playbook.claw.agent_evaluator import AgentEvaluator, EvalConfig

        class FakeEvent:
            def __init__(self, type, data):
                self.type = type
                self.data = data

        class FakeAgent:
            async def run(self, prompt):
                yield FakeEvent("text", {"text": "Something"})

        evaluator = AgentEvaluator(FakeAgent())
        result = await evaluator.evaluate(
            task="Hello",
            config=EvalConfig(mode="deterministic"),
        )

        assert result.success is True

    async def test_agent_error(self):
        from the_agents_playbook.claw.agent_evaluator import AgentEvaluator, EvalConfig

        class FakeAgent:
            async def run(self, prompt):
                yield type("E", (), {"type": "error", "data": {"message": "boom"}})()

        evaluator = AgentEvaluator(FakeAgent())
        result = await evaluator.evaluate(
            task="fail task",
            config=EvalConfig(mode="deterministic", expected_substring="anything"),
        )

        assert result.success is False
        assert result.error == "boom"

    async def test_collects_tool_calls(self):
        from the_agents_playbook.claw.agent_evaluator import AgentEvaluator, EvalConfig

        class FakeEvent:
            def __init__(self, type, data):
                self.type = type
                self.data = data

        class FakeAgent:
            async def run(self, prompt):
                yield FakeEvent("tool_call", {"tool_name": "calc", "arguments": {}})
                yield FakeEvent("tool_result", {"output": "42"})
                yield FakeEvent("text", {"text": "Answer: 42"})

        evaluator = AgentEvaluator(FakeAgent())
        result = await evaluator.evaluate(
            task="Calculate 6*7",
            config=EvalConfig(mode="deterministic", expected_substring="42"),
        )

        assert result.success is True
        assert result.tool_calls == 1
