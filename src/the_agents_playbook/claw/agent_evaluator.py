"""Agent evaluator — actually run agents and collect results for evaluation.

Fixes the placeholder in EvaluationHarness by providing a real implementation
that runs an Agent, collects events, extracts tool calls, timing, and the
final response, then scores the result.

Supports two evaluation modes:
- Deterministic: substring matching for binary pass/fail
- LLM-as-judge: nuanced quality scoring with explicit rubrics
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .llm_judge import LLMJudge

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for agent evaluation.

    Attributes:
        mode: "deterministic" (substring match) or "llm_judge".
        timeout_seconds: Max time to wait for the agent to complete a task.
        expected_substring: For deterministic mode, string that must appear
            in the final response to pass.
        judge_criteria: For llm_judge mode, rubric for the LLM judge.
        judge_provider: Optional LLM provider for the judge.
        judge_model: Model to use for the LLM judge.
    """

    mode: str = "deterministic"  # "deterministic" or "llm_judge"
    timeout_seconds: float = 60.0
    expected_substring: str | None = None
    judge_criteria: dict[str, str] = field(default_factory=dict)
    judge_provider: Any = None
    judge_model: str = "gpt-4o"


@dataclass
class AgentRunResult:
    """Collected data from a single agent run.

    Attributes:
        events: All events yielded by the agent.
        tool_calls: List of tool calls made.
        tool_call_count: Total number of tool calls.
        final_response: The agent's final text response.
        error: Error message if the agent failed.
        latency_seconds: Total wall-clock time for the run.
        token_estimate: Rough token estimate (not available without provider support).
    """

    events: list[Any] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_count: int = 0
    final_response: str | None = None
    error: str | None = None
    latency_seconds: float = 0.0
    token_estimate: int = 0


class AgentEvaluator:
    """Run agents and evaluate their performance.

    Usage:
        evaluator = AgentEvaluator(agent=my_agent)
        result = await evaluator.evaluate(
            task="What is 2+2?",
            config=EvalConfig(
                mode="deterministic",
                expected_substring="4",
            ),
        )
        print(result.success)  # True if "4" is in the response
    """

    def __init__(self, agent: Any) -> None:
        """Initialize with an agent instance.

        Args:
            agent: An object with a run(prompt) async generator method,
                like the Agent class from the loop module.
        """
        self._agent = agent
        self._judge: LLMJudge | None = None

    async def evaluate(
        self,
        task: str,
        config: EvalConfig | None = None,
    ) -> "BenchmarkResult":  # noqa: F821
        """Run the agent on a task and evaluate the result.

        Args:
            task: The task description or prompt.
            config: Evaluation configuration. Defaults to deterministic mode.

        Returns:
            BenchmarkResult with success, score, and metadata.
        """
        config = config or EvalConfig()
        start = time.monotonic()

        # Run the agent and collect results
        run_result = await self._run_agent(task, config.timeout_seconds)
        latency = time.monotonic() - start

        # Score based on mode
        if config.mode == "llm_judge":
            score, success = await self._score_with_judge(
                task,
                run_result,
                config,
            )
        else:
            score, success = self._score_deterministic(
                run_result,
                config.expected_substring,
            )

        from .evaluation import BenchmarkResult

        return BenchmarkResult(
            task=task,
            success=success,
            score=score,
            tool_calls=run_result.tool_call_count,
            tokens_used=run_result.token_estimate,
            latency_seconds=latency,
            error=run_result.error,
        )

    async def evaluate_suite(
        self,
        tasks: list[dict[str, Any]],
        config: EvalConfig | None = None,
    ) -> "SuiteResult":  # noqa: F821
        """Evaluate multiple tasks and return aggregated results.

        Args:
            tasks: List of dicts with "task" key and optional "expected",
                "criteria" keys.
            config: Base evaluation configuration. Task-specific overrides
                take precedence.

        Returns:
            SuiteResult with aggregated metrics.
        """
        from .evaluation import BenchmarkResult, SuiteResult

        results: list[BenchmarkResult] = []
        for task_def in tasks:
            task_config = EvalConfig(
                mode=task_def.get("mode", config.mode if config else "deterministic"),
                expected_substring=task_def.get(
                    "expected", config.expected_substring if config else None
                ),
                judge_criteria=task_def.get(
                    "criteria", config.judge_criteria if config else {}
                ),
                judge_provider=task_def.get(
                    "judge_provider", config.judge_provider if config else None
                ),
                judge_model=task_def.get(
                    "judge_model", config.judge_model if config else "gpt-4o"
                ),
            )
            result = await self.evaluate(task_def["task"], task_config)
            results.append(result)

        return SuiteResult(results=results)

    async def _run_agent(
        self,
        task: str,
        timeout_seconds: float,
    ) -> AgentRunResult:
        """Execute the agent and collect all events."""
        run_result = AgentRunResult()

        try:
            async for event in self._agent.run(task):
                run_result.events.append(event)
                event_type = getattr(event, "type", None)
                event_data = getattr(event, "data", {})

                if event_type == "tool_call":
                    run_result.tool_calls.append(event_data)
                    run_result.tool_call_count += 1
                elif event_type == "text":
                    run_result.final_response = event_data.get("text")
                elif event_type == "error":
                    run_result.error = event_data.get("message")
        except Exception as exc:
            run_result.error = str(exc)
            logger.error("Agent run failed: %s", exc)

        return run_result

    def _score_deterministic(
        self,
        run_result: AgentRunResult,
        expected_substring: str | None,
    ) -> tuple[float, bool]:
        """Binary pass/fail based on substring matching."""
        if run_result.error:
            return 0.0, False

        if not expected_substring:
            # No expected string — pass if we got any response
            return (1.0, True) if run_result.final_response else (0.0, False)

        if (
            run_result.final_response
            and expected_substring in run_result.final_response
        ):
            return 1.0, True

        return 0.0, False

    async def _score_with_judge(
        self,
        task: str,
        run_result: AgentRunResult,
        config: EvalConfig,
    ) -> tuple[float, bool]:
        """Score using LLM-as-judge."""
        if run_result.error or not run_result.final_response:
            return 0.0, False

        if not config.judge_criteria:
            logger.warning(
                "LLM judge mode but no criteria — falling back to deterministic"
            )
            return 1.0, True

        if self._judge is None:
            self._judge = LLMJudge(
                provider=config.judge_provider,
                model=config.judge_model,
            )

        judge_result = await self._judge.judge(
            task=task,
            result=run_result.final_response,
            criteria=config.judge_criteria,
        )

        return judge_result.overall, judge_result.overall >= 0.6
