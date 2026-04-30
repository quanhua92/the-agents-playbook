"""LLM-as-Judge — use a second LLM call to evaluate agent output quality.

The LLM judge pattern: instead of brittle substring matching, send
the task, result, and a rubric to a second LLM call. The judge scores
the output on multiple dimensions and provides reasoning.

This is the standard approach for evaluating agents on open-ended tasks
where there is no single correct answer.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    """Result from an LLM-as-judge evaluation.

    Attributes:
        scores: Per-criterion scores (criterion_name -> 0.0-1.0).
        reasoning: The judge's explanation for its scores.
        overall: Weighted average or overall quality score (0.0-1.0).
        raw_response: The raw LLM response for debugging.
    """

    scores: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    overall: float = 0.0
    raw_response: str = ""


class LLMJudge:
    """Evaluate agent outputs using an LLM as a judge.

    Sends the task, result, and rubric to an LLM and parses the
    structured response into per-criterion scores.

    Usage:
        judge = LLMJudge(provider=my_provider, model="gpt-4o")
        result = await judge.judge(
            task="Summarize the article about climate change",
            result="Climate change is caused by greenhouse gases...",
            criteria={
                "accuracy": "Is the summary factually accurate?",
                "completeness": "Does it cover the key points?",
                "conciseness": "Is it appropriately brief?",
            },
        )
        print(result.overall)  # e.g. 0.85
    """

    def __init__(
        self,
        provider: Any = None,
        model: str = "gpt-4o",
    ) -> None:
        self._provider = provider
        self._model = model

    @property
    def provider(self) -> Any:
        return self._provider

    async def judge(
        self,
        task: str,
        result: str,
        criteria: dict[str, str],
    ) -> JudgeResult:
        """Evaluate an agent's output against a rubric.

        Args:
            task: The original task/prompt given to the agent.
            result: The agent's output to evaluate.
            criteria: Mapping of criterion name to description.
                e.g. {"accuracy": "Is it factually correct?"}

        Returns:
            JudgeResult with per-criterion scores, reasoning, and overall.
        """
        if not self._provider:
            return self._mock_judge(task, result, criteria)

        criteria_text = "\n".join(
            f"- {name}: {desc} (score 0.0-1.0)"
            for name, desc in criteria.items()
        )

        system_prompt = (
            "You are an impartial evaluator. Score the agent's output against "
            "each criterion. Return ONLY valid JSON with this exact structure:\n"
            "{\n"
            '  "scores": {"criterion_name": 0.0},\n'
            '  "reasoning": "Brief explanation of your scores",\n'
            '  "overall": 0.0\n'
            "}\n\n"
            "Be fair but rigorous. A score of 1.0 means perfect, 0.0 means "
            "completely failed that criterion."
        )

        user_prompt = (
            f"## Task\n{task}\n\n"
            f"## Agent Output\n{result}\n\n"
            f"## Evaluation Criteria\n{criteria_text}\n\n"
            "Evaluate the agent output and return JSON."
        )

        from ..providers.types import InputMessage, MessageRequest, ToolChoice

        request = MessageRequest(
            model=self._model,
            system=system_prompt,
            messages=[InputMessage(role="user", content=user_prompt)],
            response_format=None,
            tool_choice=ToolChoice(type="auto"),
        )

        try:
            response = await self._provider.send_message(request)
            raw = response.message.content or ""
        except Exception as exc:
            logger.warning("LLM judge failed: %s", exc)
            raw = ""

        return self._parse_response(raw, criteria)

    async def judge_batch(
        self,
        evaluations: list[dict[str, str]],
        criteria: dict[str, str],
    ) -> list[JudgeResult]:
        """Judge multiple task/result pairs with the same criteria.

        Args:
            evaluations: List of {"task": ..., "result": ...} dicts.
            criteria: Evaluation rubric applied to all pairs.

        Returns:
            List of JudgeResult in the same order.
        """
        results = []
        for ev in evaluations:
            result = await self.judge(
                task=ev["task"],
                result=ev["result"],
                criteria=criteria,
            )
            results.append(result)
        return results

    def _parse_response(
        self,
        raw: str,
        criteria: dict[str, str],
    ) -> JudgeResult:
        """Parse the LLM's JSON response into a JudgeResult."""
        if not raw:
            return JudgeResult(
                reasoning="No response from judge",
                raw_response=raw,
            )

        # Try to extract JSON from the response
        try:
            # Handle markdown code blocks
            if "```" in raw:
                json_str = raw.split("```")[1]
                if json_str.startswith(("json", "JSON")):
                    json_str = json_str[4:]
                parsed = json.loads(json_str.strip())
            else:
                parsed = json.loads(raw)
        except (json.JSONDecodeError, IndexError):
            logger.warning("Could not parse judge response as JSON: %s", raw[:200])
            return JudgeResult(
                reasoning=f"Failed to parse judge response",
                raw_response=raw,
            )

        scores = {}
        for name in criteria:
            scores[name] = float(parsed.get("scores", {}).get(name, 0.0))

        reasoning = str(parsed.get("reasoning", ""))
        overall = float(parsed.get("overall", 0.0))

        # If no overall provided, compute average
        if overall == 0.0 and scores:
            overall = sum(scores.values()) / len(scores)

        return JudgeResult(
            scores=scores,
            reasoning=reasoning,
            overall=overall,
            raw_response=raw,
        )

    def _mock_judge(
        self,
        task: str,
        result: str,
        criteria: dict[str, str],
    ) -> JudgeResult:
        """Mock judge for testing without an LLM provider."""
        return JudgeResult(
            scores={name: 0.5 for name in criteria},
            reasoning="Mock evaluation — no provider configured",
            overall=0.5,
        )
