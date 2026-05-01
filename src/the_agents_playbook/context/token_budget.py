"""Token budget — manage context window capacity like a financial budget.

Context windows are finite. Before sending a request, reserve tokens
for the response. Priority-trim dynamic layers first when over budget.
Track usage per model for cost control.

Key insight: a 128k context window does NOT mean you can use all 128k
for the prompt. You must reserve tokens for the response, system
overhead, and tool output.
"""

import logging
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TokenBudget:
    """Manage available tokens in a context window.

    Tracks total capacity, reserved amounts, and available tokens.
    When approaching the limit, callers should trim lower-priority
    context layers first.

    Usage:
        budget = TokenBudget(total=128_000, reserved_for_response=4096)
        budget.reserve(50_000)  # True — 78k remaining
        budget.reserve(30_000)  # True — 48k remaining
        budget.reserve(50_000)  # False — would exceed budget
        budget.available()       # 48000
        budget.release(10_000)   # now 58000 available

    Attributes:
        total: Total context window size in tokens.
        reserved_for_response: Tokens held back for the LLM response.
        _reserved: Currently reserved tokens (prompt + overhead).
        _reservations: Log of individual reservations for auditing.
    """

    total: int
    reserved_for_response: int = 4096
    _reserved: int = field(default=0, init=False)
    _reservations: list[dict[str, Any]] = field(default_factory=list, init=False)

    @property
    def available(self) -> int:
        """Tokens available for additional reservations."""
        return max(0, self.total - self.reserved_for_response - self._reserved)

    def reserve(self, tokens: int) -> bool:
        """Attempt to reserve tokens. Returns True if successful.

        Args:
            tokens: Number of tokens to reserve.

        Returns:
            True if the reservation was made, False if insufficient budget.
        """
        if tokens < 0:
            logger.warning("Negative reservation (%d) — ignoring", tokens)
            return False

        if self._reserved + tokens > self.total - self.reserved_for_response:
            logger.debug(
                "Reservation failed: need %d, have %d available",
                tokens,
                self.available,
            )
            return False

        self._reserved += tokens
        self._reservations.append(
            {
                "tokens": tokens,
                "reserved_total": self._reserved,
                "available": self.available,
                "timestamp": monotonic(),
            }
        )
        return True

    def release(self, tokens: int) -> None:
        """Release previously reserved tokens.

        Args:
            tokens: Number of tokens to release (capped at current reservation).
        """
        tokens = min(tokens, self._reserved)
        self._reserved -= tokens
        logger.debug("Released %d tokens (now %d reserved)", tokens, self._reserved)

    def release_all(self) -> None:
        """Release all reserved tokens."""
        self._reserved = 0

    def utilization(self) -> float:
        """Fraction of total budget that is reserved (0.0-1.0)."""
        if self.total == 0:
            return 0.0
        return self._reserved / self.total

    def summary(self) -> dict[str, Any]:
        """Return a summary dict for logging/diagnostics."""
        return {
            "total": self.total,
            "reserved_for_response": self.reserved_for_response,
            "reserved": self._reserved,
            "available": self.available,
            "utilization": round(self.utilization(), 3),
            "reservation_count": len(self._reservations),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"TokenBudget(total={s['total']}, reserved={s['reserved']}, "
            f"available={s['available']}, utilized={s['utilization']:.1%})"
        )


@dataclass
class UsageRecord:
    """A single token usage record.

    Attributes:
        model: Model identifier (e.g., "gpt-4o").
        input_tokens: Tokens sent to the model.
        output_tokens: Tokens received from the model.
        source: What generated this usage (e.g., "agent_loop", "evaluation").
        timestamp: When the usage occurred.
    """

    model: str
    input_tokens: int
    output_tokens: int
    source: str = ""
    timestamp: float = field(default_factory=monotonic)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# Approximate cost per 1M tokens (USD) for common models.
# Update these as pricing changes.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # (input_cost_per_1M, output_cost_per_1M)
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-haiku-4-20250414": (0.80, 4.00),
}


class UsageTracker:
    """Track token usage across models for cost monitoring.

    Usage:
        tracker = UsageTracker()
        tracker.record("gpt-4o", 1500, 200, "agent_loop")
        tracker.record("gpt-4o", 800, 100, "evaluation")
        print(tracker.total_cost())  # e.g. $0.044
        print(tracker.summary())
    """

    def __init__(
        self, custom_pricing: dict[str, tuple[float, float]] | None = None
    ) -> None:
        self._records: list[UsageRecord] = []
        self._pricing = {**MODEL_PRICING}
        if custom_pricing:
            self._pricing.update(custom_pricing)

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        source: str = "",
    ) -> UsageRecord:
        """Record a single usage event.

        Args:
            model: Model identifier.
            input_tokens: Input tokens consumed.
            output_tokens: Output tokens generated.
            source: What generated this usage.

        Returns:
            The UsageRecord that was stored.
        """
        record = UsageRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            source=source,
        )
        self._records.append(record)
        return record

    def total_tokens(self) -> tuple[int, int, int]:
        """Return (total_input, total_output, total_all) across all records."""
        total_in = sum(r.input_tokens for r in self._records)
        total_out = sum(r.output_tokens for r in self._records)
        return total_in, total_out, total_in + total_out

    def total_cost(self) -> float:
        """Estimate total cost in USD based on model pricing."""
        cost = 0.0
        for r in self._records:
            pricing = self._pricing.get(r.model)
            if pricing:
                input_cost = r.input_tokens * pricing[0] / 1_000_000
                output_cost = r.output_tokens * pricing[1] / 1_000_000
                cost += input_cost + output_cost
            else:
                logger.debug(
                    "No pricing for model %s — skipping cost estimate", r.model
                )
        return cost

    def by_model(self) -> dict[str, dict[str, Any]]:
        """Aggregate usage by model.

        Returns:
            Dict mapping model name to {input_tokens, output_tokens,
            total_tokens, cost, request_count}.
        """
        models: dict[str, dict[str, Any]] = {}
        for r in self._records:
            if r.model not in models:
                models[r.model] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "request_count": 0,
                }
            models[r.model]["input_tokens"] += r.input_tokens
            models[r.model]["output_tokens"] += r.output_tokens
            models[r.model]["request_count"] += 1

        for model, data in models.items():
            pricing = self._pricing.get(model)
            if pricing:
                data["cost"] = (
                    data["input_tokens"] * pricing[0] / 1_000_000
                    + data["output_tokens"] * pricing[1] / 1_000_000
                )
            else:
                data["cost"] = None
            data["total_tokens"] = data["input_tokens"] + data["output_tokens"]

        return models

    def summary(self) -> dict[str, Any]:
        """Return a full usage summary."""
        total_in, total_out, total_all = self.total_tokens()
        return {
            "total_requests": len(self._records),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_all,
            "estimated_cost_usd": round(self.total_cost(), 6),
            "by_model": self.by_model(),
        }
