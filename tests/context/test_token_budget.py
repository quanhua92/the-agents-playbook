"""Tests for token budget management and usage tracking."""

import pytest

from the_agents_playbook.context.token_budget import (
    TokenBudget,
    UsageRecord,
    UsageTracker,
)


class TestTokenBudget:
    def test_initial_state(self):
        budget = TokenBudget(total=128_000)
        assert budget.available == 128_000 - 4096  # default reserved_for_response

    def test_custom_response_reserve(self):
        budget = TokenBudget(total=128_000, reserved_for_response=8192)
        assert budget.available == 128_000 - 8192

    def test_reserve_success(self):
        budget = TokenBudget(total=128_000)
        assert budget.reserve(50_000) is True
        assert budget.available == 128_000 - 4096 - 50_000

    def test_reserve_exact(self):
        budget = TokenBudget(total=128_000)
        available = budget.available
        assert budget.reserve(available) is True
        assert budget.available == 0

    def test_reserve_exceeds(self):
        budget = TokenBudget(total=128_000)
        budget.reserve(budget.available)
        assert budget.reserve(1) is False

    def test_reserve_negative(self):
        budget = TokenBudget(total=128_000)
        assert budget.reserve(-100) is False

    def test_release(self):
        budget = TokenBudget(total=128_000)
        budget.reserve(50_000)
        budget.release(20_000)
        assert budget.available == 128_000 - 4096 - 30_000

    def test_release_more_than_reserved(self):
        budget = TokenBudget(total=128_000)
        budget.reserve(10_000)
        budget.release(100_000)  # capped at 10_000
        assert budget.available == 128_000 - 4096

    def test_release_all(self):
        budget = TokenBudget(total=128_000)
        budget.reserve(50_000)
        budget.release_all()
        assert budget.available == 128_000 - 4096

    def test_utilization(self):
        budget = TokenBudget(total=100_000, reserved_for_response=0)
        budget.reserve(25_000)
        assert budget.utilization() == 0.25

    def test_utilization_zero_total(self):
        budget = TokenBudget(total=0)
        assert budget.utilization() == 0.0

    def test_summary(self):
        budget = TokenBudget(total=128_000)
        budget.reserve(50_000)
        s = budget.summary()
        assert s["total"] == 128_000
        assert s["reserved"] == 50_000
        assert s["reservation_count"] == 1

    def test_repr(self):
        budget = TokenBudget(total=128_000)
        r = repr(budget)
        assert "128000" in r
        assert "TokenBudget" in r


class TestUsageRecord:
    def test_total_tokens(self):
        record = UsageRecord(model="gpt-4o", input_tokens=100, output_tokens=50)
        assert record.total_tokens == 150

    def test_defaults(self):
        record = UsageRecord(model="gpt-4o", input_tokens=100, output_tokens=50)
        assert record.source == ""
        assert record.timestamp > 0


class TestUsageTracker:
    def test_record_and_count(self):
        tracker = UsageTracker()
        tracker.record("gpt-4o", 100, 50)
        tracker.record("gpt-4o", 200, 100)
        assert tracker.total_tokens() == (300, 150, 450)

    def test_empty(self):
        tracker = UsageTracker()
        assert tracker.total_tokens() == (0, 0, 0)
        assert tracker.total_cost() == 0.0

    def test_total_cost_known_model(self):
        tracker = UsageTracker()
        tracker.record("gpt-4o", 1_000_000, 1_000_000)
        # gpt-4o: $2.50/1M input + $10.00/1M output
        cost = tracker.total_cost()
        assert cost == pytest.approx(12.50)

    def test_total_cost_unknown_model(self):
        tracker = UsageTracker()
        tracker.record("unknown-model", 1000, 500)
        assert tracker.total_cost() == 0.0

    def test_by_model(self):
        tracker = UsageTracker()
        tracker.record("gpt-4o", 1000, 200, "agent")
        tracker.record("gpt-4o-mini", 500, 100, "eval")
        tracker.record("gpt-4o", 2000, 300, "agent")

        by_model = tracker.by_model()
        assert "gpt-4o" in by_model
        assert "gpt-4o-mini" in by_model
        assert by_model["gpt-4o"]["request_count"] == 2
        assert by_model["gpt-4o"]["input_tokens"] == 3000

    def test_custom_pricing(self):
        tracker = UsageTracker(custom_pricing={"my-model": (1.0, 2.0)})
        tracker.record("my-model", 1_000_000, 1_000_000)
        assert tracker.total_cost() == pytest.approx(3.0)

    def test_summary(self):
        tracker = UsageTracker()
        tracker.record("gpt-4o", 1000, 200, "test")

        s = tracker.summary()
        assert s["total_requests"] == 1
        assert s["total_tokens"] == 1200
        assert "gpt-4o" in s["by_model"]
