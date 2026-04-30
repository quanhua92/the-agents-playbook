"""07-segmented-decay.py — Segmented memory with tiered exponential decay.

Not all memories are equal. This example shows how to classify memories
into segments (identity, preference, context, etc.), each with its own
decay rate. Permanent memories like a user's name never fade. Temporary
context like "I'm at a coffee shop" decays within hours.

No SDK imports needed — we build everything from the MemoryRecord and
MemoryDecay classes to show the concept clearly.
"""

from the_agents_playbook.memory import (
    MemoryDecay,
    MemoryLifecycle,
    MemoryRecord,
    MemorySegment,
    MemoryTier,
)


def print_bar(label: str, value: float, width: int = 40) -> None:
    """Print a labeled progress bar."""
    filled = int(value * width)
    bar = "#" * filled + "-" * (width - filled)
    print(f"  {label:25s} [{bar}] {value:.4f}")


def main():
    decay = MemoryDecay()

    # --- Create memories with different segments ---
    print("=== Creating Segmented Memories ===\n")

    records = [
        # Permanent — identity should never decay
        MemoryRecord(
            content="User's name is Alice",
            source="user",
            segment=MemorySegment.IDENTITY,
        ),
        # Permanent — corrections supersede old facts
        MemoryRecord(
            content="User prefers they/them pronouns (corrected from she/her)",
            source="user",
            segment=MemorySegment.CORRECTION,
            supersedes="pronouns-001",
        ),
        # Long-term — preferences decay very slowly
        MemoryRecord(
            content="User prefers dark mode and concise responses",
            source="user",
            segment=MemorySegment.PREFERENCE,
        ),
        # Long-term — relationship context
        MemoryRecord(
            content="User is a senior engineer at Acme Corp",
            source="user",
            segment=MemorySegment.RELATIONSHIP,
        ),
        # Medium-term — project details
        MemoryRecord(
            content="Working on migrating auth service to OAuth 2.0",
            source="assistant",
            segment=MemorySegment.PROJECT,
        ),
        # Medium-term — knowledge from tool results
        MemoryRecord(
            content="The auth service uses JWT tokens with 15min expiry",
            source="tool:read_file",
            segment=MemorySegment.KNOWLEDGE,
        ),
        # Short-term — temporary context, decays fast
        MemoryRecord(
            content="User is currently at a coffee shop with spotty WiFi",
            source="user",
            segment=MemorySegment.CONTEXT,
        ),
        # Short-term — another temporary detail
        MemoryRecord(
            content="User mentioned they're in a hurry today",
            source="user",
            segment=MemorySegment.CONTEXT,
        ),
    ]

    # --- Show segment defaults ---
    print("Segment defaults:")
    from the_agents_playbook.memory import SEGMENT_DEFAULTS

    for segment, config in SEGMENT_DEFAULTS.items():
        print(f"  {segment.value:15s} tier={config.tier.value:12s} "
              f"importance={config.importance:.1f} decay_rate={config.decay_rate}")

    # --- Score each memory at different time points ---
    print("\n=== Decay Scores Over Time ===\n")

    time_points = [0, 1, 7, 30, 90, 365]
    for days in time_points:
        print(f"\n--- {days} days since creation ---")
        for record in records:
            score = decay.score(record, days)
            print_bar(
                f"[{record.segment.value}] {record.content[:35]}",
                score,
            )

    # --- Simulate access boosting ---
    print("\n\n=== Access Boost Effect ===")
    print("The knowledge fact was recalled 10 times.\n")

    knowledge_record = records[5]  # "The auth service uses JWT tokens..."
    for i in range(10):
        knowledge_record.record_access()

    for days in [0, 7, 30, 90]:
        score = decay.score(knowledge_record, days)
        print(f"  {days:3d} days: score = {score:.4f}")

    # --- Show pruning ---
    print("\n\n=== Pruning Stale Memories ===\n")

    # Manually backdate the context records to simulate old age
    import time
    now = time.monotonic()
    records[6].timestamp = now - 365 * 86400   # 1 year old
    records[7].timestamp = now - 365 * 86400   # 1 year old

    pruned = decay.prune(records)

    print(f"Records pruned: {len(pruned)}")
    for record in pruned:
        days_old = (now - record.timestamp) / 86400.0
        score = decay.score(record, days_old)
        print(f"  [{record.lifecycle.value}] {record.content[:50]} "
              f"(was {days_old:.0f} days old, score={score:.4f})")

    print("\nRemaining active records:")
    for record in records:
        if record.lifecycle == MemoryLifecycle.ACTIVE:
            print(f"  [{record.segment.value}] {record.content[:50]}")

    # --- Show correction superseding ---
    print("\n\n=== Correction Superseding ===\n")
    print("When a user corrects a fact, the old fact is superseded:")
    print(f"  New fact: {records[1].content}")
    print(f"  Supersedes: {records[1].supersedes}")

    # --- Lifecycle summary ---
    print("\n\n=== Full Lifecycle Summary ===\n")
    lifecycle_counts = {}
    for record in records:
        lc = record.lifecycle.value
        lifecycle_counts[lc] = lifecycle_counts.get(lc, 0) + 1

    for state, count in lifecycle_counts.items():
        print(f"  {state:12s}: {count} record(s)")


if __name__ == "__main__":
    main()
