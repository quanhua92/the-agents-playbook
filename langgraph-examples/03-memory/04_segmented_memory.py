"""04_segmented_memory.py -- Segmented recall in a LangGraph agent.

This example demonstrates how segmented memory can be integrated with
a LangGraph ReAct agent. A custom tool (`recall_memory`) uses segment
filters to recall only relevant categories of memories.

In production, this pattern lets you:
- Recall identity facts before any turn
- Recall project context only when discussing work
- Never recall stale temporary context
"""

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

from shared import get_openai_llm


@tool
def recall_memory(query: str, segment: str = "all", top_k: int = 3) -> str:
    """Recall stored memories about the user.

    Filter by segment to get specific categories:
    identity (name, email), expertise (skills),
    preference (likes/dislikes), relationship (org, team),
    goal (intentions), feedback (agent perf signals),
    project (work details), knowledge (facts from tools),
    context (temporary situation), or 'all'.
    """
    from the_agents_playbook.memory import (
        MemoryRecord,
        MemorySegment,
    )

    # Simulated memory store
    memories = [
        MemoryRecord(
            content="User's name is Alice",
            source="user",
            segment=MemorySegment.IDENTITY,
        ),
        MemoryRecord(
            content="Alice is proficient in Rust and Go",
            source="user",
            segment=MemorySegment.EXPERTISE,
        ),
        MemoryRecord(
            content="Alice prefers dark mode",
            source="user",
            segment=MemorySegment.PREFERENCE,
        ),
        MemoryRecord(
            content="Alice is a senior engineer at Acme Corp",
            source="user",
            segment=MemorySegment.RELATIONSHIP,
        ),
        MemoryRecord(
            content="Alice wants to transition to staff engineer",
            source="user",
            segment=MemorySegment.GOAL,
        ),
        MemoryRecord(
            content="User said responses were too verbose",
            source="user",
            segment=MemorySegment.FEEDBACK,
            tags=["verbosity"],
        ),
        MemoryRecord(
            content="Working on OAuth migration",
            source="user",
            segment=MemorySegment.PROJECT,
            tags=["auth-migration"],
        ),
        MemoryRecord(
            content="Auth service uses JWT with 15min expiry",
            source="tool",
            segment=MemorySegment.KNOWLEDGE,
            tags=["auth-migration"],
        ),
        MemoryRecord(
            content="Alice is at a coffee shop",
            source="user",
            segment=MemorySegment.CONTEXT,
        ),
        MemoryRecord(
            content="Alice mentioned she's in a hurry today",
            source="user",
            segment=MemorySegment.CONTEXT,
        ),
    ]

    # Filter by segment if specified
    if segment != "all":
        try:
            seg = MemorySegment(segment)
            memories = [m for m in memories if m.segment == seg]
        except ValueError:
            pass  # invalid segment, return all

    # Simple keyword matching (no embeddings needed for demo)
    scored = []
    for m in memories:
        # Count matching words
        query_words = set(query.lower().split())
        content_words = set(m.content.lower().split())
        overlap = len(query_words & content_words)
        scored.append((m, overlap))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Return top matches; if nothing scored above zero, return the first
    # record from the filtered set so the agent always has context.
    results = scored[:top_k]
    if not results or all(score == 0 for _, score in results):
        if memories:
            return f"[{memories[0].segment.value}] {memories[0].content}"
        return f"No memories found (segment={segment})"

    lines = []
    for m, score in results:
        lines.append(f"[{m.segment.value}] {m.content}")
    return "\n".join(lines)


def main():
    print("=== Segmented Memory in LangGraph Agent ===\n")

    llm = get_openai_llm()

    # Build a ReAct agent with a recall_memory tool
    agent = create_agent(
        llm,
        tools=[recall_memory],
        system_prompt=(
            "You have access to a segmented memory system. "
            "Use recall_memory to look up user information. "
            "You can filter by segment type for more relevant results."
        ),
    )

    # Scenario 1: Recall identity
    print("--- Querying identity memories ---")
    result = agent.invoke(
        {
            "messages": [HumanMessage(content="What's my name?")],
        }
    )
    for msg in result["messages"]:
        role = getattr(msg, "type", "?")
        text = getattr(msg, "content", "")
        if role == "tool":
            print(f"  [tool call result] {text[:120]}")
        else:
            print(f"  [{role}] {text[:120]}")
    print()

    # Scenario 2: Recall project context
    print("--- Querying project memories ---")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What am I working on? Search only project memories."
                )
            ],
        }
    )
    for msg in result["messages"]:
        role = getattr(msg, "type", "?")
        text = getattr(msg, "content", "")
        if role == "tool":
            print(f"  [tool call result] {text[:120]}")
        else:
            print(f"  [{role}] {text[:120]}")
    print()

    # Scenario 3: Recall expertise
    print("--- Querying expertise ---")
    result = agent.invoke(
        {
            "messages": [HumanMessage(content="What programming languages do I know?")],
        }
    )
    for msg in result["messages"]:
        role = getattr(msg, "type", "?")
        text = getattr(msg, "content", "")
        if role == "tool":
            print(f"  [tool call result] {text[:120]}")
        else:
            print(f"  [{role}] {text[:120]}")
    print()

    # Scenario 4: Recall goals and preferences together
    print("--- Querying goals and preferences ---")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What are my career goals and how should you format "
                    "responses for me? Search all memories."
                )
            ],
        }
    )
    for msg in result["messages"]:
        role = getattr(msg, "type", "?")
        text = getattr(msg, "content", "")
        if role == "tool":
            print(f"  [tool call result] {text[:200]}")
        else:
            print(f"  [{role}] {text[:200]}")
    print()

    # Scenario 5: Recall feedback to calibrate behavior
    print("--- Querying feedback ---")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Have I given you any feedback about your responses? "
                    "Check feedback memories."
                )
            ],
        }
    )
    for msg in result["messages"]:
        role = getattr(msg, "type", "?")
        text = getattr(msg, "content", "")
        if role == "tool":
            print(f"  [tool call result] {text[:120]}")
        else:
            print(f"  [{role}] {text[:120]}")
    print()

    # Scenario 6: Segment defaults table
    print("--- Segment defaults ---")
    from the_agents_playbook.memory import (
        MemoryDecay,
        MemorySegment,
        MemoryRecord,
        MemoryLifecycle,
        SEGMENT_DEFAULTS,
    )

    for segment, config in SEGMENT_DEFAULTS.items():
        print(
            f"  {segment.value:15s} tier={config.tier.value:12s} "
            f"importance={config.importance:.2f} decay_rate={config.decay_rate}"
        )

    # Scenario 7: Show that context memories decay over time
    print("\n--- Demonstrating decay ---")
    print("(Each segment fades at its own rate)\n")

    decay = MemoryDecay()

    def print_bar(label: str, value: float, width: int = 30) -> None:
        filled = int(value * width)
        bar = "#" * filled + "-" * (width - filled)
        print(f"  {label:30s} [{bar}] {value:.4f}")

    records = [
        MemoryRecord(
            content="User's name is Alice",
            source="user",
            segment=MemorySegment.IDENTITY,
        ),
        MemoryRecord(
            content="Alice is proficient in Rust and Go",
            source="user",
            segment=MemorySegment.EXPERTISE,
        ),
        MemoryRecord(
            content="Alice prefers dark mode",
            source="user",
            segment=MemorySegment.PREFERENCE,
        ),
        MemoryRecord(
            content="Alice is a senior engineer at Acme Corp",
            source="user",
            segment=MemorySegment.RELATIONSHIP,
        ),
        MemoryRecord(
            content="Alice wants to transition to staff engineer",
            source="user",
            segment=MemorySegment.GOAL,
        ),
        MemoryRecord(
            content="User said responses were too verbose",
            source="user",
            segment=MemorySegment.FEEDBACK,
        ),
        MemoryRecord(
            content="Working on OAuth migration",
            source="user",
            segment=MemorySegment.PROJECT,
        ),
        MemoryRecord(
            content="Auth service uses JWT with 15min expiry",
            source="tool",
            segment=MemorySegment.KNOWLEDGE,
        ),
        MemoryRecord(
            content="Alice is at a coffee shop",
            source="user",
            segment=MemorySegment.CONTEXT,
        ),
    ]

    time_points = [0, 1, 7, 30, 90, 365]
    for days in time_points:
        print(f"\n  --- {days} days since creation ---")
        for record in records:
            score = decay.score(record, days)
            print_bar(
                f"[{record.segment.value}] {record.content[:30]}",
                score,
            )

    # Scenario 8: Access boost — frequently recalled memories score higher
    print("\n--- Access boost ---")
    print("The knowledge fact was recalled 10 times.\n")
    knowledge_record = MemoryRecord(
        content="Auth service uses JWT with 15min expiry",
        source="tool",
        segment=MemorySegment.KNOWLEDGE,
    )
    for _ in range(10):
        knowledge_record.record_access()
    for days in [0, 7, 30, 90]:
        score = decay.score(knowledge_record, days)
        print(f"  {days:3d} days: score = {score:.4f}")

    # Scenario 9: Pruning — stale context gets removed
    print("\n--- Pruning stale memories ---\n")
    import time as _time

    now = _time.monotonic()
    stale = [
        MemoryRecord(
            content="At coffee shop with spotty WiFi",
            source="user",
            segment=MemorySegment.CONTEXT,
        ),
        MemoryRecord(
            content="In a hurry today", source="user", segment=MemorySegment.CONTEXT
        ),
        MemoryRecord(
            content="Working on OAuth migration",
            source="user",
            segment=MemorySegment.PROJECT,
        ),
    ]
    stale[0].timestamp = now - 365 * 86400
    stale[1].timestamp = now - 365 * 86400
    pruned = decay.prune(stale)
    print(f"Records pruned: {len(pruned)}")
    for r in pruned:
        print(f"  [{r.lifecycle.value}] {r.content}")
    print("\nRemaining active records:")
    for r in stale:
        if r.lifecycle == MemoryLifecycle.ACTIVE:
            print(f"  [{r.segment.value}] {r.content}")

    # Lifecycle summary
    lifecycle_counts = {}
    for r in stale:
        lc = r.lifecycle.value
        lifecycle_counts[lc] = lifecycle_counts.get(lc, 0) + 1
    print("\nLifecycle summary:")
    for state, count in lifecycle_counts.items():
        print(f"  {state:12s}: {count} record(s)")

    # Scenario 10: Tag-based scoping
    print("\n--- Tag-based scoping ---")
    print("Tags group records across segments for scope clearing:\n")
    all_records = [
        MemoryRecord(
            content="Working on OAuth migration",
            source="user",
            segment=MemorySegment.PROJECT,
            tags=["auth-migration"],
        ),
        MemoryRecord(
            content="Auth service uses JWT with 15min expiry",
            source="tool",
            segment=MemorySegment.KNOWLEDGE,
            tags=["auth-migration"],
        ),
        MemoryRecord(
            content="User said responses were too verbose",
            source="user",
            segment=MemorySegment.FEEDBACK,
            tags=["verbosity"],
        ),
    ]
    tagged = [r for r in all_records if "auth-migration" in r.tags]
    for r in tagged:
        print(f"  [{r.segment.value}] {r.content}  tags={r.tags}")


if __name__ == "__main__":
    main()
