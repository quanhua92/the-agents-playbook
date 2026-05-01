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
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from shared import get_openai_llm


def recall_memory(query: str, segment: str = "all", top_k: int = 3) -> str:
    """Recall memories, optionally filtered by segment.

    This is a simplified in-memory version for demonstration.
    A real implementation would use vector search with segment metadata.

    Args:
        query: The search query.
        segment: Memory segment to filter (identity, expertise,
            preference, relationship, goal, feedback, project,
            knowledge, context, or all).
        top_k: Maximum number of results.
    """
    from the_agents_playbook.memory import (
        MemoryRecord,
        MemorySegment,
        MemoryDecay,
    )

    # Simulated memory store
    memories = [
        MemoryRecord(content="User's name is Alice", source="user",
                     segment=MemorySegment.IDENTITY),
        MemoryRecord(content="Alice is proficient in Rust and Go", source="user",
                     segment=MemorySegment.EXPERTISE),
        MemoryRecord(content="Alice prefers dark mode", source="user",
                     segment=MemorySegment.PREFERENCE),
        MemoryRecord(content="Alice is a senior engineer at Acme Corp",
                     source="user", segment=MemorySegment.RELATIONSHIP),
        MemoryRecord(content="Alice wants to transition to staff engineer",
                     source="user", segment=MemorySegment.GOAL),
        MemoryRecord(content="User said responses were too verbose",
                     source="user", segment=MemorySegment.FEEDBACK),
        MemoryRecord(content="Working on OAuth migration", source="user",
                     segment=MemorySegment.PROJECT),
        MemoryRecord(content="Auth service uses JWT with 15min expiry",
                     source="tool", segment=MemorySegment.KNOWLEDGE),
        MemoryRecord(content="Alice is at a coffee shop", source="user",
                     segment=MemorySegment.CONTEXT),
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

    results = scored[:top_k]
    if not results or all(score == 0 for _, score in results):
        return f"No matching memories found (segment={segment}, query={query})"

    lines = []
    for m, score in results:
        lines.append(f"[{m.segment.value}] {m.content}")
    return "\n".join(lines)


def main():
    print("=== Segmented Memory in LangGraph Agent ===\n")

    llm = get_openai_llm()

    # Build a ReAct agent with a recall_memory tool
    agent = create_react_agent(
        model=llm,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "recall_memory",
                    "description": (
                        "Recall stored memories about the user. "
                        "Filter by segment to get specific categories: "
                        "identity (name, email), expertise (skills), "
                        "preference (likes/dislikes), relationship (org, team), "
                        "goal (intentions), feedback (agent perf signals), "
                        "project (work details), knowledge (facts from tools), "
                        "context (temporary situation), or 'all'."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What to search for",
                            },
                            "segment": {
                                "type": "string",
                                "description": "Memory segment filter",
                                "enum": [
                                    "identity", "expertise", "preference",
                                    "relationship", "goal", "feedback",
                                    "project", "knowledge", "context",
                                    "all",
                                ],
                                "default": "all",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Max results (default 3)",
                                "default": 3,
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ],
        prompt=(
            "You have access to a segmented memory system. "
            "Use recall_memory to look up user information. "
            "You can filter by segment type for more relevant results."
        ),
    )

    # Scenario 1: Recall identity
    print("--- Querying identity memories ---")
    result = agent.invoke({
        "messages": [HumanMessage(content="What's my name?")],
    })
    print(result["messages"][-1].content)
    print()

    # Scenario 2: Recall project context
    print("--- Querying project memories ---")
    result = agent.invoke({
        "messages": [HumanMessage(
            content="What am I working on? Search only project memories."
        )],
    })
    print(result["messages"][-1].content)
    print()

    # Scenario 3: Show that context memories decay
    print("--- Demonstrating decay ---")
    print("(Context memories have high decay rate, project memories moderate)")
    from the_agents_playbook.memory import MemoryDecay, MemorySegment

    decay = MemoryDecay()
    # Use the same records from recall_memory for demonstration
    segments = [
        ("Identity (permanent)", MemorySegment.IDENTITY),
        ("Expertise (long-term)", MemorySegment.EXPERTISE),
        ("Preference (long-term)", MemorySegment.PREFERENCE),
        ("Goal (long-term)", MemorySegment.GOAL),
        ("Feedback (medium-term)", MemorySegment.FEEDBACK),
        ("Project (medium-term)", MemorySegment.PROJECT),
        ("Knowledge (medium-term)", MemorySegment.KNOWLEDGE),
        ("Context (short-term)", MemorySegment.CONTEXT),
    ]

    for label, seg in segments:
        from the_agents_playbook.memory import MemoryRecord
        record = MemoryRecord(content="test", source="test", segment=seg)
        score_30d = decay.score(record, 30)
        score_90d = decay.score(record, 90)
        print(f"  {label:30s} 30d={score_30d:.3f}  90d={score_90d:.3f}")


if __name__ == "__main__":
    main()
