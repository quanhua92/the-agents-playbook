"""03_long_context.py -- SessionCompactor for long conversation management.

When a conversation exceeds a token threshold, old messages are summarized
into a single summary message. Recent messages are kept intact.

In a real LangGraph workflow, you would call compactor.compact(messages)
on the message history extracted from a checkpoint before re-invoking the graph.

This uses the SessionCompactor from shared/compactor.py, which is adapted
from the root project's the_agents_playbook/memory/session.py.
"""

from shared import SessionCompactor


def main():
    compactor = SessionCompactor(max_tokens=100, keep_recent=4)

    # Simulate a long conversation
    messages = [
        {"role": "user", "content": "Tell me about Python."},
        {"role": "assistant", "content": "Python is a versatile programming language used for web development, data science, AI, and more."},
        {"role": "user", "content": "What about lists?"},
        {"role": "assistant", "content": "Lists are ordered collections in Python. They can hold any type and support indexing, slicing, and comprehension."},
        {"role": "user", "content": "And dictionaries?"},
        {"role": "assistant", "content": "Dictionaries are key-value pairs. They use {} syntax and support .get(), .keys(), .values(), and .items()."},
        {"role": "user", "content": "What about async?"},
        {"role": "assistant", "content": "Async programming in Python uses asyncio. You define coroutines with async def and run them with asyncio.run()."},
        {"role": "user", "content": "What's the most recent topic we discussed?"},
    ]

    tokens_before = SessionCompactor.estimate_tokens(messages)
    print(f"Before compaction: {len(messages)} messages, ~{tokens_before} tokens")

    compacted = compactor.compact(messages)
    tokens_after = SessionCompactor.estimate_tokens(compacted)

    print(f"After compaction:  {len(compacted)} messages, ~{tokens_after} tokens")
    print(f"Saved: ~{tokens_before - tokens_after} tokens\n")

    print("=== Compacted Messages ===")
    for i, msg in enumerate(compacted):
        role = msg["role"]
        content = msg["content"]
        is_summary = "[Conversation summary]" in content
        label = " [SUMMARY]" if is_summary else ""
        print(f"  {i + 1}. [{role}]{label} {content[:120]}{'...' if len(content) > 120 else ''}")

    # Show that under-threshold messages are unchanged
    print("\n=== Under Threshold (no compaction) ===")
    short = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    result = compactor.compact(short)
    print(f"  Input:  {len(short)} messages")
    print(f"  Output: {len(result)} messages (unchanged: {len(short) == len(result)})")


if __name__ == "__main__":
    main()
