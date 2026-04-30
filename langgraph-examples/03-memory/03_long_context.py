"""03_long_context.py -- SessionCompactor for long conversation management.

When a conversation exceeds a token threshold, old messages are summarized
into a single summary message. Recent messages are kept intact.

In a real LangGraph workflow, you would call compactor.compact(messages)
on the message history extracted from a checkpoint before re-invoking the graph.

This uses the SessionCompactor from shared/compactor.py, which is adapted
from the root project's the_agents_playbook/memory/session.py.
"""

from shared import SessionCompactor, get_openai_llm


def _make_summarizer(llm):
    """Return a summarize_fn that uses a real LLM to compress old messages."""
    def summarize(messages):
        conversation = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        prompt = (
            "Summarize the following conversation in 2-3 short sentences. "
            "Focus on topics discussed and key takeaways.\n\n"
            f"{conversation}"
        )
        response = llm.invoke(prompt)
        return response.content
    return summarize


def main():
    messages = [
        {"role": "user", "content": "Tell me about Python."},
        {"role": "assistant", "content": "Python is a versatile programming language used for web development, data science, AI, and more. It emphasizes readability with significant whitespace and supports multiple paradigms including object-oriented, functional, and procedural programming."},
        {"role": "user", "content": "What about lists?"},
        {"role": "assistant", "content": "Lists are ordered collections in Python. They can hold any type, support indexing, slicing, and list comprehension. You can nest lists to create multi-dimensional structures and use methods like .append(), .extend(), .pop(), and .sort() for manipulation."},
        {"role": "user", "content": "And dictionaries?"},
        {"role": "assistant", "content": "Dictionaries are key-value pairs. They use {} syntax and support .get(), .keys(), .values(), and .items(). As of Python 3.7, dictionaries maintain insertion order. They are implemented as hash tables, giving O(1) average lookup time."},
        {"role": "user", "content": "How do error handling and exceptions work?"},
        {"role": "assistant", "content": "Python uses try/except/finally blocks for error handling. You can catch specific exceptions like ValueError, TypeError, KeyError, or use a bare except to catch everything. The finally block always runs regardless of whether an exception occurred. You can also raise custom exceptions by subclassing the Exception class."},
        {"role": "user", "content": "What about async programming?"},
        {"role": "assistant", "content": "Async programming in Python uses asyncio. You define coroutines with async def and run them with asyncio.run(). Use await to yield control to the event loop. Common use cases include network I/O, web scraping, and concurrent API calls. Libraries like aiohttp and asyncpg provide async versions of popular tools."},
        {"role": "user", "content": "Can you explain decorators?"},
        {"role": "assistant", "content": "Decorators are functions that modify other functions. They use the @decorator syntax, which is syntactic sugar for wrapping a function. Common built-in decorators include @staticmethod, @classmethod, and @property. You can build custom decorators using closures or the functools.wraps decorator to preserve metadata."},
        {"role": "user", "content": "How does Python handle package management?"},
        {"role": "assistant", "content": "Python uses pip as its default package manager alongside requirements.txt for dependency specification. Modern projects increasingly use Poetry or uv for dependency resolution and virtual environment management. Python 3.11+ introduced tomllib for reading pyproject.toml, which has become the standard for project configuration."},
        {"role": "user", "content": "What's the most recent topic we discussed?"},
    ]

    tokens_before = SessionCompactor.estimate_tokens(messages)

    # ── Non-LLM based compaction (simple concatenation) ──────────────────
    print("=== Non-LLM Based Compaction ===\n")

    compactor = SessionCompactor(max_tokens=100, keep_recent=4)

    print(f"Before compaction: {len(messages)} messages, ~{tokens_before} tokens")

    compacted = compactor.compact(messages)
    tokens_after = SessionCompactor.estimate_tokens(compacted)

    print(f"After compaction:  {len(compacted)} messages, ~{tokens_after} tokens")
    print(f"Saved: ~{tokens_before - tokens_after} tokens\n")

    print("--- Compacted Messages ---")
    for i, msg in enumerate(compacted):
        role = msg["role"]
        content = msg["content"]
        is_summary = "[Conversation summary]" in content
        label = " [SUMMARY]" if is_summary else ""
        print(f"  {i + 1}. [{role}]{label} {content}")

    # Show that under-threshold messages are unchanged
    print("\n--- Under Threshold (no compaction) ---")
    short = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    result = compactor.compact(short)
    print(f"  Input:  {len(short)} messages")
    print(f"  Output: {len(result)} messages (unchanged: {len(short) == len(result)})")

    # ── LLM-based compaction ─────────────────────────────────────────────
    print("\n\n=== LLM-Based Compaction ===\n")

    llm_compactor = SessionCompactor(max_tokens=100, keep_recent=4, summarize_fn=_make_summarizer(get_openai_llm()))

    print(f"Before compaction: {len(messages)} messages, ~{tokens_before} tokens")

    llm_compacted = llm_compactor.compact(messages)
    tokens_after_llm = SessionCompactor.estimate_tokens(llm_compacted)

    print(f"LLM summary response: {llm_compacted[0]['content']}\n")

    print(f"After compaction:  {len(llm_compacted)} messages, ~{tokens_after_llm} tokens")
    print(f"Saved: ~{tokens_before - tokens_after_llm} tokens\n")

    print("--- Compacted Messages ---")
    for i, msg in enumerate(llm_compacted):
        role = msg["role"]
        content = msg["content"]
        is_summary = "[Conversation summary]" in content
        label = " [SUMMARY]" if is_summary else ""
        print(f"  {i + 1}. [{role}]{label} {content}")


if __name__ == "__main__":
    main()
