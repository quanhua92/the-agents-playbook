"""01-fact-storage.py — Create, store, and retrieve Fact objects.

Demonstrates the Fact dataclass — the atomic unit of memory.
Facts have content, source, timestamp, optional embedding, and tags.
"""

import asyncio
from the_agents_playbook.memory import Fact


async def main():
    # Create facts with minimal info
    fact1 = Fact(
        content="User prefers Python over JavaScript", source="user preference"
    )
    fact2 = Fact(content="Project uses OpenAI-compatible API", source="project config")
    fact3 = Fact(
        content="User works in the /workspace directory",
        source="user context",
        tags=["environment", "filesystem"],
    )

    print("=== Fact Objects ===")
    for f in [fact1, fact2, fact3]:
        print(f"  Content:  {f.content}")
        print(f"  Source:   {f.source}")
        print(f"  Tags:     {f.tags}")
        print(f"  Time:     {f.timestamp:.2f}")
        print(f"  Embedding: {f.embedding is not None}")
        print()

    # Facts are dataclasses — easy to serialize
    import dataclasses

    as_dict = dataclasses.asdict(fact1)
    # Exclude embedding (numpy array doesn't serialize to JSON directly)
    as_dict.pop("embedding")
    import json

    print("=== Serialized ===")
    print(json.dumps(as_dict, indent=2))


asyncio.run(main())
