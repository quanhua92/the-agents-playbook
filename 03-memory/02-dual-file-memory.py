"""02-dual-file-memory.py — MEMORY.md + HISTORY.md dual-file memory system.

DualFileMemory stores long-term facts in MEMORY.md (structured, searchable)
and raw events in HISTORY.md (append-only log). This separates "what we know"
from "what happened".
"""

import asyncio
import tempfile
from pathlib import Path

from the_agents_playbook.memory import DualFileMemory, Fact


async def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = DualFileMemory(directory=tmpdir)

        # --- Store facts ---
        print("=== Storing Facts ===")
        await memory.store(Fact(content="User prefers Python", source="user"))
        await memory.store(Fact(content="Project uses FastAPI", source="project"))
        await memory.store(Fact(content="User prefers Python", source="duplicate"))  # dedup

        # --- Read MEMORY.md ---
        print("\n=== MEMORY.md Contents ===")
        facts = memory.read_facts()
        print(f"Total facts: {len(facts)}")
        for f in facts:
            print(f"  [{f.source}] {f.content}")

        # --- Append raw events to HISTORY.md ---
        print("\n=== Appending History Events ===")
        await memory.store_event("User asked about memory systems", source="user")
        await memory.store_event("Explained MEMORY.md vs HISTORY.md", source="assistant")

        print("HISTORY.md:")
        print(memory.read_history())

        # --- Recall by substring matching ---
        print("=== Recall (query='Python') ===")
        results = await memory.recall("Python")
        for f in results:
            print(f"  [{f.source}] {f.content}")

        print("\n=== Recall (query='FastAPI') ===")
        results = await memory.recall("FastAPI")
        for f in results:
            print(f"  [{f.source}] {f.content}")

        print("\n=== Recall (query='nonexistent') ===")
        results = await memory.recall("nonexistent")
        print(f"  Found {len(results)} results")

        # --- Show file paths ---
        print(f"\nMEMORY.md: {memory.memory_path}")
        print(f"HISTORY.md: {memory.history_path}")


asyncio.run(main())
