"""04-consolidation.py — Extract structured facts from raw conversation history.

The LLMConsolidator reads HISTORY.md, sends it to the LLM, and extracts
durable facts (preferences, decisions, project context) into MEMORY.md.

This is consolidation vs compaction:
- Compaction: summarize → loses intent, misses edge cases
- Consolidation: extract and index → preserves facts in searchable form
"""

import asyncio
import tempfile

from the_agents_playbook.memory import DualFileMemory, LLMConsolidator


async def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = DualFileMemory(directory=tmpdir)

        # Simulate conversation history by appending raw events
        print("=== Writing Conversation History ===")
        events = [
            ("user", "Hey, I'm working on an agent project using Python"),
            ("assistant", "That sounds great! What framework are you using?"),
            (
                "user",
                "No framework — I'm building it from scratch with httpx and pydantic",
            ),
            ("assistant", "Interesting choice. Any specific LLM provider?"),
            ("user", "Using OpenAI-compatible API via OpenRouter"),
            ("user", "Oh and I prefer functional programming patterns over OOP"),
            ("assistant", "Got it, I'll keep that in mind for future suggestions"),
            ("user", "The project is called The Agents Playbook"),
        ]

        for role, text in events:
            await memory.store_event(text, source=role)

        print(f"HISTORY.md has {len(events)} entries:")
        print(memory.read_history()[:300] + "...")

        print("\n=== MEMORY.md Before Consolidation ===")
        print(f"Facts: {len(memory.read_facts())}")

        # Run consolidation (requires OPENAI_API_KEY in .env)
        print("\n=== Running Consolidation ===")
        consolidator = LLMConsolidator(memory=memory)
        new_facts = await consolidator.consolidate()

        print(f"\nExtracted {len(new_facts)} new facts:")
        for f in new_facts:
            print(f"  [{f.source}] {f.content}")

        print("\n=== MEMORY.md After Consolidation ===")
        all_facts = memory.read_facts()
        print(f"Total facts: {len(all_facts)}")
        for f in all_facts:
            print(f"  [{f.source}] {f.content}")

        # Re-running consolidation should not duplicate
        print("\n=== Re-running Consolidation (dedup check) ===")
        new_facts2 = await consolidator.consolidate()
        print(f"New facts on second run: {len(new_facts2)}")
        print(f"Total facts in MEMORY.md: {len(memory.read_facts())}")


asyncio.run(main())
