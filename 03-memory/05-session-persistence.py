"""05-session-persistence.py — Save and restore conversation state across restarts.

SessionPersistence writes conversation messages as JSONL (one JSON object
per line). Each line has role, content, and timestamp. This enables pausing
a conversation and resuming it later.
"""

import asyncio
import tempfile
from pathlib import Path

from the_agents_playbook.memory import SessionPersistence


async def main():
    session = SessionPersistence()

    with tempfile.TemporaryDirectory() as tmpdir:
        session_path = Path(tmpdir) / "conversation.jsonl"

        # --- Save a conversation ---
        print("=== Saving Conversation ===")
        messages = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "And what about Germany?"},
            {"role": "assistant", "content": "The capital of Germany is Berlin."},
        ]

        await session.save(messages, session_path)
        print(f"Saved {len(messages)} messages to {session_path.name}")

        # Show the JSONL content
        print("\nRaw JSONL:")
        print(session_path.read_text())

        # --- Load the conversation ---
        print("=== Loading Conversation ===")
        loaded = await session.load(session_path)
        print(f"Loaded {len(loaded)} messages:")
        for msg in loaded:
            print(f"  [{msg['role']}] {msg['content']}")

        # --- Append a message ---
        print("\n=== Appending a Message ===")
        await session.append(
            {"role": "user", "content": "What about Japan?"},
            session_path,
        )

        loaded_after = await session.load(session_path)
        print(f"Messages after append: {len(loaded_after)}")
        print(f"Last message: {loaded_after[-1]['content']}")

        # --- Malformed line recovery ---
        print("\n=== Malformed Line Recovery ===")
        # Manually corrupt a line
        with open(session_path, "a") as f:
            f.write("THIS IS NOT JSON\n")
            f.write(
                '{"role": "assistant", "content": "The capital of Japan is Tokyo."}\n'
            )

        recovered = await session.load(session_path)
        print(f"Loaded {len(recovered)} valid messages (1 malformed skipped)")
        print(f"Last valid: {recovered[-1]['content']}")

        # --- List sessions ---
        print("\n=== List Sessions ===")
        session2_path = Path(tmpdir) / "other-session.jsonl"
        await session.save([{"role": "user", "content": "test"}], session2_path)

        sessions = session.list_sessions(tmpdir)
        print(f"Found {len(sessions)} session(s):")
        for s in sessions:
            print(f"  {s.name}")

        # --- Load non-existent file ---
        print("\n=== Missing File ===")
        empty = await session.load(Path(tmpdir) / "nonexistent.jsonl")
        print(f"Loaded from missing file: {empty}")


asyncio.run(main())
