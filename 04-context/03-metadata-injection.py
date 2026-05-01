"""03-metadata-injection.py — Date, cwd, git status injection.

Demonstrates structured metadata injectors that produce
DYNAMIC priority ContextLayer instances.
"""

import asyncio
from pathlib import Path

from the_agents_playbook.context import inject_cwd, inject_date, inject_git_status


async def main():
    # --- Date injection (sync) ---

    date_layer = inject_date()
    print("=== Date Layer ===")
    print(f"Name: {date_layer.name}")
    print(f"Priority: {date_layer.priority.name}")
    print(f"Content:\n{date_layer.content}")
    print()

    # --- CWD injection (sync) ---

    cwd_layer = inject_cwd()
    print("=== CWD Layer ===")
    print(f"Name: {cwd_layer.name}")
    print(f"Priority: {cwd_layer.priority.name}")
    print(f"Content: {cwd_layer.content}")
    print()

    # Custom cwd
    custom_cwd = inject_cwd(cwd="/tmp")
    print(f"Custom CWD: {custom_cwd.content}")
    print()

    # --- Git status injection (async, requires git) ---

    git_layer = await inject_git_status()
    print("=== Git Status Layer ===")
    print(f"Name: {git_layer.name}")
    print(f"Priority: {git_layer.priority.name}")
    print(f"Content:\n{git_layer.content}")
    print()

    # Custom repo root
    repo_layer = await inject_git_status(repo_root=Path(__file__).parent.parent)
    print("=== Git Status (repo root) ===")
    print(repo_layer.content)


if __name__ == "__main__":
    asyncio.run(main())
