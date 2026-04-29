"""04-built-in-tools.py — Shell, file I/O, and web search tools in action.

Demonstrates the three built-in tools that come with the SDK:
- ShellTool: subprocess execution with deny patterns
- FileReadTool / FileWriteTool: workspace-restricted file operations
- WebSearchTool: httpx-based web search
"""

import asyncio
import tempfile
from pathlib import Path

from the_agents_playbook.tools import ToolRegistry
from the_agents_playbook.tools.builtins import ShellTool, FileReadTool, FileWriteTool, WebSearchTool


async def main():
    registry = ToolRegistry()

    # Set up built-in tools
    workspace = Path(tempfile.mkdtemp())
    print(f"Workspace: {workspace}\n")

    registry.register(ShellTool(workspace=workspace, timeout_seconds=5.0))
    registry.register(FileReadTool(workspace=workspace))
    registry.register(FileWriteTool(workspace=workspace))
    registry.register(WebSearchTool())

    print(f"Registered tools: {registry.list_tools()}\n")

    # --- Shell execution ---
    print("=== Shell ===")
    result = await registry.dispatch("shell", {"command": "echo 'Hello from shell'"})
    print(f"echo: {result.output}")

    result = await registry.dispatch("shell", {"command": "ls /tmp | head -5"})
    print(f"ls:   {result.output}")

    # Deny pattern blocks dangerous commands
    result = await registry.dispatch("shell", {"command": "rm -rf /"})
    print(f"rm:   {result.output} (error={result.error})")
    print()

    # --- File I/O ---
    print("=== File Write ===")
    result = await registry.dispatch(
        "file_write",
        {"path": "test.txt", "content": "Hello from file write!"},
    )
    print(f"Write: {result.output}")

    print("\n=== File Read ===")
    result = await registry.dispatch("file_read", {"path": "test.txt"})
    print(f"Read:  {result.output}")

    # Path traversal is blocked
    result = await registry.dispatch("file_read", {"path": "../../etc/passwd"})
    print(f"Traversal: {result.output} (error={result.error})")

    # Nested directory creation
    result = await registry.dispatch(
        "file_write",
        {"path": "nested/dir/file.txt", "content": "deep file"},
    )
    print(f"Nested: {result.output}")
    print()

    # --- Web search ---
    print("=== Web Search ===")
    result = await registry.dispatch(
        "web_search",
        {"query": "Python programming language"},
    )
    print(f"Search result:\n{result.output}")


asyncio.run(main())
