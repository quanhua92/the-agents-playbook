"""06-mcp-bridge.py — Connect to an external MCP server and discover its tools.

The MCPBridge starts an MCP server as a subprocess, performs the JSON-RPC
handshake (initialize → initialized), discovers available tools, and wraps
them as Tool instances that can be registered in a ToolRegistry.

This example uses a filesystem MCP server if available. If not, it shows
the bridge setup without executing.
"""

import asyncio
import sys

from the_agents_playbook.tools import ToolRegistry
from the_agents_playbook.tools.mcp import MCPBridge


async def main():
    # Configure the MCP server to connect to.
    # This example uses the Anthropic filesystem MCP server.
    # Install it with: npx @anthropic/mcp-server-filesystem /tmp
    #
    # You can substitute any MCP server here — the bridge protocol is the same.
    bridge = MCPBridge(
        command="npx",
        args=["-y", "@anthropic/mcp-server-filesystem", "/tmp"],
    )

    print("Starting MCP bridge...")
    try:
        await bridge.start()
    except Exception as e:
        print(f"Failed to start MCP server: {e}")
        print()
        print("This example requires the @anthropic/mcp-server-filesystem package.")
        print("Install it with: npm install -g @anthropic/mcp-server-filesystem")
        print("Or run: npx @anthropic/mcp-server-filesystem /tmp")
        print()
        print("If you have another MCP server, modify the MCPBridge() call above.")
        sys.exit(1)

    # Get discovered tools
    tools = bridge.get_tools()
    print(f"Discovered {len(tools)} tools:")

    registry = ToolRegistry()
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")
        print(f"    Parameters: {tool.parameters}")
        registry.register(tool)

    print()

    # Execute a tool through the registry (same interface as built-in tools)
    if tools:
        tool_name = tools[0].name
        print(f"Executing tool: {tool_name}")

        result = await registry.dispatch(tool_name, {})
        print(f"Result: {result.output}")
        print(f"Error:  {result.error}")

    # Clean up
    await bridge.stop()
    print("\nMCP bridge stopped.")


asyncio.run(main())
