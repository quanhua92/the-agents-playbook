"""MCPBridge — JSON-RPC bridge to external MCP servers over stdio."""

import asyncio
import json
import logging
from typing import Any

from .protocol import Tool, ToolResult

logger = logging.getLogger(__name__)


class MCPConnectionError(Exception):
    """Raised when the MCP server cannot be started or communicated with."""


class _MCPTool(Tool):
    """Wraps an MCP server tool as a Tool instance."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        send_request,
    ) -> None:
        self._name = name
        self._description = description
        self._parameters = parameters
        self._send_request = send_request

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            result = await self._send_request(
                "tools/call",
                {"name": self._name, "arguments": kwargs},
            )
            return ToolResult(output=json.dumps(result, indent=2))
        except Exception as e:
            return ToolResult(output=f"MCP tool call failed: {e}", error=True)


class MCPBridge:
    """Connect to an external MCP server over stdio and expose its tools.

    The bridge starts the server as a subprocess, initializes via the
    MCP handshake (initialize → initialized), and discovers available tools.

    Usage:
        bridge = MCPBridge("npx", ["-y", "@anthropic/mcp-server-filesystem", "/tmp"])
        await bridge.start()
        tools = bridge.get_tools()
        await bridge.stop()

    Tools returned by the bridge implement the Tool protocol and can be
    registered directly in a ToolRegistry.
    """

    def __init__(self, command: str, args: list[str]) -> None:
        self._command = command
        self._args = args
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._tools: list[_MCPTool] = []

    async def _send_request(self, method: str, params: dict | None = None) -> Any:
        """Send a JSON-RPC request and wait for the response."""
        if not self._process or self._process.stdin is None:
            raise MCPConnectionError("MCP server not started")

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
        }
        if params:
            request["params"] = params

        request_json = json.dumps(request)
        logger.debug("MCP send: %s", request_json)

        self._process.stdin.write(request_json.encode() + b"\n")
        await self._process.stdin.drain()

        # Read response lines until we get our response ID
        while True:
            if self._process.stdout is None:
                raise MCPConnectionError("MCP server stdout closed")
            line = await self._process.stdout.readline()
            if not line:
                raise MCPConnectionError("MCP server closed connection")
            line = line.decode().strip()
            if not line:
                continue

            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("MCP non-JSON line: %s", line)
                continue

            if response.get("id") == self._request_id:
                if "error" in response:
                    raise MCPConnectionError(f"MCP error: {response['error']}")
                logger.debug("MCP recv: %s", json.dumps(response, indent=2))
                return response.get("result", {})

    async def start(self) -> None:
        """Start the MCP server subprocess and perform the handshake."""
        logger.info("Starting MCP server: %s %s", self._command, " ".join(self._args))

        self._process = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Initialize handshake
        await self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "the-agents-playbook", "version": "0.1.0"},
            },
        )

        # Send initialized notification (no id = notification, no response expected)
        notification = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }
        )
        if self._process.stdin:
            self._process.stdin.write(notification.encode() + b"\n")
            await self._process.stdin.drain()

        # Discover tools
        result = await self._send_request("tools/list", {})
        self._tools = [
            _MCPTool(
                name=tool.get("name", "unknown"),
                description=tool.get("description", ""),
                parameters=tool.get(
                    "inputSchema", {"type": "object", "properties": {}}
                ),
                send_request=self._send_request,
            )
            for tool in result.get("tools", [])
        ]

        logger.info(
            "MCP bridge connected, discovered %d tools: %s",
            len(self._tools),
            [t.name for t in self._tools],
        )

    def get_tools(self) -> list[Tool]:
        """Return discovered tools as Tool instances, ready for ToolRegistry."""
        return list(self._tools)

    async def stop(self) -> None:
        """Stop the MCP server subprocess."""
        if self._process:
            logger.info("Stopping MCP server")
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            except ProcessLookupError:
                pass
            self._process = None
