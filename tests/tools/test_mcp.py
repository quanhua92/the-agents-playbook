"""Tests for tools/mcp.py — MCPBridge."""

import json
from typing import Any

import pytest

from the_agents_playbook.tools.mcp import MCPBridge, MCPConnectionError


def _make_mock_process(
    responses: list[str] | None = None,
):
    """Create a mock subprocess that returns predefined responses."""
    response_iter = iter(responses or [])

    class MockStdin:
        def __init__(self):
            self.written: list[str] = []

        def write(self, data: bytes):
            self.written.append(data.decode())

        async def drain(self):
            pass

    class MockStdout:
        def __init__(self):
            self._lines = list(response_iter)

        async def readline(self):
            if self._lines:
                line = self._lines.pop(0)
                return (line + "\n").encode()
            return b""

    class MockProcess:
        stdin = MockStdin()
        stdout = MockStdout()
        stderr = None

        async def wait(self):
            pass

        def terminate(self):
            pass

        def kill(self):
            pass

    return MockProcess()


def _jsonrpc_response(request_id: int, result: Any) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result})


def _init_response(request_id: int) -> str:
    return _jsonrpc_response(request_id, {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "serverInfo": {"name": "test-server", "version": "1.0.0"},
    })


def _tools_list_response(request_id: int, tools: list[dict]) -> str:
    return _jsonrpc_response(request_id, {"tools": tools})


SAMPLE_TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file from disk",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_files",
        "description": "List files in a directory",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
]


async def test_bridge_start_discovers_tools(monkeypatch):
    """Start the bridge, perform handshake, discover tools."""
    process = _make_mock_process([
        _init_response(1),       # response to initialize
        _tools_list_response(2, SAMPLE_TOOLS),  # response to tools/list
    ])

    async def mock_create_subprocess(*args, **kwargs):
        return process

    monkeypatch.setattr(
        "the_agents_playbook.tools.mcp.asyncio.create_subprocess_exec",
        mock_create_subprocess,
    )

    bridge = MCPBridge("npx", ["-y", "test-server"])
    await bridge.start()

    tools = bridge.get_tools()
    assert len(tools) == 2
    assert tools[0].name == "read_file"
    assert tools[1].name == "list_files"

    await bridge.stop()


async def test_bridge_send_request(monkeypatch):
    """Send a JSON-RPC request and get a response."""
    process = _make_mock_process([
        _init_response(1),
        _tools_list_response(2, []),
        _jsonrpc_response(3, {"content": "file contents"}),
    ])

    async def mock_create_subprocess(*args, **kwargs):
        return process

    monkeypatch.setattr(
        "the_agents_playbook.tools.mcp.asyncio.create_subprocess_exec",
        mock_create_subprocess,
    )

    bridge = MCPBridge("npx", ["-y", "test-server"])
    await bridge.start()

    # Use the internal send_request to call a tool
    result = await bridge._send_request("tools/call", {
        "name": "read_file",
        "arguments": {"path": "/tmp/test.txt"},
    })
    assert result["content"] == "file contents"

    await bridge.stop()


async def test_mcp_tool_implements_protocol(monkeypatch):
    """Tools discovered via MCP should implement the Tool protocol."""
    process = _make_mock_process([
        _init_response(1),
        _tools_list_response(2, SAMPLE_TOOLS),
    ])

    async def mock_create_subprocess(*args, **kwargs):
        return process

    monkeypatch.setattr(
        "the_agents_playbook.tools.mcp.asyncio.create_subprocess_exec",
        mock_create_subprocess,
    )

    bridge = MCPBridge("npx", ["-y", "test-server"])
    await bridge.start()

    from the_agents_playbook.tools.protocol import Tool
    tools = bridge.get_tools()
    for tool in tools:
        assert isinstance(tool, Tool)
        assert tool.name
        assert tool.description
        assert tool.parameters

    await bridge.stop()


async def test_bridge_not_started_raises():
    """Calling get_tools before start should return empty."""
    bridge = MCPBridge("npx", ["-y", "test-server"])
    assert bridge.get_tools() == []


async def test_send_request_not_started_raises():
    """Sending a request without starting raises MCPConnectionError."""
    bridge = MCPBridge("npx", ["-y", "test-server"])
    with pytest.raises(MCPConnectionError, match="not started"):
        await bridge._send_request("tools/list")
