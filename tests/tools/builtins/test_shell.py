"""Tests for tools/builtins/shell.py — ShellTool."""

from pathlib import Path

import pytest

from the_agents_playbook.tools.builtins.shell import ShellTool


@pytest.fixture
def shell_tool(tmp_path: Path) -> ShellTool:
    return ShellTool(workspace=tmp_path, timeout_seconds=5.0)


async def test_basic_command(shell_tool: ShellTool):
    result = await shell_tool.execute(command="echo hello")
    assert "hello" in result.output
    assert result.error is False


async def test_exit_code_captured(shell_tool: ShellTool):
    result = await shell_tool.execute(command="exit 1")
    assert result.error is True
    assert result.metadata["exit_code"] == 1


async def test_stderr_captured(shell_tool: ShellTool):
    result = await shell_tool.execute(command="echo error >&2")
    assert "[stderr]" in result.output


async def test_timeout(shell_tool: ShellTool):
    tool = ShellTool(timeout_seconds=0.01)
    result = await tool.execute(command="sleep 10")
    assert result.error is True
    assert "timed out" in result.output


async def test_deny_pattern_default(shell_tool: ShellTool):
    result = await shell_tool.execute(command="sudo rm -rf /")
    assert result.error is True
    assert "blocked" in result.output


async def test_deny_pattern_custom():
    tool = ShellTool(deny_patterns=[r"dangerous"])
    result = await tool.execute(command="echo dangerous command")
    assert result.error is True
    assert "blocked" in result.output


async def test_workspace_restriction(shell_tool: ShellTool):
    """Commands run with cwd set to workspace."""
    result = await shell_tool.execute(command="pwd")
    assert str(shell_tool._workspace) in result.output


async def test_name_and_description():
    tool = ShellTool()
    assert tool.name == "shell"
    assert "shell command" in tool.description.lower()


async def test_parameters_schema():
    tool = ShellTool()
    schema = tool.parameters
    assert schema["type"] == "object"
    assert "command" in schema["properties"]
    assert "command" in schema["required"]
