"""Tests for tools/builtins/files.py — FileReadTool and FileWriteTool."""

from pathlib import Path

import pytest

from the_agents_playbook.tools.builtins.files import FileReadTool, FileWriteTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def read_tool(workspace: Path) -> FileReadTool:
    return FileReadTool(workspace=workspace)


@pytest.fixture
def write_tool(workspace: Path) -> FileWriteTool:
    return FileWriteTool(workspace=workspace)


# --- FileWriteTool ---


async def test_write_creates_file(write_tool: FileWriteTool, workspace: Path):
    result = await write_tool.execute(path="hello.txt", content="world")
    assert result.error is False
    assert (workspace / "hello.txt").read_text() == "world"


async def test_write_creates_nested_dirs(write_tool: FileWriteTool, workspace: Path):
    result = await write_tool.execute(path="a/b/c.txt", content="nested")
    assert result.error is False
    assert (workspace / "a" / "b" / "c.txt").read_text() == "nested"


async def test_write_overwrites(write_tool: FileWriteTool, workspace: Path):
    await write_tool.execute(path="f.txt", content="first")
    result = await write_tool.execute(path="f.txt", content="second")
    assert result.error is False
    assert (workspace / "f.txt").read_text() == "second"


async def test_write_path_traversal(write_tool: FileWriteTool):
    result = await write_tool.execute(path="../../etc/passwd", content="hack")
    assert result.error is True
    assert "outside workspace" in result.output.lower()


# --- FileReadTool ---


async def test_read_existing(write_tool: FileWriteTool, read_tool: FileReadTool):
    await write_tool.execute(path="data.txt", content="hello world")
    result = await read_tool.execute(path="data.txt")
    assert result.error is False
    assert result.output == "hello world"


async def test_read_not_found(read_tool: FileReadTool):
    result = await read_tool.execute(path="nonexistent.txt")
    assert result.error is True
    assert "not found" in result.output.lower()


async def test_read_directory(read_tool: FileReadTool, workspace: Path):
    result = await read_tool.execute(path=".")
    assert result.error is True
    assert "not a file" in result.output.lower()


async def test_read_path_traversal(read_tool: FileReadTool):
    result = await read_tool.execute(path="../../etc/passwd")
    assert result.error is True
    assert "outside workspace" in result.output.lower()


async def test_read_symlink_traversal(read_tool: FileReadTool, workspace: Path):
    """Symlinks pointing outside workspace should be blocked."""
    # Create a symlink inside workspace pointing outside
    link = workspace / "escape"
    try:
        link.symlink_to("/etc")
    except OSError:
        pytest.skip("symlink creation not supported")
    result = await read_tool.execute(path="escape/passwd")
    assert result.error is True


# --- Tool protocol ---


def test_read_tool_name():
    tool = FileReadTool(workspace="/tmp")
    assert tool.name == "file_read"


def test_write_tool_name():
    tool = FileWriteTool(workspace="/tmp")
    assert tool.name == "file_write"


def test_parameters_schema():
    read_tool = FileReadTool(workspace="/tmp")
    write_tool = FileWriteTool(workspace="/tmp")

    assert read_tool.parameters["required"] == ["path"]
    assert write_tool.parameters["required"] == ["path", "content"]
