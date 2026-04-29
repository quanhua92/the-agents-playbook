"""Tests for context/metadata.py — inject_date, inject_cwd, inject_git_status."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from the_agents_playbook.context.metadata import (
    _no_git_layer,
    inject_cwd,
    inject_date,
    inject_git_status,
)
from the_agents_playbook.context.layers import LayerPriority


# ---------------------------------------------------------------------------
# inject_date
# ---------------------------------------------------------------------------


def test_inject_date_returns_dynamic_layer():
    layer = inject_date()
    assert layer.priority == LayerPriority.DYNAMIC
    assert layer.name == "date"
    assert "Current date:" in layer.content
    assert "Current time:" in layer.content
    assert "Unix timestamp:" in layer.content


def test_inject_date_has_valid_date():
    layer = inject_date()
    # Should contain a date in YYYY-MM-DD format
    import re
    assert re.search(r"\d{4}-\d{2}-\d{2}", layer.content)


# ---------------------------------------------------------------------------
# inject_cwd
# ---------------------------------------------------------------------------


def test_inject_cwd_default():
    layer = inject_cwd()
    assert layer.priority == LayerPriority.DYNAMIC
    assert layer.name == "cwd"
    assert "Working directory:" in layer.content


def test_inject_cwd_custom():
    layer = inject_cwd(cwd="/tmp/myproject")
    resolved = Path("/tmp/myproject").resolve()
    assert layer.content == f"Working directory: {resolved}"


def test_inject_cwd_with_path_object():
    layer = inject_cwd(cwd=Path("/var/log"))
    assert "Working directory:" in layer.content
    assert "log" in layer.content


# ---------------------------------------------------------------------------
# inject_git_status
# ---------------------------------------------------------------------------


async def test_inject_git_status_in_repo():
    """Test git status inside a real git repo (this repo)."""
    layer = await inject_git_status(repo_root=Path(__file__).parent.parent.parent)
    assert layer.priority == LayerPriority.DYNAMIC
    assert layer.name == "git_status"
    # This repo is a git repo, so we should see git info
    assert "Git repository:" in layer.content


async def test_inject_git_status_not_in_repo():
    """Test git status outside a git repo."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        layer = await inject_git_status(repo_root=tmpdir)
        assert layer.content == "Not inside a git repository."


async def test_no_git_layer():
    layer = _no_git_layer()
    assert layer.name == "git_status"
    assert layer.priority == LayerPriority.DYNAMIC
    assert layer.content == "Not inside a git repository."


async def test_inject_git_status_git_not_found():
    """When git binary is not found, should return no-git layer."""
    with patch("shutil.which", return_value=None):
        # inject_git_status uses asyncio.create_subprocess_exec
        # If git binary doesn't exist, FileNotFoundError is caught
        layer = await inject_git_status(repo_root="/nonexistent")
        assert "Not inside a git repository" in layer.content


async def test_inject_git_status_timeout():
    """When git commands timeout, should gracefully degrade."""
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(side_effect=TimeoutError("slow"))
    mock_proc.returncode = 1
    mock_proc.stdout = None

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        layer = await inject_git_status(repo_root="/tmp")
        assert "Not inside a git repository" in layer.content
