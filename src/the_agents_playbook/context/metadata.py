"""Structured metadata injection — date, cwd, git status as ContextLayer instances.

Each injector returns a DYNAMIC priority layer. Git status gracefully degrades
when not inside a git repository.
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from .layers import ContextLayer, LayerPriority

logger = logging.getLogger(__name__)


def inject_date() -> ContextLayer:
    """Inject current date and time as a context layer."""
    now = datetime.now(timezone.utc)
    content = (
        f"Current date: {now.strftime('%Y-%m-%d')}\n"
        f"Current time: {now.strftime('%H:%M:%S UTC')}\n"
        f"Unix timestamp: {int(now.timestamp())}"
    )
    return ContextLayer(
        name="date",
        content=content,
        priority=LayerPriority.DYNAMIC,
    )


def inject_cwd(cwd: Path | str | None = None) -> ContextLayer:
    """Inject current working directory as a context layer."""
    path = Path(cwd) if cwd else Path.cwd()
    content = f"Working directory: {path.resolve()}"
    return ContextLayer(
        name="cwd",
        content=content,
        priority=LayerPriority.DYNAMIC,
    )


async def inject_git_status(repo_root: Path | str | None = None) -> ContextLayer:
    """Inject git status as a context layer.

    Runs `git status --porcelain` and `git log --oneline -5` to show
    the current state of the repository. Returns a minimal layer if
    not inside a git repo.
    """
    root = Path(repo_root).resolve() if repo_root else Path.cwd()

    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "rev-parse",
            "--show-toplevel",
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        if proc.returncode != 0:
            return _no_git_layer()
        git_root = Path(stdout.decode().strip())

    except FileNotFoundError, asyncio.TimeoutError, OSError:
        return _no_git_layer()

    lines: list[str] = [f"Git repository: {git_root}"]

    # Current branch
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
            cwd=git_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        if proc.returncode == 0:
            branch = stdout.decode().strip()
            lines.append(f"Current branch: {branch}")
    except FileNotFoundError, asyncio.TimeoutError, OSError:
        pass

    # Recent commits
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "log",
            "--oneline",
            "-5",
            cwd=git_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        if proc.returncode == 0 and stdout.decode().strip():
            lines.append("Recent commits:")
            for line in stdout.decode().strip().split("\n"):
                lines.append(f"  {line}")
    except FileNotFoundError, asyncio.TimeoutError, OSError:
        pass

    # Working tree status
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "status",
            "--porcelain",
            cwd=git_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        if proc.returncode == 0:
            status_output = stdout.decode().strip()
            if status_output:
                changed = status_output.count("\n") + 1
                lines.append(f"Working tree changes: {changed} file(s)")
                for line in status_output.split("\n")[:10]:
                    lines.append(f"  {line}")
            else:
                lines.append("Working tree: clean")
    except FileNotFoundError, asyncio.TimeoutError, OSError:
        pass

    return ContextLayer(
        name="git_status",
        content="\n".join(lines),
        priority=LayerPriority.DYNAMIC,
    )


def _no_git_layer() -> ContextLayer:
    """Return a minimal layer indicating no git repository was detected."""
    return ContextLayer(
        name="git_status",
        content="Not inside a git repository.",
        priority=LayerPriority.DYNAMIC,
    )
