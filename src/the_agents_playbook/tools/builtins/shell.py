"""ShellTool — sandboxed subprocess execution with deny patterns."""

import asyncio
import logging
import re
import shlex
from pathlib import Path
from typing import Any

from ..protocol import Tool, ToolResult

logger = logging.getLogger(__name__)

DEFAULT_DENY_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"sudo\s+",
    r"chmod\s+777",
    r">\s*/dev/",
    r"mkfs",
    r"dd\s+if=",
    r":\(\)\{.*\};:",
]


class ShellTool(Tool):
    """Run shell commands in a subprocess with safety restrictions.

    Safety features:
    - Deny patterns: regex list of dangerous commands (rm -rf /, sudo, etc.)
    - Workspace restriction: commands run with cwd set to workspace path
    - Timeout: configurable execution timeout (default 30s)

    The deny pattern system blocks obvious destructive commands but is
    not a complete security boundary. For real sandboxing, use containers.
    """

    def __init__(
        self,
        deny_patterns: list[str] | None = None,
        workspace: Path | str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._deny_patterns = [re.compile(p) for p in (deny_patterns or DEFAULT_DENY_PATTERNS)]
        self._workspace = Path(workspace) if workspace else None
        self._timeout = timeout_seconds

    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        return "Execute a shell command and return its output. Use for running system commands."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        }

    async def execute(self, command: str, **kwargs: Any) -> ToolResult:
        cmd_str = command.strip()

        # Check deny patterns against the raw command string
        for pattern in self._deny_patterns:
            if pattern.search(cmd_str):
                return ToolResult(
                    output=f"Command blocked by deny pattern: {pattern.pattern}",
                    error=True,
                )

        logger.info("Executing shell command: %s", cmd_str)

        try:
            process = await asyncio.create_subprocess_shell(
                cmd_str,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._workspace) if self._workspace else None,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self._timeout
            )

            output_parts = []
            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))
            if stderr:
                output_parts.append(f"[stderr] {stderr.decode('utf-8', errors='replace')}")

            return_code = process.returncode or 0
            if return_code != 0:
                output_parts.append(f"[exit code: {return_code}]")

            return ToolResult(
                output="\n".join(output_parts) if output_parts else "(no output)",
                error=return_code != 0,
                metadata={"exit_code": return_code},
            )

        except asyncio.TimeoutError:
            return ToolResult(
                output=f"Command timed out after {self._timeout}s",
                error=True,
            )
        except Exception as e:
            return ToolResult(
                output=f"Failed to execute command: {e}",
                error=True,
            )
