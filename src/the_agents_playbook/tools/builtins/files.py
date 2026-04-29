"""FileReadTool and FileWriteTool — workspace-restricted file I/O."""

import logging
from pathlib import Path
from typing import Any

from ..protocol import Tool, ToolResult

logger = logging.getLogger(__name__)


def _resolve_and_check(path: str, workspace: Path) -> Path:
    """Resolve a path and verify it's within the workspace.

    Prevents path traversal attacks like '../../../etc/passwd'.
    """
    resolved = (workspace / path).resolve()
    workspace_resolved = workspace.resolve()

    try:
        resolved.relative_to(workspace_resolved)
    except ValueError:
        raise PermissionError(
            f"Path '{path}' resolves outside workspace '{workspace_resolved}'"
        )

    return resolved


class FileReadTool(Tool):
    """Read file contents, restricted to a workspace directory.

    Prevents reading files outside the configured workspace root
    via path traversal checks on the resolved path.
    """

    def __init__(self, workspace: Path | str) -> None:
        self._workspace = Path(workspace)

    @property
    def name(self) -> str:
        return "file_read"

    @property
    def description(self) -> str:
        return "Read the contents of a file. Returns the file content as text."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file, relative to workspace root",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        }

    async def execute(self, path: str, **kwargs: Any) -> ToolResult:
        try:
            resolved = _resolve_and_check(path, self._workspace)
        except PermissionError as e:
            return ToolResult(output=str(e), error=True)

        if not resolved.exists():
            return ToolResult(output=f"File not found: {path}", error=True)

        if not resolved.is_file():
            return ToolResult(output=f"Not a file: {path}", error=True)

        try:
            content = resolved.read_text(encoding="utf-8")
            return ToolResult(output=content)
        except Exception as e:
            return ToolResult(output=f"Failed to read file: {e}", error=True)


class FileWriteTool(Tool):
    """Write content to a file, restricted to a workspace directory.

    Creates parent directories if they don't exist.
    Prevents writing outside the workspace via path traversal checks.
    """

    def __init__(self, workspace: Path | str) -> None:
        self._workspace = Path(workspace)

    @property
    def name(self) -> str:
        return "file_write"

    @property
    def description(self) -> str:
        return "Write content to a file. Creates parent directories if needed."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file, relative to workspace root",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        }

    async def execute(self, path: str, content: str, **kwargs: Any) -> ToolResult:
        try:
            resolved = _resolve_and_check(path, self._workspace)
        except PermissionError as e:
            return ToolResult(output=str(e), error=True)

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            logger.info("Wrote %d bytes to %s", len(content), resolved)
            return ToolResult(output=f"Successfully wrote to {path}")
        except Exception as e:
            return ToolResult(output=f"Failed to write file: {e}", error=True)
