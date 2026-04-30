"""Draft-before-act — workers compose actions, dispatchers commit them.

The principle of least privilege applied to agents. Workers should never
directly perform irreversible external actions (send emails, make API calls,
deploy code). Instead, they create Drafts — staged action proposals that
a dispatcher or human must explicitly approve before execution.

This pattern naturally extends the existing PermissionMiddleware + RiskLevel
system: drafts are the mechanism by which a worker agent's proposed actions
are reviewed before commitment.
"""

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..tools.protocol import Tool, ToolResult

logger = logging.getLogger(__name__)


class DraftStatus(str, Enum):
    """Lifecycle states for a draft action."""

    PENDING = "pending"      # Created by worker, awaiting review
    SENT = "sent"            # Approved and committed by dispatcher
    REJECTED = "rejected"    # Explicitly rejected
    EXPIRED = "expired"      # Timed out without approval


class DraftKind(str, Enum):
    """Categories of drafted actions."""

    EMAIL = "email"
    API_CALL = "api_call"
    FILE_WRITE = "file_write"
    COMMAND = "command"
    MESSAGE = "message"
    CUSTOM = "custom"


@dataclass
class Draft:
    """A staged action proposal from a worker agent.

    Attributes:
        draft_id: Unique identifier for this draft.
        kind: Category of the proposed action.
        summary: Human-readable summary for the reviewer.
        payload: The full action details (email body, API params, etc.).
        status: Current lifecycle state.
        worker_id: Which agent created this draft.
        created_at: Timestamp when the draft was created.
        resolved_at: Timestamp when the draft was approved/rejected.
        expires_after_seconds: How long before the draft auto-expires.
    """

    draft_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    kind: DraftKind = DraftKind.CUSTOM
    summary: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    status: DraftStatus = DraftStatus.PENDING
    worker_id: str = ""
    created_at: float = 0.0
    resolved_at: float | None = None
    expires_after_seconds: float = 300.0  # 5 minutes default

    def __post_init__(self) -> None:
        if self.created_at == 0.0:
            import time
            self.created_at = time.monotonic()

    @property
    def is_pending(self) -> bool:
        return self.status == DraftStatus.PENDING


class DraftStore:
    """In-memory store for draft actions.

    Tracks all pending and resolved drafts. Workers create drafts,
    dispatchers list pending ones and approve/reject them.

    Usage:
        store = DraftStore()
        draft = store.save(Draft(
            kind=DraftKind.EMAIL,
            summary="Send meeting recap to team",
            payload={"to": "team@example.com", "subject": "...", "body": "..."},
            worker_id="research-agent",
        ))
        pending = store.list_pending()
        store.approve(draft.draft_id)
    """

    def __init__(self) -> None:
        self._drafts: dict[str, Draft] = {}

    def save(self, draft: Draft) -> Draft:
        """Save a new draft to the store."""
        self._drafts[draft.draft_id] = draft
        logger.info(
            "Draft created: %s [%s] %s",
            draft.draft_id, draft.kind.value, draft.summary[:50],
        )
        return draft

    def get(self, draft_id: str) -> Draft | None:
        """Look up a draft by ID."""
        return self._drafts.get(draft_id)

    def list_pending(self, worker_id: str | None = None) -> list[Draft]:
        """List all pending drafts, optionally filtered by worker."""
        drafts = [
            d for d in self._drafts.values()
            if d.is_pending
        ]
        if worker_id:
            drafts = [d for d in drafts if d.worker_id == worker_id]
        return sorted(drafts, key=lambda d: d.created_at)

    def approve(self, draft_id: str) -> Draft | None:
        """Approve a pending draft."""
        import time
        draft = self._drafts.get(draft_id)
        if draft is None:
            return None
        if not draft.is_pending:
            logger.warning("Cannot approve draft %s: status=%s", draft_id, draft.status)
            return draft
        draft.status = DraftStatus.SENT
        draft.resolved_at = time.monotonic()
        logger.info("Draft approved: %s [%s]", draft_id, draft.kind.value)
        return draft

    def reject(self, draft_id: str) -> Draft | None:
        """Reject a pending draft."""
        import time
        draft = self._drafts.get(draft_id)
        if draft is None:
            return None
        if not draft.is_pending:
            logger.warning("Cannot reject draft %s: status=%s", draft_id, draft.status)
            return draft
        draft.status = DraftStatus.REJECTED
        draft.resolved_at = time.monotonic()
        logger.info("Draft rejected: %s [%s]", draft_id, draft.kind.value)
        return draft

    def expire_stale(self) -> list[Draft]:
        """Mark pending drafts that have exceeded their expiration time."""
        import time
        now = time.monotonic()
        expired: list[Draft] = []
        for draft in self._drafts.values():
            if not draft.is_pending:
                continue
            elapsed = now - draft.created_at
            if elapsed > draft.expires_after_seconds:
                draft.status = DraftStatus.EXPIRED
                draft.resolved_at = now
                expired.append(draft)
                logger.info("Draft expired: %s [%s]", draft.draft_id, draft.kind.value)
        return expired

    @property
    def all_drafts(self) -> list[Draft]:
        """Return all drafts (including resolved)."""
        return list(self._drafts.values())


class DraftTool(Tool):
    """Tool for worker agents to stage actions as drafts.

    Workers call this tool instead of directly performing external actions.
    The draft is stored for review by the dispatcher or a human.

    Usage:
        store = DraftStore()
        tool = DraftTool(store, worker_id="research-agent")
        result = await tool.execute(
            kind="email",
            summary="Send weekly report",
            payload={"to": "boss@example.com", "body": "..."},
        )
        # Returns: DraftResult with draft_id, "pending" status
    """

    def __init__(self, store: DraftStore, worker_id: str = "worker") -> None:
        self._store = store
        self._worker_id = worker_id

    @property
    def name(self) -> str:
        return "create_draft"

    @property
    def description(self) -> str:
        return (
            "Stage an action for approval instead of executing it directly. "
            "Use this for emails, API calls, and other external actions. "
            "Returns a draft ID that must be approved before execution."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": [k.value for k in DraftKind],
                    "description": "Type of action to draft",
                },
                "summary": {
                    "type": "string",
                    "description": "Brief human-readable summary of the action",
                },
                "payload": {
                    "type": "object",
                    "description": "Full action details (email body, API params, etc.)",
                },
            },
            "required": ["kind", "summary", "payload"],
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        kind_str = kwargs.get("kind", "custom")
        try:
            kind = DraftKind(kind_str)
        except ValueError:
            kind = DraftKind.CUSTOM

        draft = self._store.save(Draft(
            kind=kind,
            summary=kwargs.get("summary", ""),
            payload=kwargs.get("payload", {}),
            worker_id=self._worker_id,
        ))

        return ToolResult(
            output=(
                f"Draft created: {draft.draft_id}\n"
                f"Kind: {draft.kind.value}\n"
                f"Status: {draft.status.value}\n"
                f"Summary: {draft.summary}\n"
                f"Awaiting approval before execution."
            ),
        )


class ApprovalTool(Tool):
    """Tool for dispatchers to approve or reject pending drafts.

    Usage:
        store = DraftStore()
        tool = ApprovalTool(store)
        result = await tool.execute(action="approve", draft_id="abc123")
    """

    def __init__(self, store: DraftStore) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "manage_draft"

    @property
    def description(self) -> str:
        return (
            "Approve or reject a pending draft action. "
            "Use 'list' to see all pending drafts, "
            "'approve' to commit a draft, or 'reject' to cancel it."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "approve", "reject"],
                    "description": "Action to perform",
                },
                "draft_id": {
                    "type": "string",
                    "description": "Draft ID (required for approve/reject)",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "list")

        if action == "list":
            pending = self._store.list_pending()
            if not pending:
                return ToolResult(output="No pending drafts.")

            lines = [f"Pending drafts ({len(pending)}):"]
            for d in pending:
                lines.append(
                    f"  [{d.draft_id}] {d.kind.value}: {d.summary} "
                    f"(from {d.worker_id})"
                )
            return ToolResult(output="\n".join(lines))

        draft_id = kwargs.get("draft_id", "")
        if not draft_id:
            return ToolResult(
                output="Error: draft_id required for approve/reject",
                error=True,
            )

        if action == "approve":
            draft = self._store.approve(draft_id)
            if draft is None:
                return ToolResult(output=f"Draft {draft_id} not found.", error=True)
            if draft.status == DraftStatus.SENT:
                return ToolResult(
                    output=(
                        f"Draft {draft_id} approved and committed.\n"
                        f"Action: {draft.kind.value}\n"
                        f"Payload: {draft.payload}"
                    ),
                )
            return ToolResult(
                output=f"Cannot approve draft {draft_id}: status={draft.status.value}",
                error=True,
            )

        elif action == "reject":
            draft = self._store.reject(draft_id)
            if draft is None:
                return ToolResult(output=f"Draft {draft_id} not found.", error=True)
            if draft.status == DraftStatus.REJECTED:
                return ToolResult(output=f"Draft {draft_id} rejected.")
            return ToolResult(
                output=f"Cannot reject draft {draft_id}: status={draft.status.value}",
                error=True,
            )

        return ToolResult(output=f"Unknown action: {action}", error=True)
