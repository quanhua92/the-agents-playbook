"""07-draft-system.py — Workers compose actions, dispatchers commit them.

The draft-before-act pattern: instead of letting a worker agent directly
send emails or make API calls, it creates a Draft. A dispatcher agent
(or human) reviews pending drafts and approves or rejects them.

This is the principle of least privilege applied to agent actions.
No SDK imports beyond guardrails — self-contained demonstration.
"""

import asyncio

from the_agents_playbook.guardrails import (
    Draft,
    DraftKind,
    DraftStatus,
    DraftStore,
    DraftTool,
    ApprovalTool,
)


async def main():
    store = DraftStore()

    # --- Worker creates draft emails ---
    print("=== Worker Agent: Creating Drafts ===\n")

    draft_tool = DraftTool(store, worker_id="research-agent")

    # Draft 1: An email
    result = await draft_tool.execute(
        kind="email",
        summary="Weekly progress report to manager",
        payload={
            "to": "manager@example.com",
            "subject": "Weekly Progress Report",
            "body": "This week we completed the OAuth migration and fixed 3 critical bugs.",
        },
    )
    print(f"Worker: {result.output}\n")

    # Draft 2: An API call
    result = await draft_tool.execute(
        kind="api_call",
        summary="Deploy v2.1.0 to staging",
        payload={
            "endpoint": "https://api.example.com/deploy",
            "method": "POST",
            "body": {"version": "2.1.0", "env": "staging"},
        },
    )
    print(f"Worker: {result.output}\n")

    # Draft 3: A file write
    result = await draft_tool.execute(
        kind="file_write",
        summary="Update config with new database URL",
        payload={
            "path": "config/database.yml",
            "content": "url: postgres://new-db.example.com:5432/prod",
        },
    )
    print(f"Worker: {result.output}\n")

    # --- Dispatcher reviews pending drafts ---
    print("=== Dispatcher Agent: Reviewing Pending Drafts ===\n")

    approval_tool = ApprovalTool(store)

    # List all pending
    result = await approval_tool.execute(action="list")
    print(f"Dispatcher:\n{result.output}\n")

    # Approve the email
    pending = store.list_pending()
    if pending:
        email_draft = pending[0]
        print(f"--- Reviewing: {email_draft.summary} ---")
        print(f"  Kind: {email_draft.kind.value}")
        print(f"  Payload: {email_draft.payload}")
        print(f"  From: {email_draft.worker_id}\n")

        # Dispatcher approves
        result = await approval_tool.execute(
            action="approve",
            draft_id=email_draft.draft_id,
        )
        print(f"Dispatcher: {result.output}\n")

    # Reject the API call
    pending = store.list_pending()
    if pending:
        api_draft = pending[0]
        print(f"--- Reviewing: {api_draft.summary} ---")
        print("  Dispatcher: Rejecting (needs security review first)\n")

        result = await approval_tool.execute(
            action="reject",
            draft_id=api_draft.draft_id,
        )
        print(f"Dispatcher: {result.output}\n")

    # --- Show final state ---
    print("=== Final Draft Store State ===\n")

    for draft in store.all_drafts:
        status_icon = {
            DraftStatus.SENT: "✓",
            DraftStatus.REJECTED: "✗",
            DraftStatus.PENDING: "?",
            DraftStatus.EXPIRED: "!",
        }.get(draft.status, "?")
        print(
            f"  [{status_icon}] {draft.draft_id}  "
            f"{draft.kind.value:12s}  "
            f"{draft.status.value:10s}  "
            f"{draft.summary[:40]}"
        )

    # --- Expiration demo ---
    print("\n=== Expiration Demo ===\n")

    # Create a draft with very short expiration
    short_draft = store.save(
        Draft(
            kind=DraftKind.MESSAGE,
            summary="Quick notification (expires in 0s for demo)",
            expires_after_seconds=0.0,
        )
    )
    print(f"Created short-lived draft: {short_draft.draft_id}")

    expired = store.expire_stale()
    print(f"Expired drafts: {len(expired)}")
    for d in expired:
        print(f"  [{d.status.value}] {d.draft_id}: {d.summary[:40]}")


asyncio.run(main())
