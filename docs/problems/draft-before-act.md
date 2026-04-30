# Draft-Before-Act Safety Pattern

## Problem

The existing `PermissionMiddleware` and `RiskLevel` system can block or prompt for approval on individual tool calls, but there's no concept of staging an action for later review. A worker agent with `send_email` and `deploy` tools could fire off irreversible actions immediately. In a multi-agent system, workers should compose actions but never commit them directly.

## Solution

Four classes in `guardrails/drafts.py` implement a staging-and-approval pipeline:

| Class | Role |
|---|---|
| `Draft` | A staged action proposal with kind, summary, payload, and lifecycle status |
| `DraftStore` | In-memory registry that tracks all pending and resolved drafts |
| `DraftTool` | Tool for worker agents — creates drafts instead of executing actions |
| `ApprovalTool` | Tool for dispatchers/humans — lists, approves, and rejects pending drafts |

### Lifecycle

Every draft goes through one of four states:

```
PENDING → SENT       (approved by dispatcher/human)
PENDING → REJECTED   (rejected by dispatcher/human)
PENDING → EXPIRED   (timed out without approval, default 300s)
```

Once resolved, a draft cannot change state again.

### The flow

```
1. Worker agent decides to send an email
2. Worker calls DraftTool.execute(kind="email", summary="...", payload={...})
3. Draft is saved with status PENDING
4. Dispatcher calls ApprovalTool.execute(action="list")
5. Dispatcher reviews the draft details
6. Dispatcher calls ApprovalTool.execute(action="approve", draft_id="abc")
7. Draft status changes to SENT — action is committed
```

### Draft kinds

Six categories cover common external actions:

```
EMAIL     — send an email or message
API_CALL  — make an HTTP request to an external service
FILE_WRITE — create or modify a file on disk
COMMAND   — execute a shell command
MESSAGE   — send a notification
CUSTOM    — any other action type
```

### Creating drafts as a worker

```python
from the_agents_playbook.guardrails import DraftTool, DraftStore, DraftKind

store = DraftStore()
tool = DraftTool(store, worker_id="research-agent")

result = await tool.execute(
    kind="email",
    summary="Weekly progress report",
    payload={
        "to": "team@example.com",
        "subject": "Weekly Report",
        "body": "Completed OAuth migration this week.",
    },
)
# Returns: "Draft created: abc123\nKind: email\nStatus: pending\n..."
```

### Reviewing drafts as a dispatcher

```python
from the_agents_playbook.guardrails import ApprovalTool

tool = ApprovalTool(store)

# List all pending drafts
result = await tool.execute(action="list")

# Approve one
result = await tool.execute(action="approve", draft_id="abc123")

# Reject another
result = await tool.execute(action="reject", draft_id="def456")

# Expire stale drafts (call periodically)
expired = store.expire_stale()
```

### Relationship to existing guardrails

This extends the `PermissionMiddleware` + `RiskLevel` pattern:

- `RiskLevel.READ_ONLY` → tool executes immediately (no draft needed)
- `RiskLevel.WORKSPACE_WRITE` → tool could be wrapped with drafting
- `RiskLevel.DANGER` → tool should always go through the draft pipeline

In a multi-agent system, workers have `DraftTool` in their tool set instead of direct email/deploy tools. Only the dispatcher has `ApprovalTool`.

## Code Reference

- `src/the_agents_playbook/guardrails/drafts.py` — `Draft` dataclass, `DraftStore`, `DraftTool`, `ApprovalTool`, `DraftStatus`, `DraftKind`
- `src/the_agents_playbook/guardrails/permissions.py` — `RiskLevel`, `PermissionMiddleware` (the draft system extends this)

## Playground Example

- `06-human-in-the-loop/07-draft-system.py` — worker creates draft email, dispatcher approves/rejects

## LangGraph Example

- `langgraph-examples/06-human-in-the-loop/03_draft_approval.py` — uses `interrupt()` to pause and show draft details
