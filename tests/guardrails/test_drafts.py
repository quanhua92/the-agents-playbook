"""Tests for draft-before-act safety pattern."""

from the_agents_playbook.guardrails.drafts import (
    ApprovalTool,
    Draft,
    DraftKind,
    DraftStatus,
    DraftStore,
    DraftTool,
)


class TestDraft:
    def test_defaults(self):
        draft = Draft(summary="test")
        assert draft.status == DraftStatus.PENDING
        assert draft.kind == DraftKind.CUSTOM
        assert draft.draft_id
        assert draft.created_at > 0

    def test_is_pending(self):
        draft = Draft()
        assert draft.is_pending

    def test_not_pending_after_sent(self):
        draft = Draft()
        draft.status = DraftStatus.SENT
        assert not draft.is_pending


class TestDraftStore:
    def test_save_and_get(self):
        store = DraftStore()
        draft = Draft(summary="test draft")
        store.save(draft)
        assert store.get(draft.draft_id) is draft

    def test_get_missing(self):
        store = DraftStore()
        assert store.get("nonexistent") is None

    def test_list_pending(self):
        store = DraftStore()
        store.save(Draft(summary="pending 1"))
        store.save(Draft(summary="pending 2"))
        store.save(Draft(summary="also pending"))

        pending = store.list_pending()
        assert len(pending) == 3
        assert all(d.is_pending for d in pending)

    def test_list_pending_excludes_resolved(self):
        store = DraftStore()
        d1 = store.save(Draft(summary="pending"))
        d2 = store.save(Draft(summary="sent"))
        store.approve(d2.draft_id)

        pending = store.list_pending()
        assert len(pending) == 1
        assert pending[0].draft_id == d1.draft_id

    def test_list_pending_filter_by_worker(self):
        store = DraftStore()
        store.save(Draft(summary="from a", worker_id="agent-a"))
        store.save(Draft(summary="from b", worker_id="agent-b"))

        pending_a = store.list_pending(worker_id="agent-a")
        assert len(pending_a) == 1
        assert pending_a[0].worker_id == "agent-a"

    def test_approve(self):
        store = DraftStore()
        draft = store.save(Draft(summary="approve me"))
        result = store.approve(draft.draft_id)
        assert result.status == DraftStatus.SENT
        assert result.resolved_at is not None

    def test_approve_missing(self):
        store = DraftStore()
        assert store.approve("missing") is None

    def test_approve_already_resolved(self):
        store = DraftStore()
        draft = store.save(Draft(summary="test"))
        store.approve(draft.draft_id)
        result = store.approve(draft.draft_id)
        # Already SENT, not PENDING
        assert result.status == DraftStatus.SENT

    def test_reject(self):
        store = DraftStore()
        draft = store.save(Draft(summary="reject me"))
        result = store.reject(draft.draft_id)
        assert result.status == DraftStatus.REJECTED

    def test_reject_missing(self):
        store = DraftStore()
        assert store.reject("missing") is None

    def test_expire_stale(self):
        store = DraftStore()
        draft = store.save(
            Draft(
                summary="expires fast",
                expires_after_seconds=0.0,
            )
        )
        expired = store.expire_stale()
        assert len(expired) == 1
        assert draft.status == DraftStatus.EXPIRED

    def test_expire_skips_fresh(self):
        store = DraftStore()
        store.save(Draft(summary="fresh", expires_after_seconds=9999.0))
        expired = store.expire_stale()
        assert len(expired) == 0

    def test_all_drafts(self):
        store = DraftStore()
        store.save(Draft(summary="a"))
        store.save(Draft(summary="b"))
        assert len(store.all_drafts) == 2


class TestDraftTool:
    async def test_create_email_draft(self):
        store = DraftStore()
        tool = DraftTool(store, worker_id="worker-1")
        result = await tool.execute(
            kind="email",
            summary="Send report",
            payload={"to": "boss@example.com"},
        )
        assert not result.error
        assert "pending" in result.output.lower()
        assert store.list_pending()

    async def test_create_with_invalid_kind(self):
        store = DraftStore()
        tool = DraftTool(store)
        result = await tool.execute(
            kind="invalid_kind",
            summary="test",
            payload={},
        )
        assert not result.error
        # Falls back to CUSTOM
        draft = store.list_pending()[0]
        assert draft.kind == DraftKind.CUSTOM

    async def test_name_and_description(self):
        tool = DraftTool(DraftStore())
        assert tool.name == "create_draft"
        assert "stage" in tool.description.lower()


class TestApprovalTool:
    async def test_list_empty(self):
        tool = ApprovalTool(DraftStore())
        result = await tool.execute(action="list")
        assert "no pending" in result.output.lower()

    async def test_list_with_pending(self):
        store = DraftStore()
        store.save(Draft(summary="pending draft"))
        tool = ApprovalTool(store)
        result = await tool.execute(action="list")
        assert "1" in result.output

    async def test_approve(self):
        store = DraftStore()
        draft = store.save(Draft(summary="approve me"))
        tool = ApprovalTool(store)
        result = await tool.execute(action="approve", draft_id=draft.draft_id)
        assert "approved" in result.output.lower()

    async def test_approve_missing_id(self):
        tool = ApprovalTool(DraftStore())
        result = await tool.execute(action="approve")
        assert result.error

    async def test_reject(self):
        store = DraftStore()
        draft = store.save(Draft(summary="reject me"))
        tool = ApprovalTool(store)
        result = await tool.execute(action="reject", draft_id=draft.draft_id)
        assert "rejected" in result.output.lower()

    async def test_unknown_action(self):
        tool = ApprovalTool(DraftStore())
        result = await tool.execute(action="delete")
        assert result.error

    async def test_name_and_description(self):
        tool = ApprovalTool(DraftStore())
        assert tool.name == "manage_draft"
        assert "approve" in tool.description.lower()
