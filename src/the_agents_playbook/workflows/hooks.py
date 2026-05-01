"""Workflow hooks — event-driven plugin system for workflow execution.

Extends the HookSystem pattern with workflow-specific events:
pre/post step execution, workflow start/complete/fail. Enables
validation, audit, notifications, and custom middleware.
"""

import logging
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

HookFn = Callable[..., Coroutine[Any, Any, None]]

# Standard workflow hook events
PRE_STEP_EXECUTE = "pre_step_execute"
POST_STEP_EXECUTE = "post_step_execute"
PRE_WORKFLOW_RUN = "pre_workflow_run"
POST_WORKFLOW_RUN = "post_workflow_run"
ON_STEP_FAILURE = "on_step_failure"


class WorkflowHookSystem:
    """Register and emit workflow-specific hook events.

    Usage:
        hooks = WorkflowHookSystem()
        hooks.on(PRE_STEP_EXECUTE, validate_inputs)
        hooks.on(POST_STEP_EXECUTE, audit_result)
        await hooks.emit(PRE_STEP_EXECUTE, step_id="plan", input_data={...})
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[HookFn]] = {}

    def on(self, event: str, fn: HookFn) -> None:
        """Register a handler for a workflow event."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(fn)
        logger.debug(
            "Registered workflow hook %s → %s",
            event,
            getattr(fn, "__qualname__", repr(fn)),
        )

    def off(self, event: str, fn: HookFn | None = None) -> None:
        """Remove a handler (or all handlers) for an event."""
        if event not in self._handlers:
            return
        if fn is None:
            del self._handlers[event]
        else:
            self._handlers[event] = [h for h in self._handlers[event] if h is not fn]

    async def emit(self, event: str, **kwargs: Any) -> None:
        """Fire all handlers for a workflow event. Errors are isolated."""
        handlers = self._handlers.get(event, [])
        if not handlers:
            return

        for fn in handlers:
            try:
                await fn(**kwargs)
            except Exception:
                logger.exception(
                    "Workflow hook handler %s raised an error",
                    getattr(fn, "__qualname__", repr(fn)),
                )

    def handlers(self, event: str) -> list[HookFn]:
        """Return handlers registered for an event."""
        return self._handlers.get(event, [])

    def clear(self) -> None:
        """Remove all handlers."""
        self._handlers.clear()
