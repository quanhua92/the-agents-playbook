"""HookSystem — event-driven observability for the agent loop.

Hooks fire at key points in the agent lifecycle, enabling logging,
monitoring, audit trails, and custom middleware without modifying
the agent core.
"""

import logging
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

HookFn = Callable[..., Coroutine[Any, Any, None]]

# Standard hook events
ON_TURN_START = "on_turn_start"
ON_TOOL_CALL = "on_tool_call"
ON_TOOL_RESULT = "on_tool_result"
ON_TURN_END = "on_turn_end"


class HookSystem:
    """Register and emit named hook events.

    Usage:
        hooks = HookSystem()
        hooks.on("on_tool_call", my_logger)
        await hooks.emit("on_tool_call", tool_name="shell", arguments={...})
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[HookFn]] = {}

    def on(self, event: str, fn: HookFn) -> None:
        """Register a handler for a named event.

        Args:
            event: Event name (e.g., "on_tool_call").
            fn: Async function to call when the event fires.
        """
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(fn)
        logger.debug("Registered hook %s → %s", event, getattr(fn, "__qualname__", repr(fn)))

    def off(self, event: str, fn: HookFn | None = None) -> None:
        """Remove a handler (or all handlers) for an event.

        Args:
            event: Event name.
            fn: Specific handler to remove. If None, removes all handlers for the event.
        """
        if event not in self._handlers:
            return

        if fn is None:
            del self._handlers[event]
        else:
            self._handlers[event] = [h for h in self._handlers[event] if h is not fn]

    def handlers(self, event: str) -> list[HookFn]:
        """Return handlers registered for an event."""
        return self._handlers.get(event, [])

    async def emit(self, event: str, **kwargs: Any) -> None:
        """Fire all handlers for a named event.

        Handlers are called in registration order. Errors in one handler
        do not prevent subsequent handlers from running.

        Args:
            event: Event name.
            **kwargs: Data to pass to handlers.
        """
        hooks = self._handlers.get(event, [])
        if not hooks:
            return

        logger.debug("Emitting %s to %d handler(s)", event, len(hooks))

        for fn in hooks:
            try:
                await fn(**kwargs)
            except Exception:
                logger.exception("Hook handler %s raised an error", getattr(fn, "__qualname__", repr(fn)))

    def clear(self) -> None:
        """Remove all registered handlers."""
        self._handlers.clear()
