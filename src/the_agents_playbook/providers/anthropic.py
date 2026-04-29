"""Anthropic Messages API provider."""

import json
import logging
from typing import Any

import httpx

from the_agents_playbook import settings
from the_agents_playbook.providers.base import BaseProvider
from the_agents_playbook.providers.types import (
    InputMessage,
    MessageRequest,
    MessageResponse,
    OutputMessage,
    ProviderError,
)

logger = logging.getLogger(__name__)

# Maps Anthropic stop_reason values to provider-agnostic strings.
_STOP_REASON_MAP = {
    "end_turn": "end_turn",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
    "tool_use": "tool_calls",
}


class AnthropicProvider(BaseProvider):
    def __init__(self, **kwargs: Any):
        kwargs.setdefault("provider_name", "anthropic")
        super().__init__(**kwargs)

    # --- Required abstract implementations ---

    def _chat_endpoint(self) -> str:
        return settings.anthropic_base_url + "/messages"

    def _build_headers(self) -> dict[str, str]:
        return {
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "User-Agent": "TheAgentsPlaybook/0.1",
        }

    def _build_auth_from_pool(self) -> tuple[str, str]:
        """Anthropic uses x-api-key header instead of Bearer."""
        if not self._credential_pool:
            raise ProviderError("No credential pool configured")
        return ("x-api-key", self._credential_pool.current)

    def _build_body(self, request: MessageRequest) -> dict[str, Any]:
        # Anthropic puts system as a top-level field, not inside messages.
        body: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [
                m.model_dump(exclude_none=True) for m in request.messages
            ],
        }
        if request.system:
            body["system"] = request.system

        if request.tools:
            body["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in request.tools
            ]

        if request.tool_choice:
            tc = request.tool_choice
            if tc.type == "required":
                body["tool_choice"] = {"type": "any"}
            elif tc.type == "function":
                body["tool_choice"] = {
                    "type": "tool",
                    "name": tc.function_name,
                }
            # "auto" is the default for Anthropic — omit

        return body

    def _parse_response(self, response: httpx.Response) -> MessageResponse:
        raw = response.json()
        logger.debug("Raw response from Anthropic API: %s", json.dumps(raw, indent=2))

        content = raw.get("content", [])
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in content:
            if block["type"] == "text":
                text_parts.append(block["text"])
            elif block["type"] == "tool_use":
                tool_calls.append(
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                )

        stop_reason = _STOP_REASON_MAP.get(
            raw.get("stop_reason", "unknown"), "unknown"
        )

        return MessageResponse(
            message=OutputMessage(
                role="assistant",
                content="\n".join(text_parts) if text_parts else None,
                tool_calls=tool_calls,
            ),
            stop_reason=stop_reason,
        )
