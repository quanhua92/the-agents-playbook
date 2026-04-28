import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx

from the_agents_playbook.providers.types import MessageRequest, MessageResponse


class BaseProvider(ABC):
    _client: httpx.AsyncClient | None = None

    # --- Abstract methods  ---
    @abstractmethod
    def _build_body(self, request: MessageRequest) -> dict[str, Any]:
        pass

    @abstractmethod
    def _build_headers(self) -> dict[str, str]:
        pass

    @abstractmethod
    def _chat_endpoint(self) -> str:
        pass

    @abstractmethod
    def _parse_response(self, response: httpx.Response) -> MessageResponse:
        pass

    # --- Public API ---
    async def send_message(self, request: MessageRequest) -> MessageResponse:
        client = await self._get_client()
        body = self._build_body(request)
        logging.info(
            f"Client configured with base URL: {client.base_url} and headers: {client.headers}"
        )
        logging.info(
            f"Sending request to {self._chat_endpoint()} with body: {json.dumps(body, indent=2)}"
        )
        response = await client.post(self._chat_endpoint(), json=body)
        response.raise_for_status()
        return self._parse_response(response)

    # -- Helpers ---
    # -- HTTP client management ---
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None and not self._client.is_closed:
            return self._client
        self._client = httpx.AsyncClient(
            headers=self._build_headers(),
            timeout=httpx.Timeout(10.0, connect=5.0, read=120.0, write=10.0, pool=10.0),
        )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
