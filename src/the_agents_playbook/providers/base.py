import asyncio
import json
import logging
import random
from abc import ABC, abstractmethod
from typing import Any

import httpx

from the_agents_playbook.providers.types import (
    MessageRequest,
    MessageResponse,
    ProviderErrorCode,
    ProviderError,
    RetryConfig,
)

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    _client: httpx.AsyncClient | None = None

    def __init__(self, retry_config: RetryConfig | None = None):
        self._retry_config = retry_config or RetryConfig()

    # --- Abstract methods ---
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
        return await self._with_retry(lambda: self._do_send(request))

    # --- Internal send (single attempt, no retry) ---
    async def _do_send(self, request: MessageRequest) -> MessageResponse:
        client = await self._get_client()
        body = self._build_body(request)
        logger.info(
            f"Sending request to {self._chat_endpoint()} with body: {json.dumps(body, indent=2)}"
        )
        response = await client.post(self._chat_endpoint(), json=body)
        self._check_status(response)
        return self._parse_response(response)

    # --- Retry with exponential backoff + jitter ---
    async def _with_retry(self, fn):
        """Execute fn with exponential backoff + jitter for retryable errors.

        Formula:
            delay = min(base_delay * 2^attempt, max_delay)
            if jitter: delay *= random.uniform(0.5, 1.5)

        Only retries if the caught ProviderError has retryable=True
        AND error.code is in retryable_codes.
        """
        cfg = self._retry_config
        last_error = None

        for attempt in range(cfg.max_retries + 1):
            try:
                return await fn()
            except ProviderError as e:
                last_error = e
                if not e.retryable or e.code not in cfg.retryable_codes:
                    raise
                if attempt == cfg.max_retries:
                    raise

                delay = min(cfg.base_delay * (2 ** attempt), cfg.max_delay)
                if cfg.jitter:
                    delay *= random.uniform(0.5, 1.5)

                logger.warning(
                    f"Retry {attempt + 1}/{cfg.max_retries} after {e.code.value} "
                    f"(HTTP {e.status_code}): {e}. Waiting {delay:.1f}s..."
                )
                await asyncio.sleep(delay)

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                last_error = ProviderError(
                    str(e),
                    code=ProviderErrorCode.CONNECTION_FAILED,
                    retryable=True,
                )
                if attempt == cfg.max_retries:
                    raise last_error

                delay = min(cfg.base_delay * (2 ** attempt), cfg.max_delay)
                if cfg.jitter:
                    delay *= random.uniform(0.5, 1.5)

                logger.warning(
                    f"Retry {attempt + 1}/{cfg.max_retries} after connection error: {e}. "
                    f"Waiting {delay:.1f}s..."
                )
                await asyncio.sleep(delay)

        raise last_error

    # --- Error classification ---
    def _check_status(self, response: httpx.Response) -> None:
        """Map HTTP status to typed ProviderError. Raises on non-2xx."""
        status = response.status_code
        if 200 <= status < 300:
            return

        try:
            body = response.json()
        except Exception:
            body = None

        if status in (401, 403):
            raise ProviderError(
                f"Authentication failed: {body}",
                code=ProviderErrorCode.AUTH_FAILED,
                status_code=status,
                retryable=False,
                raw_body=body,
            )
        elif status == 429:
            raise ProviderError(
                f"Rate limited: {body}",
                code=ProviderErrorCode.RATE_LIMITED,
                status_code=status,
                retryable=True,
                raw_body=body,
            )
        elif status == 400:
            raise ProviderError(
                f"Bad request: {body}",
                code=ProviderErrorCode.BAD_REQUEST,
                status_code=status,
                retryable=False,
                raw_body=body,
            )
        elif status >= 500:
            raise ProviderError(
                f"Server error {status}: {body}",
                code=ProviderErrorCode.SERVER_ERROR,
                status_code=status,
                retryable=True,
                raw_body=body,
            )
        else:
            raise ProviderError(
                f"Unexpected HTTP {status}: {body}",
                code=ProviderErrorCode.UNKNOWN,
                status_code=status,
                retryable=False,
                raw_body=body,
            )

    # --- HTTP client management ---
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
