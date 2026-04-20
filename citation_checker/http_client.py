"""Shared async HTTP client with per-host rate limiting and retry logic."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

log = logging.getLogger(__name__)

_DEFAULT_MAILTO = "citation-checker@example.com"

# Per-host rate-limit configuration
_HOST_CONFIG: dict[str, dict] = {
    "api.crossref.org":           {"concurrency": 5, "min_delay": 0.1},
    "export.arxiv.org":           {"concurrency": 1, "min_delay": 3.0},
    "api.openalex.org":           {"concurrency": 5, "min_delay": 0.1},
    "api.semanticscholar.org":    {"concurrency": 1, "min_delay": 1.5},
}
_DEFAULT_HOST_CONFIG = {"concurrency": 3, "min_delay": 0.5}


class CitationHttpError(Exception):
    """Raised when all retries are exhausted or a permanent error occurs."""

    def __init__(self, url: str, status_code: Optional[int], message: str) -> None:
        self.url = url
        self.status_code = status_code
        super().__init__(f"{message} [url={url}, status={status_code}]")


@dataclass
class _HostState:
    semaphore: asyncio.Semaphore
    min_delay: float
    last_request_at: float = field(default=0.0)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class CitationHttpClient:
    """Async HTTP client with per-host rate limiting and exponential-backoff retries."""

    def __init__(
        self,
        timeout: float = 10.0,
        max_retries: int = 3,
        mailto: str = _DEFAULT_MAILTO,
    ) -> None:
        self._timeout = timeout
        self._max_retries = max_retries
        self._mailto = mailto
        self._user_agent = f"CitationChecker/1.0 (mailto:{mailto})"
        self._host_states: dict[str, _HostState] = {}
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "CitationHttpClient":
        self._client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(self._timeout),
            headers={"User-Agent": self._user_agent},
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_json(self, url: str, params: Optional[dict] = None) -> dict:
        """GET a URL and return parsed JSON."""
        response = await self._request(url, params=params, accept="application/json")
        try:
            return response.json()
        except Exception as exc:
            raise CitationHttpError(url, response.status_code, f"JSON decode error: {exc}") from exc

    async def get_xml(self, url: str, params: Optional[dict] = None) -> str:
        """GET a URL and return the response body as text (for XML feeds)."""
        response = await self._request(url, params=params, accept="application/atom+xml")
        return response.text

    async def get_html(self, url: str) -> str:
        """GET a URL and return the response body as HTML text."""
        response = await self._request(url, params=None, accept="text/html,application/xhtml+xml")
        return response.text

    async def head_url(self, url: str) -> Optional[int]:
        """Issue a HEAD (or GET fallback) request. Returns status code or None on error."""
        assert self._client is not None, "Must be used as async context manager"
        try:
            resp = await self._client.head(url, timeout=self._timeout)
            if resp.status_code == 405:
                resp = await self._client.get(url, timeout=self._timeout)
            return resp.status_code
        except Exception as exc:
            log.debug("URL check failed for %s: %s", url, exc)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _request(
        self, url: str, params: Optional[dict], accept: str
    ) -> httpx.Response:
        assert self._client is not None, "Must be used as async context manager"
        host = _extract_host(url)
        state = self._get_host_state(host)

        backoff = 2.0
        last_status: Optional[int] = None

        for attempt in range(self._max_retries):
            await self._acquire(state)
            try:
                resp = await self._client.get(
                    url,
                    params=params,
                    headers={"Accept": accept},
                )
                last_status = resp.status_code

                if resp.status_code == 429:
                    retry_after = float(
                        resp.headers.get("Retry-After", backoff ** attempt)
                    )
                    log.debug("Rate limited on %s, sleeping %.1fs", host, retry_after)
                    await asyncio.sleep(retry_after)
                    continue

                if resp.status_code >= 500:
                    delay = backoff ** attempt
                    log.debug("Server error %d on %s, retry in %.1fs", resp.status_code, host, delay)
                    await asyncio.sleep(delay)
                    continue

                # 404 and other 4xx are returned as-is for callers to handle
                return resp

            except httpx.TimeoutException as exc:
                if attempt == self._max_retries - 1:
                    raise CitationHttpError(url, None, "Timeout after retries") from exc
                delay = backoff ** attempt
                log.debug("Timeout on %s, retry in %.1fs", url, delay)
                await asyncio.sleep(delay)

            except httpx.RequestError as exc:
                if attempt == self._max_retries - 1:
                    raise CitationHttpError(url, None, f"Request error: {exc}") from exc
                await asyncio.sleep(backoff ** attempt)

            finally:
                self._release(state)

        raise CitationHttpError(url, last_status, "Max retries exceeded")

    def _get_host_state(self, host: str) -> _HostState:
        if host not in self._host_states:
            cfg = _HOST_CONFIG.get(host, _DEFAULT_HOST_CONFIG)
            self._host_states[host] = _HostState(
                semaphore=asyncio.Semaphore(cfg["concurrency"]),
                min_delay=cfg["min_delay"],
            )
        return self._host_states[host]

    async def _acquire(self, state: _HostState) -> None:
        await state.semaphore.acquire()
        async with state.lock:
            elapsed = time.monotonic() - state.last_request_at
            if elapsed < state.min_delay:
                await asyncio.sleep(state.min_delay - elapsed)
            state.last_request_at = time.monotonic()

    def _release(self, state: _HostState) -> None:
        state.semaphore.release()


def _extract_host(url: str) -> str:
    from urllib.parse import urlparse
    return urlparse(url).netloc
