"""Shared async HTTP client with per-host rate limiting and retry logic."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import httpx

log = logging.getLogger(__name__)

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
        mailto: Optional[str] = None,
        allow_local_urls: bool = False,
    ) -> None:
        self._timeout = timeout
        self._max_retries = max_retries
        self._mailto = mailto
        # Drop the (mailto:...) portion entirely when no real address was
        # supplied — sending a placeholder demotes CrossRef polite-pool to
        # impolite under a fake identity, which is worse than going anonymous.
        if mailto:
            self._user_agent = f"CitationChecker/1.0 (mailto:{mailto})"
        else:
            self._user_agent = "CitationChecker/1.0"
        self._allow_local_urls = allow_local_urls
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

    async def get_html(self, url: str) -> httpx.Response:
        """GET a URL and return the full HTML response (status + body).

        Returning the response (not just the text) lets callers reject
        non-2xx pages — a 403 bot-challenge body would otherwise be parsed
        as a real article title.
        """
        return await self._request(url, params=None, accept="text/html,application/xhtml+xml")

    async def head_url(self, url: str) -> Optional[int]:
        """Issue a HEAD (or GET fallback) request. Returns status code or None on error.

        Walks redirect chains manually, gating every hop through the SSRF
        safe-URL check. Capped at 3 redirects.
        """
        assert self._client is not None, "Must be used as async context manager"
        if not _is_safe_url(url, allow_local=self._allow_local_urls):
            log.debug("URL check refused unsafe URL: %s", url)
            return None
        current = url
        try:
            for _ in range(4):  # initial + up to 3 redirects
                resp = await self._client.head(
                    current, timeout=self._timeout, follow_redirects=False,
                )
                if resp.status_code == 405:
                    resp = await self._client.get(
                        current, timeout=self._timeout, follow_redirects=False,
                    )
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location")
                    if not location:
                        return resp.status_code
                    next_url = str(httpx.URL(current).join(location))
                    if not _is_safe_url(next_url, allow_local=self._allow_local_urls):
                        log.debug("URL check refused unsafe redirect target: %s", next_url)
                        return None
                    current = next_url
                    continue
                return resp.status_code
            log.debug("URL check exceeded redirect cap for %s", url)
            return None
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
        if not _is_safe_url(url, allow_local=self._allow_local_urls):
            raise CitationHttpError(url, None, "blocked: unsafe URL")
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
    return urlparse(url).netloc


def _is_safe_url(url: str, *, allow_local: bool) -> bool:
    """Return False for URLs that should never be fetched.

    Blocks:
      - Schemes other than http/https.
      - Hosts that parse as an IP literal in private, loopback,
        link-local, reserved, or multicast ranges (unless ``allow_local``
        is True).
      - Empty hosts.

    Hostnames are NOT resolved via DNS — that would race with the actual
    request (DNS rebinding) and add latency. The parse-only check still
    blocks the obvious cases: ``http://127.0.0.1/``,
    ``http://169.254.169.254/`` (AWS metadata), ``http://[::1]/``.
    """
    try:
        parsed = urlparse(url)
    except (ValueError, TypeError):
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.hostname
    if not host:
        return False
    if allow_local:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # Hostname (not an IP literal) — allow.
        return True
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )
