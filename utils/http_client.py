"""
Async HTTP client singleton using httpx.

The singleton is recycled periodically (and on every timeout) so a stale
connection pool can't poison the whole process. httpx Limits has a
keepalive_expiry that aggressively reaps idle sockets; combined with
this self-healing behaviour, individual VLR.gg outages no longer leave
the backend permanently wedged.
"""
import asyncio
import logging
import time

import httpx

from utils.utils import headers
from utils.constants import DEFAULT_REQUEST_DELAY, DEFAULT_RETRIES, DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {500, 502, 503, 504}

_client: httpx.AsyncClient | None = None
_client_created_at: float = 0.0
# After this many seconds, the next get_http_client() call recycles the
# pool — even if it's still functional. Cheap insurance against zombie
# keepalive connections building up. Aggressive (4 min) because Railway
# egress to VLR.gg has been observed to silently degrade over time.
_CLIENT_TTL = 240  # 4 minutes


def _build_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers=headers,
        timeout=httpx.Timeout(DEFAULT_TIMEOUT),
        follow_redirects=True,
        limits=httpx.Limits(
            max_connections=50,
            max_keepalive_connections=5,
            keepalive_expiry=10.0,
        ),
    )


def get_http_client() -> httpx.AsyncClient:
    """Get or (re)create the async HTTP client."""
    global _client, _client_created_at
    now = time.time()
    if (
        _client is None
        or _client.is_closed
        or now - _client_created_at > _CLIENT_TTL
    ):
        if _client is not None and not _client.is_closed:
            # Fire-and-forget close so we don't block — old client's tasks
            # will fail, but the new client serves new requests.
            try:
                asyncio.create_task(_client.aclose())
            except RuntimeError:
                # No running loop; let GC handle it.
                pass
        _client = _build_client()
        _client_created_at = now
    return _client


async def reset_http_client():
    """Force-recycle the client. Called when we detect a stuck pool."""
    global _client, _client_created_at
    old = _client
    _client = None
    _client_created_at = 0.0
    if old is not None and not old.is_closed:
        try:
            await asyncio.wait_for(old.aclose(), timeout=2.0)
        except (asyncio.TimeoutError, Exception):
            logger.warning("HTTP client did not close cleanly; abandoning.")
    logger.info("HTTP client pool reset.")


_consecutive_timeouts = 0
_CONSECUTIVE_TIMEOUT_THRESHOLD = 3


async def _fetch_fresh(url: str, timeout: float) -> httpx.Response:
    """One-shot fetch using a brand-new client.
    Last-resort fallback for situations where the singleton pool
    is unrecoverably wedged (silently dropped connections, half-open
    sockets, etc). Slower per-call (~50ms TLS handshake), but bypasses
    any shared state.
    """
    async with httpx.AsyncClient(
        headers=headers,
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
    ) as fresh:
        return await fresh.get(url)


async def fetch_with_retries(
    url: str,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: int | float | httpx.Timeout | None = None,
    max_retries: int = DEFAULT_RETRIES,
    request_delay: float = DEFAULT_REQUEST_DELAY,
) -> httpx.Response:
    """Fetch a URL with bounded retries for transient upstream failures.

    If a request times out, immediately try once more with a fresh,
    unshared client — bypassing any zombie connections in the pool.
    """
    global _consecutive_timeouts
    client = client or get_http_client()
    retries = max(1, max_retries)
    last_response: httpx.Response | None = None
    effective_timeout = timeout if timeout is not None else 10.0
    if isinstance(effective_timeout, httpx.Timeout):
        effective_timeout = float(getattr(effective_timeout, "read", 10.0) or 10.0)

    for attempt in range(1, retries + 1):
        try:
            response = await client.get(url, timeout=timeout)
            _consecutive_timeouts = 0
        except httpx.TimeoutException:
            _consecutive_timeouts += 1
            # Last-ditch: try once with a brand-new client. If THAT also
            # times out, the upstream is genuinely unreachable.
            try:
                logger.warning(
                    "Pool fetch timed out for %s — trying fresh client.", url
                )
                fresh_resp = await _fetch_fresh(url, float(effective_timeout))
                _consecutive_timeouts = 0
                if _consecutive_timeouts >= _CONSECUTIVE_TIMEOUT_THRESHOLD:
                    await reset_http_client()
                    _consecutive_timeouts = 0
                return fresh_resp
            except Exception:
                # Both pool and fresh client timed out — pool may be fine,
                # the upstream is just down. Recycle the pool anyway and
                # retry per the loop's normal retry budget.
                if _consecutive_timeouts >= _CONSECUTIVE_TIMEOUT_THRESHOLD:
                    await reset_http_client()
                    _consecutive_timeouts = 0
                    client = get_http_client()
            if attempt >= retries:
                raise
            await asyncio.sleep(request_delay * (2 ** (attempt - 1)))
            continue
        except httpx.RequestError as exc:
            if attempt >= retries:
                raise
            logger.warning(
                "Retrying %s after request error on attempt %d/%d: %s",
                url, attempt, retries, exc,
            )
            await asyncio.sleep(request_delay * (2 ** (attempt - 1)))
            continue

        last_response = response
        if response.status_code not in RETRYABLE_STATUS_CODES or attempt >= retries:
            return response

        logger.warning(
            "Retrying %s after upstream status %d on attempt %d/%d",
            url, response.status_code, attempt, retries,
        )
        await asyncio.sleep(request_delay * (2 ** (attempt - 1)))

    if last_response is not None:
        return last_response
    raise RuntimeError(f"Failed to fetch {url} without producing a response")


async def close_http_client():
    """Close the HTTP client. Call during app shutdown."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None
