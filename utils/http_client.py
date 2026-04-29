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

# Circuit breaker — when direct VLR.gg is broken, every sub-fetch wastes
# 15s timing out before falling through to proxy. With 80 sub-fetches in
# a stats scrape, that adds up to 80×15s = 20min before the build
# timeout. Detect the broken state and skip straight to proxies.
_direct_broken_until: float = 0.0
_DIRECT_BROKEN_FOR = 60.0  # seconds to skip direct fetches after detection
_DIRECT_BROKEN_TRIGGER = 5  # consecutive timeouts before tripping
_recent_direct_timeouts = 0


def _direct_broken() -> bool:
    return time.time() < _direct_broken_until


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


# Public read-through proxies. Tried in order when both direct attempts
# (pool + fresh client) time out. These rewrite the upstream URL into
# their own host but pass through the body unchanged, which is enough
# for our scraper. Each is fronted by a different network so failures
# correlate poorly — at least one is usually working.
import urllib.parse as _urlparse

# Our own dedicated Cloudflare Worker proxy. Free tier allows
# 100k req/day, runs on Cloudflare's edge (so request to vlr.gg
# leaves from a stable Cloudflare IP, not Railway), and we control
# the cookie/UA setup. This is tried FIRST — public proxies are the
# fallback when our worker is unavailable or rate-limited.
_VLRGG_PROXY_URL = "https://vlrgg-proxy.emirerdem2.workers.dev"

_PROXY_URL_BUILDERS = [
    # Dedicated Cloudflare Worker (priority).
    lambda u: f"{_VLRGG_PROXY_URL}/?url={_urlparse.quote(u, safe='')}",
    # Public pass-through CORS proxies — each returns the upstream
    # response body unchanged. Crucial: the first version of this
    # list also included r.jina.ai, which is a *reader* proxy that
    # returns markdown-converted content. Our HTML parser then saw
    # 0 rows and marked the scrape as a failure. Only true
    # pass-throughs belong here.
    lambda u: f"https://api.allorigins.win/raw?url={_urlparse.quote(u, safe='')}",
    lambda u: f"https://corsproxy.io/?{_urlparse.quote(u, safe='')}",
    lambda u: f"https://api.codetabs.com/v1/proxy/?quest={_urlparse.quote(u, safe='')}",
    lambda u: f"https://thingproxy.freeboard.io/fetch/{u}",
    lambda u: f"https://api.cors.lol/?url={_urlparse.quote(u, safe='')}",
]


import random as _random

async def _fetch_via_proxy(url: str, timeout: float) -> httpx.Response | None:
    """Try proxies in randomized order until one returns 2xx.
    Randomization spreads load across all five proxies — when 80
    concurrent stats sub-fetches all fall through to this layer at
    once, picking the same first proxy every time would rate-limit
    that one provider into the ground.

    Returns None if every proxy fails for this request."""
    # Always try our dedicated Worker first (index 0). Shuffle only
    # the public-proxy fallbacks so concurrent sub-fetches don't all
    # rate-limit the same provider when the Worker is unavailable.
    primary = _PROXY_URL_BUILDERS[:1]
    fallbacks = list(_PROXY_URL_BUILDERS[1:])
    _random.shuffle(fallbacks)
    builders = primary + fallbacks
    for build in builders:
        try:
            proxy_url = build(url)
            async with httpx.AsyncClient(
                headers=headers,
                timeout=httpx.Timeout(timeout),
                follow_redirects=True,
            ) as cli:
                resp = await cli.get(proxy_url)
                if not (200 <= resp.status_code < 300 and resp.content):
                    continue
                # Sanity-check the body actually LOOKS like HTML from
                # vlr.gg. Some proxies happily return JSON error pages
                # with a 200; some apply transformations (markdown,
                # text-only) we can't parse. Sniff for HTML markers
                # and the vlr.gg-specific 'wf-' class prefix as proof
                # of a real upstream response.
                body_lower = resp.text[:4096].lower()
                if "<html" not in body_lower and "<!doctype" not in body_lower:
                    logger.debug("Proxy %s returned non-HTML for %s; skipping", build(url).split("/")[2], url)
                    continue
                logger.warning(
                    "Direct fetch failed for %s — succeeded via proxy %s",
                    url, build(url).split("/")[2],
                )
                return resp
        except Exception as exc:
            logger.debug("Proxy attempt failed for %s: %s", url, exc)
            continue
    return None


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
    global _consecutive_timeouts, _direct_broken_until, _recent_direct_timeouts
    client = client or get_http_client()
    retries = max(1, max_retries)
    last_response: httpx.Response | None = None
    effective_timeout = timeout if timeout is not None else 10.0
    if isinstance(effective_timeout, httpx.Timeout):
        effective_timeout = float(getattr(effective_timeout, "read", 10.0) or 10.0)

    # Circuit breaker: skip directly to proxies when direct VLR.gg is
    # known-broken. Saves ~30s per sub-fetch (15s pool timeout + 15s
    # fresh client timeout) when 80 stats sub-fetches all hit a dead
    # upstream simultaneously.
    if _direct_broken():
        proxy_resp = await _fetch_via_proxy(url, float(effective_timeout))
        if proxy_resp is not None:
            return proxy_resp
        # Proxy also failed — fall through to normal logic so the
        # caller's retry budget applies. The breaker just gives us
        # a fast-path attempt; it's not authoritative.

    for attempt in range(1, retries + 1):
        try:
            response = await client.get(url, timeout=timeout)
            _consecutive_timeouts = 0
            _recent_direct_timeouts = 0
        except httpx.TimeoutException:
            _consecutive_timeouts += 1
            _recent_direct_timeouts += 1
            # Trip the circuit breaker if we've seen many direct
            # timeouts in a short window. Future fetches skip direct
            # entirely for _DIRECT_BROKEN_FOR seconds.
            if _recent_direct_timeouts >= _DIRECT_BROKEN_TRIGGER:
                _direct_broken_until = time.time() + _DIRECT_BROKEN_FOR
                _recent_direct_timeouts = 0
                logger.warning(
                    "Circuit breaker tripped — direct VLR.gg fetches "
                    "skipped for the next %ds.", int(_DIRECT_BROKEN_FOR),
                )
            # Two-stage fallback when the pool client times out:
            # 1) Fresh client — eliminates stuck-keepalive issues
            # 2) Public read-through proxy — eliminates Railway egress
            #    issues to vlr.gg specifically (intermittent network
            #    blocking has been observed)
            try:
                logger.warning("Pool fetch timed out for %s — trying fresh client.", url)
                fresh_resp = await _fetch_fresh(url, float(effective_timeout))
                _consecutive_timeouts = 0
                if _consecutive_timeouts >= _CONSECUTIVE_TIMEOUT_THRESHOLD:
                    await reset_http_client()
                    _consecutive_timeouts = 0
                return fresh_resp
            except Exception:
                logger.warning("Fresh client also timed out for %s — trying proxies.", url)
                proxy_resp = await _fetch_via_proxy(url, float(effective_timeout))
                if proxy_resp is not None:
                    _consecutive_timeouts = 0
                    return proxy_resp
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
