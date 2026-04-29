"""
Fail-safe wrapper around vlr_player and vlr_player_matches.

Problem: VLR.gg player endpoints occasionally hang from Railway's egress —
sometimes for minutes. When a single fetch times out, the data isn't cached
and every subsequent user request re-tries the slow path, compounding the
delay. This module addresses that with three concentric defenses:

1. **Persistent background workers** — when a request fails, a background
   task keeps retrying every RETRY_INTERVAL seconds for up to MAX_RETRY_TIME
   total. As soon as ONE attempt succeeds, the cache is populated and the
   foreground keeps polling instantly.

2. **202 / poll pattern** — the public endpoint never blocks waiting for a
   slow VLR.gg fetch. On cache miss, it returns 202 immediately and starts
   the background worker. Frontend polls until the cache lands.

3. **Recent-failure tracking** — if every retry attempt fails for the full
   MAX_RETRY_TIME window, the entry is marked "recently failed" and the
   endpoint returns 502 with detail for the next FAILURE_BACKOFF seconds.
   This breaks the infinite-poll loop on permanently-broken inputs (bad
   player IDs, etc).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from utils.cache_manager import cache_manager
from utils.constants import CACHE_TTL_PLAYER, CACHE_TTL_PLAYER_MATCHES

logger = logging.getLogger(__name__)


# Tuning
RETRY_INTERVAL = 4.0          # seconds between retry attempts
MAX_RETRY_TIME = 90.0          # give up after this many seconds total
FAILURE_BACKOFF = 60.0         # how long to refuse new attempts after exhaustion
JOB_HARD_TIMEOUT = 120.0       # wall-clock cap on the entire background job

# In-memory state
_building: set[str] = set()
_failed: dict[str, float] = {}  # job_key -> unix timestamp


def _job_key(*parts: Any) -> str:
    return "|".join(str(p) for p in parts)


def get_cached_player(player_id: str, timespan: str = "90d"):
    return cache_manager.get(CACHE_TTL_PLAYER, "player", player_id, timespan)


def get_cached_player_matches(player_id: str, page: int = 1):
    return cache_manager.get(CACHE_TTL_PLAYER_MATCHES, "player_matches", player_id, page)


def get_cached_match_detail(match_id: str):
    from utils.constants import CACHE_TTL_MATCH_DETAIL
    return cache_manager.get(CACHE_TTL_MATCH_DETAIL, "match_detail", match_id)


def is_building_player(player_id: str, timespan: str = "90d") -> bool:
    return _job_key("player", player_id, timespan) in _building


def is_building_matches(player_id: str, page: int = 1) -> bool:
    return _job_key("matches", player_id, page) in _building


def is_building_match_detail(match_id: str) -> bool:
    return _job_key("match_detail", match_id) in _building


def recently_failed_match_detail(match_id: str) -> bool:
    ts = _failed.get(_job_key("match_detail", match_id))
    return bool(ts and (time.time() - ts) < FAILURE_BACKOFF)


def recently_failed_player(player_id: str, timespan: str = "90d") -> bool:
    ts = _failed.get(_job_key("player", player_id, timespan))
    return bool(ts and (time.time() - ts) < FAILURE_BACKOFF)


def recently_failed_matches(player_id: str, page: int = 1) -> bool:
    ts = _failed.get(_job_key("matches", player_id, page))
    return bool(ts and (time.time() - ts) < FAILURE_BACKOFF)


async def _persistent_fetch(
    label: str,
    job_key: str,
    fetch_coro_factory,
):
    """
    Run a fetch repeatedly with RETRY_INTERVAL spacing until it
    succeeds or MAX_RETRY_TIME elapses. Caches automatically via the
    underlying scraper's get_or_create_async.
    """
    started = time.time()
    attempt = 0
    last_exc = None
    try:
        async def loop():
            nonlocal attempt, last_exc
            while time.time() - started < MAX_RETRY_TIME:
                attempt += 1
                try:
                    result = await fetch_coro_factory()
                    # If the underlying scraper raised an HTTPException, it's
                    # already caught at the FastAPI layer — but we also want
                    # to handle the case where it returned a payload with an
                    # error status. Treat HTTP-style status >=400 as failure.
                    payload = (result or {}).get("data", {}) if isinstance(result, dict) else {}
                    status = payload.get("status") if isinstance(payload, dict) else None
                    if isinstance(status, int) and status >= 400:
                        raise RuntimeError(f"upstream status {status}")
                    logger.info(
                        "%s fetch succeeded on attempt %d (%.1fs total)",
                        label, attempt, time.time() - started,
                    )
                    _failed.pop(job_key, None)
                    return result
                except Exception as exc:
                    last_exc = exc
                    elapsed = time.time() - started
                    remaining = MAX_RETRY_TIME - elapsed
                    if remaining <= 0:
                        break
                    logger.warning(
                        "%s fetch attempt %d failed (%.1fs elapsed): %s — retrying in %.1fs",
                        label, attempt, elapsed, exc, RETRY_INTERVAL,
                    )
                    await asyncio.sleep(min(RETRY_INTERVAL, max(0.1, remaining)))
            # Exhausted
            logger.error(
                "%s fetch FAILED after %d attempts (%.1fs): %s",
                label, attempt, time.time() - started, last_exc,
            )
            _failed[job_key] = time.time()
            return None

        return await asyncio.wait_for(loop(), timeout=JOB_HARD_TIMEOUT)
    finally:
        _building.discard(job_key)


def start_persistent_player_fetch(player_id: str, timespan: str = "90d") -> bool:
    """Spawn a background worker that keeps trying the player profile
    fetch until it succeeds or MAX_RETRY_TIME runs out.
    Returns False if one is already running for this key.
    """
    job_key = _job_key("player", player_id, timespan)
    if job_key in _building:
        return False
    _building.add(job_key)

    # Local import to avoid cycles
    from api.scrapers.players import vlr_player

    async def factory():
        return await vlr_player(player_id, timespan)

    asyncio.create_task(_persistent_fetch(f"player[{player_id} ts={timespan}]", job_key, factory))
    return True


def start_persistent_matches_fetch(player_id: str, page: int = 1) -> bool:
    job_key = _job_key("matches", player_id, page)
    if job_key in _building:
        return False
    _building.add(job_key)

    from api.scrapers.players import vlr_player_matches

    async def factory():
        return await vlr_player_matches(player_id, page)

    asyncio.create_task(_persistent_fetch(f"matches[{player_id} p={page}]", job_key, factory))
    return True


def start_persistent_match_detail_fetch(match_id: str) -> bool:
    job_key = _job_key("match_detail", match_id)
    if job_key in _building:
        return False
    _building.add(job_key)

    from api.scrapers.match_detail import vlr_match_detail

    async def factory():
        return await vlr_match_detail(match_id)

    asyncio.create_task(_persistent_fetch(f"match_detail[{match_id}]", job_key, factory))
    return True
