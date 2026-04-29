"""
Bulk team-logo lookup scraper.

Solves the long-standing "rows show the Valorant V placeholder for 2-3
minutes after page load" problem on the frontend. The previous design
required a /v2/player profile fetch *per player* before we knew their
team logo, because the stats page only exposes the 3-letter org tag
(which isn't unique within a region — '100T' = both 100 Thieves and
FlyQuest RED in NA).

This scraper builds a flat (player_id -> team_logo) index from VLR's
authoritative team rosters:

  1. Hit /rankings/{region} to enumerate the top-N teams (each carries a
     unique numeric team_id and the canonical team logo URL).
  2. Fan out to /team/{team_id} for each, with bounded concurrency, to
     pull the active roster (player_ids).
  3. Flatten into per-player rows so the frontend can resolve a logo
     instantly from a stats row's player_id alone — no per-player
     profile fetch needed.

The result is cached in-process for 6h, and the frontend mirrors it to
Supabase so other tabs/visitors get an instant Postgres read.
"""
import asyncio
import logging
import re

from api.scrapers.rankings import vlr_rankings
from api.scrapers.teams import vlr_team
from utils.cache_manager import cache_manager
from utils.error_handling import handle_scraper_errors, validate_region

logger = logging.getLogger(__name__)


# 6 hours — rosters don't churn often, and the frontend force-refresh
# button can bust this if needed.
CACHE_TTL_TEAM_LOGOS = 6 * 60 * 60

# Cap per-region team count. NA rankings has ~200 teams but the top 60
# covers the realistic pro/semi-pro player base — ranks beyond that are
# largely dormant orgs whose roster pages either 404 or eat the timeout
# without contributing logos for any visible players. Empirically, 120
# teams blew through the 240s hard timeout with most of the time spent
# on dead-org requests retrying through the proxy chain.
MAX_TEAMS_PER_REGION = 60

# Concurrency for the /team/{id} fan-out. VLR's proxy chain (CF Worker
# + 4 public fallbacks) starts dropping requests above 4-5 concurrent
# /team page fetches. Lower concurrency completes the batch faster
# overall because we avoid the retry storm.
TEAM_FETCH_CONCURRENCY = 4


def _normalize_logo_url(raw: str) -> str:
    """Coerce VLR logo URLs to a canonical https form, drop placeholders."""
    if not raw:
        return ""
    # VLR sometimes returns its own 'no logo' placeholder — treat as empty.
    if "tmp/vlr.png" in raw:
        return ""
    if raw.startswith("//"):
        return "https:" + raw
    return raw


async def _fetch_team_safe(team_id: str) -> dict | None:
    """Wrap vlr_team in try/except so one failure doesn't kill the batch."""
    try:
        result = await vlr_team(team_id)
    except Exception as exc:
        logger.warning("team_logos: vlr_team(%s) failed: %s", team_id, exc)
        return None

    segments = (result or {}).get("data", {}).get("segments") or []
    if not segments:
        return None
    return segments[0]


@handle_scraper_errors
async def vlr_team_logos(region_key: str) -> dict:
    """
    Build a flat (player_id -> team identity + logo) map for a region.

    Returns the standardized response shape:
      {"data": {"status": 200, "segments": [{player_id, team_id, ...}, ...]}}

    Cached 6h per region.
    """
    region_name = validate_region(region_key)

    async def build():
        # Step 1: rankings → team_id list
        rankings_resp = await vlr_rankings(region_key)
        rankings = (rankings_resp or {}).get("data", {}).get("segments") or []
        team_seeds: list[dict] = []
        seen_ids: set[str] = set()
        for entry in rankings[:MAX_TEAMS_PER_REGION]:
            tid = entry.get("team_id") or ""
            if not tid or tid in seen_ids:
                continue
            seen_ids.add(tid)
            team_seeds.append({
                "team_id": tid,
                "rankings_logo": _normalize_logo_url(entry.get("logo", "")),
                "rankings_name": entry.get("team", "") or "",
            })

        if not team_seeds:
            logger.warning(
                "team_logos: region %s yielded zero ranked teams", region_key
            )
            return {"data": {"status": 200, "segments": []}}

        # Step 2: fan-out /team/{id} with bounded concurrency
        sem = asyncio.Semaphore(TEAM_FETCH_CONCURRENCY)

        async def bounded_fetch(t: dict):
            async with sem:
                return t, await _fetch_team_safe(t["team_id"])

        logger.info(
            "team_logos: fanning out %d /team requests for region %s (concurrency=%d)",
            len(team_seeds), region_key, TEAM_FETCH_CONCURRENCY,
        )
        results = await asyncio.gather(*[bounded_fetch(t) for t in team_seeds])
        ok_count = sum(1 for _, td in results if td)
        logger.info(
            "team_logos: %d/%d /team fetches succeeded for region %s",
            ok_count, len(team_seeds), region_key,
        )

        # Step 3: flatten roster into per-player rows
        rows: list[dict] = []
        seen_players: set[str] = set()
        for seed, team_data in results:
            if not team_data:
                # Fall back to rankings-only entry — gives us a logo for
                # the team itself but no roster mapping. Useful only for
                # the calendar; skip for the per-player table.
                continue
            tid = seed["team_id"]
            name = team_data.get("name") or seed.get("rankings_name", "")
            tag = team_data.get("tag") or ""
            logo = _normalize_logo_url(team_data.get("logo") or seed.get("rankings_logo", ""))

            for player in team_data.get("roster") or []:
                if player.get("is_staff"):
                    continue
                pid = player.get("id") or ""
                if not pid or pid in seen_players:
                    continue
                # When a player appears on multiple teams' rosters (rare
                # but happens during transitions), we keep the FIRST
                # occurrence — which is the higher-ranked team because
                # we walk seeds in rankings order.
                seen_players.add(pid)
                rows.append({
                    "player_id": pid,
                    "team_id": tid,
                    "team_name": name,
                    "team_tag": tag,
                    "team_logo": logo,
                    "region": region_name,
                })

        return {"data": {"status": 200, "segments": rows}}

    return await cache_manager.get_or_create_async(
        CACHE_TTL_TEAM_LOGOS, build, "team_logos", region_key
    )


# ---------------------------------------------------------------------------
# Sync helpers used by the v2 router (mirrors stats / player_resilient pattern)
# ---------------------------------------------------------------------------

import time as _time

_BUILDING: set[str] = set()
_TEAM_LOGOS_FAILED_AT: dict[str, float] = {}
_FAIL_COOLDOWN_S = 60
# Hard timeout for a full region build. ~120 teams × concurrency 8 → 15
# batches; each /team/{id} fetch typically completes in 2-5s through
# the proxy chain. 240s gives ~2x headroom over the realistic worst case.
_BUILD_HARD_TIMEOUT = 240


def get_cached_team_logos(region_key: str):
    """Return cached payload or None."""
    return cache_manager.get(CACHE_TTL_TEAM_LOGOS, "team_logos", region_key)


def is_building_team_logos(region_key: str) -> bool:
    return region_key in _BUILDING


def recently_failed_team_logos(region_key: str) -> bool:
    ts = _TEAM_LOGOS_FAILED_AT.get(region_key, 0)
    return (_time.time() - ts) < _FAIL_COOLDOWN_S


async def _background_build(region_key: str) -> None:
    try:
        result = await asyncio.wait_for(
            vlr_team_logos(region_key),
            timeout=_BUILD_HARD_TIMEOUT,
        )
        segments = (result or {}).get("data", {}).get("segments") or []
        if not segments:
            logger.warning(
                "team_logos: background build for %s returned 0 rows",
                region_key,
            )
            _TEAM_LOGOS_FAILED_AT[region_key] = _time.time()
        else:
            _TEAM_LOGOS_FAILED_AT.pop(region_key, None)
            logger.info(
                "team_logos: background build complete for %s (%d rows)",
                region_key, len(segments),
            )
    except asyncio.TimeoutError:
        logger.error(
            "team_logos: background build TIMED OUT after %ds for %s",
            _BUILD_HARD_TIMEOUT, region_key,
        )
        _TEAM_LOGOS_FAILED_AT[region_key] = _time.time()
    except Exception as exc:
        logger.error(
            "team_logos: background build failed for %s: %s",
            region_key, exc,
        )
        _TEAM_LOGOS_FAILED_AT[region_key] = _time.time()
    finally:
        _BUILDING.discard(region_key)


def start_background_team_logos_build(region_key: str) -> bool:
    """Kick off a background build the first time a region is requested."""
    if region_key in _BUILDING:
        return False
    _BUILDING.add(region_key)
    asyncio.create_task(_background_build(region_key))
    logger.info("team_logos: background build started for %s", region_key)
    return True
