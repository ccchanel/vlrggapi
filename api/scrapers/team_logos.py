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

# Cap per-region team count. NA rankings has ~200 teams but most active
# pro/semi-pro players are on the top ~120. Beyond that we hit dead orgs
# that just slow the build down without adding logo coverage.
MAX_TEAMS_PER_REGION = 120

# Concurrency for the /team/{id} fan-out. Higher than 8 starts hitting
# VLR's anti-bot more aggressively; we get ~95% success at 8 with the
# proxy fallthrough handling the rest.
TEAM_FETCH_CONCURRENCY = 8


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

        results = await asyncio.gather(*[bounded_fetch(t) for t in team_seeds])

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

_TEAM_LOGOS_BUILDS: dict[str, asyncio.Task] = {}
_TEAM_LOGOS_FAILED_AT: dict[str, float] = {}
_FAIL_COOLDOWN_S = 60


def get_cached_team_logos(region_key: str):
    """Return cached payload or None."""
    return cache_manager.get(CACHE_TTL_TEAM_LOGOS, "team_logos", region_key)


def is_building_team_logos(region_key: str) -> bool:
    task = _TEAM_LOGOS_BUILDS.get(region_key)
    return task is not None and not task.done()


def recently_failed_team_logos(region_key: str) -> bool:
    import time
    ts = _TEAM_LOGOS_FAILED_AT.get(region_key, 0)
    return (time.time() - ts) < _FAIL_COOLDOWN_S


def start_background_team_logos_build(region_key: str) -> None:
    """Kick off a background build the first time a region is requested."""
    if is_building_team_logos(region_key):
        return

    async def _runner():
        import time
        try:
            await vlr_team_logos(region_key)
        except Exception as exc:
            logger.warning(
                "team_logos: background build for %s failed: %s",
                region_key, exc,
            )
            _TEAM_LOGOS_FAILED_AT[region_key] = time.time()
        finally:
            _TEAM_LOGOS_BUILDS.pop(region_key, None)

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    _TEAM_LOGOS_BUILDS[region_key] = loop.create_task(_runner())
