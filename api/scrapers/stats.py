import asyncio
import logging
import time

from selectolax.parser import HTMLParser

from utils.http_client import fetch_with_retries, get_http_client
from utils.constants import VLR_STATS_URL, CACHE_TTL_STATS
from utils.cache_manager import cache_manager
from utils.error_handling import handle_scraper_errors, validate_region, validate_timespan
from utils.html_parsers import extract_text_content

# ── Background job tracking ───────────────────────────────────────────────────
_building: set[str] = set()
_failed_builds: dict[str, float] = {}  # job_key → unix timestamp of last failure
_FAILURE_BACKOFF = 60  # seconds before we retry after a failed build
_BUILD_HARD_TIMEOUT = 90  # seconds — kill a build that runs longer than this

def _job_key(region_key: str, timespan: str) -> str:
    return f"stats:{region_key}:{timespan}"

def get_cached_stats(region_key: str, timespan: str):
    """Return cached stats payload or None."""
    return cache_manager.get(CACHE_TTL_STATS, "stats", region_key, timespan)

def is_building(region_key: str, timespan: str) -> bool:
    """Return True if a background scrape is in progress."""
    return _job_key(region_key, timespan) in _building

def recently_failed(region_key: str, timespan: str) -> bool:
    """Return True if a recent build failed (within backoff window)."""
    key = _job_key(region_key, timespan)
    ts = _failed_builds.get(key)
    return bool(ts and (time.time() - ts) < _FAILURE_BACKOFF)

async def _background_build(region_key: str, timespan: str):
    key = _job_key(region_key, timespan)
    try:
        # Hard timeout so a single bad scrape can't hang for minutes
        result = await asyncio.wait_for(
            vlr_stats(region_key, timespan),
            timeout=_BUILD_HARD_TIMEOUT,
        )
        segments = (result or {}).get("data", {}).get("segments", [])
        # Empty results are LEGITIMATE (e.g. GC has no recent tournaments,
        # niche timespan with no matches) — they're cached and the frontend
        # will surface them as "no players found". Only real exceptions
        # / timeouts count as failures.
        _failed_builds.pop(key, None)
        logger.info("Background stats build complete: %s (%d players)", key, len(segments))
    except asyncio.TimeoutError:
        logger.error("Background stats build TIMED OUT after %ds: %s", _BUILD_HARD_TIMEOUT, key)
        _failed_builds[key] = time.time()
    except Exception as exc:
        logger.error("Background stats build failed for %s: %s", key, exc)
        _failed_builds[key] = time.time()
    finally:
        _building.discard(key)

def start_background_build(region_key: str, timespan: str) -> bool:
    """Fire a background scrape task. Returns False if one is already running."""
    key = _job_key(region_key, timespan)
    if key in _building:
        return False
    _building.add(key)
    asyncio.create_task(_background_build(region_key, timespan))
    logger.info("Background stats build started: %s", key)
    return True
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# Regions that map to different VLR.gg region codes
VLR_REGION_MAP = {
    "la-s": "las",
    "la-n": "lan",
}

# "la" splits into las + lan
LA_COMBINED = {"la"}

# Concurrent requests — high enough to be fast, low enough to avoid rate-limiting
_SEMAPHORE_LIMIT = 20

# Keywords that identify Game Changers events (case-insensitive). We strip
# whitespace/punctuation before checking so 'game-changers' or 'GameChangers'
# also match.
_GC_KEYWORDS = ("gamechangers", "vctgc", " gc ", " gc:")


def _is_gc_event(name: str) -> bool:
    if not name:
        return False
    raw = name.lower().strip()
    # Pad with spaces so word-boundary keywords (' gc ') can match start/end
    padded = f" {raw} "
    if any(kw in padded for kw in _GC_KEYWORDS):
        return True
    # Also catch the canonical full form regardless of separators
    flat = "".join(ch for ch in raw if ch.isalnum())
    return "gamechangers" in flat


def _cell_text(cells: list, index: int) -> str:
    if index >= len(cells):
        return ""
    return extract_text_content(cells[index])


def _parse_stats_row(item) -> dict:
    cells = item.css("td")
    player_cell = item.css_first("td.mod-player")

    player_name = extract_text_content(player_cell.css_first(".text-of")) if player_cell else ""
    org = extract_text_content(player_cell.css_first(".stats-player-country")) if player_cell else ""
    if not org:
        org = "N/A"

    # NOTE: VLR.gg's stats page only exposes the team tag (e.g. "SEN"),
    # not the full team name. The full name is available via the player
    # profile endpoint (see vlr_player), so we leave team_full empty here
    # and let the frontend resolve current team via player profile when
    # it opens the detail panel.
    team_full = ""

    player_id = ""
    if player_cell:
        player_link = player_cell.css_first("a[href^='/player/']") or player_cell.css_first("a")
        if player_link:
            href = player_link.attributes.get("href", "")
            parts = href.strip("/").split("/")
            if len(parts) >= 2 and parts[0] == "player":
                player_id = parts[1]

    agents = []
    for agent_img in item.css("td.mod-agents img"):
        src = agent_img.attributes.get("src", "")
        if not src:
            continue
        agents.append(src.split("/")[-1].split(".")[0])

    return {
        "player": player_name,
        "org": org,
        "team_full": team_full,
        "player_id": player_id,
        "agents": agents,
        "rounds_played": _cell_text(cells, 2),
        "rating": _cell_text(cells, 3),
        "average_combat_score": _cell_text(cells, 4),
        "kill_deaths": _cell_text(cells, 5),
        "kill_assists_survived_traded": _cell_text(cells, 6),
        "average_damage_per_round": _cell_text(cells, 7),
        "kills_per_round": _cell_text(cells, 8),
        "assists_per_round": _cell_text(cells, 9),
        "first_kills_per_round": _cell_text(cells, 10),
        "first_deaths_per_round": _cell_text(cells, 11),
        "headshot_percentage": _cell_text(cells, 12),
        "clutch_success_percentage": _cell_text(cells, 13),
    }


async def _discover_event_group_ids(client) -> tuple[list, list]:
    """
    Fetch the VLR.gg stats dropdown and return two lists:
      - open_ids:  event_group_ids for non-GC events
      - gc_ids:    event_group_ids for Game Changers events
    """
    # Tight timeout — if the dropdown is slow we want to fail fast,
    # not eat 90 seconds before the actual scrape can start.
    resp = await fetch_with_retries(
        f"{VLR_STATS_URL}/", client=client, timeout=10, max_retries=1
    )
    html = HTMLParser(resp.text)
    select = html.css_first("select[name='event_group_id']")
    open_ids, gc_ids = [], []
    if select:
        for option in select.css("option"):
            val = option.attributes.get("value", "all")
            if val == "all":
                continue
            name = (option.text() or "").strip()
            if _is_gc_event(name):
                gc_ids.append(val)
            else:
                open_ids.append(val)
    logger.info(
        "Discovered %d open event IDs and %d GC event IDs",
        len(open_ids), len(gc_ids)
    )
    return open_ids, gc_ids


def _stats_url(event_group_id: str, region: str, timespan: str) -> str:
    ts = "all" if timespan.lower() == "all" else f"{timespan}d"
    return (
        f"{VLR_STATS_URL}/?event_group_id={event_group_id}&event_id=all"
        f"&region={region}&country=all&min_rounds=0"
        f"&min_rating=1550&agent=all&map_id=all&timespan={ts}"
    )


async def _fetch_rows(url: str, client, semaphore: asyncio.Semaphore) -> list:
    async with semaphore:
        try:
            resp = await fetch_with_retries(url, client=client, timeout=15, max_retries=1)
            if resp.status_code != 200:
                return []
            html = HTMLParser(resp.text)
            rows = []
            for item in html.css("tbody tr"):
                parsed = _parse_stats_row(item)
                if parsed["player"]:
                    rows.append(parsed)
            return rows
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return []


def _merge(primary: list, secondary: list) -> list:
    """Merge two player lists, deduplicating by player_id then name."""
    seen_ids   = {r["player_id"] for r in primary if r["player_id"]}
    seen_names = {r["player"]    for r in primary}
    extras = [
        r for r in secondary
        if (r["player_id"] and r["player_id"] not in seen_ids)
        or (not r["player_id"] and r["player"] not in seen_names)
    ]
    return primary + extras


async def _fetch_all_for_region(region: str, timespan: str, ids: list, client) -> list:
    semaphore = asyncio.Semaphore(_SEMAPHORE_LIMIT)
    urls = [_stats_url(eid, region, timespan) for eid in ids]
    results = await asyncio.gather(*[_fetch_rows(u, client, semaphore) for u in urls])
    merged = []
    for rows in results:
        merged = _merge(merged, rows)
    return merged


async def _safe_discover(client) -> tuple[list, list]:
    """Discover event group IDs; on failure, return empty lists rather than raising."""
    try:
        return await _discover_event_group_ids(client)
    except Exception as exc:
        logger.warning("Event-group discovery failed (continuing with fallback): %s", exc)
        return [], []


async def _fetch_single_all(region: str, timespan: str, client) -> list:
    """Single-shot stats fetch using event_group_id=all. Used as a fallback when
    per-event-group scraping returns nothing or discovery fails. The region
    filter is applied server-side and is reliable for GC; for other regions
    it tends to leak across regions (which is why we normally avoid it)."""
    semaphore = asyncio.Semaphore(1)
    return await _fetch_rows(_stats_url("all", region, timespan), client, semaphore)


@handle_scraper_errors
async def vlr_stats(region_key: str, timespan: str):
    async def build():
        validate_region(region_key)
        validate_timespan(timespan)

        client = get_http_client()
        open_ids, gc_ids = await _safe_discover(client)

        if region_key == "gc":
            # 1. Try the per-event-group scrape using GC-tagged events
            rows = await _fetch_all_for_region("gc", timespan, gc_ids, client) if gc_ids else []
            # 2. If the keyword filter found nothing OR all GC events were empty,
            #    fall back to a single event_group_id=all&region=gc request.
            if not rows:
                rows = await _fetch_single_all("gc", timespan, client)
            # 3. As a last resort scan EVERY event group with region=gc — slower
            #    but guaranteed to surface any GC players present.
            if not rows and (open_ids or gc_ids):
                all_ids = open_ids + gc_ids
                rows = await _fetch_all_for_region("gc", timespan, all_ids, client)

        elif region_key in LA_COMBINED:
            if open_ids:
                las_rows, lan_rows = await asyncio.gather(
                    _fetch_all_for_region("las", timespan, open_ids, client),
                    _fetch_all_for_region("lan", timespan, open_ids, client),
                )
                rows = _merge(las_rows, lan_rows)
            else:
                # Discovery failed — single-shot fallback for both subregions
                las_rows, lan_rows = await asyncio.gather(
                    _fetch_single_all("las", timespan, client),
                    _fetch_single_all("lan", timespan, client),
                )
                rows = _merge(las_rows, lan_rows)

        else:
            vlr_region = VLR_REGION_MAP.get(region_key, region_key)
            if open_ids:
                rows = await _fetch_all_for_region(vlr_region, timespan, open_ids, client)
            else:
                rows = await _fetch_single_all(vlr_region, timespan, client)

        logger.info("vlr_stats(%s, %s): %d players", region_key, timespan, len(rows))
        return {"data": {"status": 200, "segments": rows}}

    return await cache_manager.get_or_create_async(
        CACHE_TTL_STATS, build, "stats", region_key, timespan
    )
