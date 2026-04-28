import asyncio
import logging

from selectolax.parser import HTMLParser

from utils.http_client import fetch_with_retries, get_http_client
from utils.constants import VLR_STATS_URL, CACHE_TTL_STATS
from utils.cache_manager import cache_manager
from utils.error_handling import handle_scraper_errors, validate_region, validate_timespan
from utils.html_parsers import extract_text_content

logger = logging.getLogger(__name__)

# Regions that map to different VLR.gg region codes
VLR_REGION_MAP = {
    "la-s": "las",
    "la-n": "lan",
    "col":  "cg",
}

# "la" splits into las + lan
LA_COMBINED = {"la"}

# Max concurrent requests to VLR.gg to avoid rate-limiting
_SEMAPHORE_LIMIT = 12


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

    player_id = ""
    if player_cell:
        player_link = player_cell.css_first("a")
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



# Keywords that identify major competitive events worth scraping.
# Small/niche/college events are skipped to keep response times reasonable.
_MAJOR_EVENT_KEYWORDS = (
    "valorant champions tour",
    "challengers league",
    "game changers",
    "partner series",
    "off//season",
    "china evolution series",
    "project v",
    "wave oce",
    "beacon",
    "fgc valorant",
    "valorant east",
    "red bull campus",
    "riot games one",
)

_SKIP_KEYWORDS = (
    "college",
    "txg",
    "braza",
    "evc surge",
    "exitlag",
    "hyperx momentum",
    "flyquest trailblazer",
    "yfp lockdown",
    "cbvl",
    "raidiant academy",
    "norwegian",
    "insomnia",
    "a1 esports",
    "super girl gamer",
    "community gaming",
    "golden goose",
    "nerd street",
    "blast spike",
    "trinity trials",
    "mystic singularity",
    "cecc",
    "collegiate",
    "saudi eleague",
    "return of the titans",
    "nom invitationals",
    "balkan league",
)


async def _discover_all_event_group_ids(client) -> list:
    """
    Fetch the VLR.gg stats page and return event_group_ids for all
    major competitive events (skips college/niche/one-off events).
    """
    resp = await fetch_with_retries(f"{VLR_STATS_URL}/", client=client)
    html = HTMLParser(resp.text)
    select = html.css_first("select[name='event_group_id']")
    ids = []
    if select:
        for option in select.css("option"):
            val = option.attributes.get("value", "all")
            if val == "all":
                continue
            name_lower = (option.text() or "").strip().lower()
            # Skip known niche events
            if any(skip in name_lower for skip in _SKIP_KEYWORDS):
                continue
            # Include known major events
            if any(kw in name_lower for kw in _MAJOR_EVENT_KEYWORDS):
                ids.append(val)
                continue
            # Include anything else that isn't clearly niche
            ids.append(val)
    logger.info("Discovered %d major event group IDs (from dropdown)", len(ids))
    return ids


async def _fetch_rows(url: str, client, semaphore: asyncio.Semaphore) -> list:
    async with semaphore:
        try:
            resp = await fetch_with_retries(url, client=client)
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


def _stats_url(event_group_id: str, region: str, timespan: str) -> str:
    return (
        f"{VLR_STATS_URL}/?event_group_id={event_group_id}&event_id=all"
        f"&region={region}&country=all&min_rounds=0"
        f"&min_rating=1550&agent=all&map_id=all&timespan={timespan}"
    )


async def _fetch_all_for_region(region: str, timespan: str, all_ids: list, client) -> list:
    """
    Fetch stats from every event group for a single region and merge.
    Uses a semaphore to avoid hammering VLR.gg.
    """
    semaphore = asyncio.Semaphore(_SEMAPHORE_LIMIT)
    urls = [_stats_url(eid, region, timespan) for eid in all_ids]
    results = await asyncio.gather(*[_fetch_rows(u, client, semaphore) for u in urls])

    merged = []
    for rows in results:
        merged = _merge(merged, rows)
    return merged


@handle_scraper_errors
async def vlr_stats(region_key: str, timespan: str):
    async def build():
        validate_region(region_key)
        validate_timespan(timespan)

        client = get_http_client()
        ts = "all" if timespan.lower() == "all" else f"{timespan}d"

        # Get every event group ID from the dropdown
        all_ids = await _discover_all_event_group_ids(client)

        if region_key == "gc":
            # Game Changers: use region=gc across all event groups
            rows = await _fetch_all_for_region("gc", ts, all_ids, client)

        elif region_key in LA_COMBINED:
            # "la" = las + lan combined across all event groups
            las_rows, lan_rows = await asyncio.gather(
                _fetch_all_for_region("las", ts, all_ids, client),
                _fetch_all_for_region("lan", ts, all_ids, client),
            )
            rows = _merge(las_rows, lan_rows)

        else:
            vlr_region = VLR_REGION_MAP.get(region_key, region_key)
            rows = await _fetch_all_for_region(vlr_region, ts, all_ids, client)

        logger.info("vlr_stats(%s, %s): %d players", region_key, timespan, len(rows))
        return {"data": {"status": 200, "segments": rows}}

    return await cache_manager.get_or_create_async(
        CACHE_TTL_STATS, build, "stats", region_key, timespan
    )
