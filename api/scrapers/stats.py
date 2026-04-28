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
}

# "la" splits into las + lan
LA_COMBINED = {"la"}


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


def _stats_url(region: str, timespan: str) -> str:
    ts = "all" if timespan.lower() == "all" else f"{timespan}d"
    return (
        f"{VLR_STATS_URL}/?event_group_id=all&event_id=all"
        f"&region={region}&country=all&min_rounds=0"
        f"&min_rating=1550&agent=all&map_id=all&timespan={ts}"
    )


async def _fetch_rows(url: str, client) -> list:
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


@handle_scraper_errors
async def vlr_stats(region_key: str, timespan: str):
    async def build():
        validate_region(region_key)
        validate_timespan(timespan)

        client = get_http_client()

        if region_key == "gc":
            rows = await _fetch_rows(_stats_url("gc", timespan), client)

        elif region_key in LA_COMBINED:
            # "la" = las + lan combined
            las_rows, lan_rows = await asyncio.gather(
                _fetch_rows(_stats_url("las", timespan), client),
                _fetch_rows(_stats_url("lan", timespan), client),
            )
            rows = _merge(las_rows, lan_rows)

        else:
            vlr_region = VLR_REGION_MAP.get(region_key, region_key)
            rows = await _fetch_rows(_stats_url(vlr_region, timespan), client)

        logger.info("vlr_stats(%s, %s): %d players", region_key, timespan, len(rows))
        return {"data": {"status": 200, "segments": rows}}

    return await cache_manager.get_or_create_async(
        CACHE_TTL_STATS, build, "stats", region_key, timespan
    )
