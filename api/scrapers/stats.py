import asyncio
import logging

from selectolax.parser import HTMLParser

from utils.http_client import fetch_with_retries, get_http_client
from utils.constants import VLR_STATS_URL, CACHE_TTL_STATS
from utils.cache_manager import cache_manager
from utils.error_handling import handle_scraper_errors, validate_region, validate_timespan
from utils.html_parsers import extract_text_content

logger = logging.getLogger(__name__)

# All these regions need a specific event_group_id paired with their region code —
# event_group_id=all silently returns global data regardless of region= value.
VCT_REGIONS = {"na", "eu", "ap"}

# These regions also need event_group_id filtering but don't have a VCT top-league
# entry — Challengers alone is sufficient.
CHAL_ONLY_REGIONS = {"la-s", "la-n", "br", "kr", "jp", "oce", "mn", "cn", "col"}

# VLR.gg stats URL region codes differ from our API keys for some regions
VLR_REGION_MAP = {
    "la-s": "las",
    "la-n": "lan",
    "col": "cg",
}

# "la" isn't a VLR.gg region — we split it into las + lan
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


async def _discover_event_groups(client) -> dict:
    """
    Parse the VLR.gg stats page dropdown to find current event_group_ids.
    Options are listed newest-first so first match = current season.
    """
    resp = await fetch_with_retries(f"{VLR_STATS_URL}/", client=client)
    html = HTMLParser(resp.text)
    select = html.css_first("select[name='event_group_id']")
    groups = {"vct": "all", "challengers": "all", "gc": "all"}
    if select:
        for option in select.css("option"):
            text = (option.text() or "").strip().lower()
            val = option.attributes.get("value", "all")
            if val == "all":
                continue
            if "game changer" in text and groups["gc"] == "all":
                groups["gc"] = val
            elif "challengers league" in text and groups["challengers"] == "all":
                groups["challengers"] = val
            elif (
                "valorant champions tour" in text
                and "game changer" not in text
                and "off//season" not in text
                and "partner series" not in text
                and groups["vct"] == "all"
            ):
                groups["vct"] = val
            if all(v != "all" for v in groups.values()):
                break
    logger.info("Discovered event groups: %s", groups)
    return groups


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
    seen_ids = {r["player_id"] for r in primary if r["player_id"]}
    seen_names = {r["player"] for r in primary}
    extras = [
        r for r in secondary
        if (r["player_id"] and r["player_id"] not in seen_ids)
        or (not r["player_id"] and r["player"] not in seen_names)
    ]
    return primary + extras


def _chal_url(eid: str, region: str, ts: str) -> str:
    return (
        f"{VLR_STATS_URL}/?event_group_id={eid}&event_id=all"
        f"&region={region}&country=all&min_rounds=0"
        f"&min_rating=1550&agent=all&map_id=all&timespan={ts}"
    )


@handle_scraper_errors
async def vlr_stats(region_key: str, timespan: str):
    async def build():
        validate_region(region_key)
        validate_timespan(timespan)

        client = get_http_client()
        ts = "all" if timespan.lower() == "all" else f"{timespan}d"
        groups = await _discover_event_groups(client)

        if region_key == "gc":
            rows = await _fetch_rows(_chal_url(groups["gc"], "gc", ts), client)

        elif region_key in VCT_REGIONS:
            # VCT top league + Challengers in parallel for full coverage
            vct_url = _chal_url(groups["vct"], region_key, ts)
            chal_url = _chal_url(groups["challengers"], region_key, ts)
            vct_rows, chal_rows = await asyncio.gather(
                _fetch_rows(vct_url, client),
                _fetch_rows(chal_url, client),
            )
            rows = _merge(vct_rows, chal_rows)

        elif region_key in CHAL_ONLY_REGIONS:
            # Challengers covers these regions — event_group_id=all doesn't filter properly
            vlr_region = VLR_REGION_MAP.get(region_key, region_key)
            rows = await _fetch_rows(_chal_url(groups["challengers"], vlr_region, ts), client)

        elif region_key in LA_COMBINED:
            # "la" = las + lan combined
            las_rows, lan_rows = await asyncio.gather(
                _fetch_rows(_chal_url(groups["challengers"], "las", ts), client),
                _fetch_rows(_chal_url(groups["challengers"], "lan", ts), client),
            )
            rows = _merge(las_rows, lan_rows)

        else:
            vlr_region = VLR_REGION_MAP.get(region_key, region_key)
            rows = await _fetch_rows(
                f"{VLR_STATS_URL}/?event_group_id=all&event_id=all"
                f"&region={vlr_region}&country=all&min_rounds=0"
                f"&min_rating=1550&agent=all&map_id=all&timespan={ts}",
                client,
            )

        return {"data": {"status": 200, "segments": rows}}

    return await cache_manager.get_or_create_async(
        CACHE_TTL_STATS, build, "stats", region_key, timespan
    )
