import asyncio
import logging

from selectolax.parser import HTMLParser

from utils.http_client import fetch_with_retries, get_http_client
from utils.constants import VLR_STATS_URL, CACHE_TTL_STATS
from utils.cache_manager import cache_manager
from utils.error_handling import handle_scraper_errors, validate_region, validate_timespan
from utils.html_parsers import extract_text_content

logger = logging.getLogger(__name__)

# VCT circuits: region= alone with event_group_id=all returns global garbage.
# Must pair a specific event_group_id with region= to get proper regional data.
VCT_REGIONS = {"na", "eu", "ap"}

# VLR.gg stats URL uses different region codes than our API keys.
# "la" (generic Latin America) is not a valid VLR.gg region — split into las+lan.
VLR_REGION_MAP = {
    "la-s": "las",
    "la-n": "lan",
    "col": "cg",
}

# Regions that need two requests (las + lan) to cover all of Latin America
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
    Parse the VLR.gg stats page dropdown once to discover current event_group_ids.
    Options are listed newest-first so first match = current season.
    Returns ids for: vct, challengers, gc.
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
                logger.info("GC event_group_id=%s (%s)", val, text)
            elif "challengers league" in text and groups["challengers"] == "all":
                groups["challengers"] = val
                logger.info("Challengers event_group_id=%s (%s)", val, text)
            elif (
                "valorant champions tour" in text
                and "game changer" not in text
                and "off//season" not in text
                and "partner series" not in text
                and groups["vct"] == "all"
            ):
                groups["vct"] = val
                logger.info("VCT event_group_id=%s (%s)", val, text)
            if all(v != "all" for v in groups.values()):
                break
    return groups


async def _fetch_rows(url: str, client) -> list:
    """Fetch a stats page URL and return parsed player rows."""
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


@handle_scraper_errors
async def vlr_stats(region_key: str, timespan: str):
    async def build():
        validate_region(region_key)
        validate_timespan(timespan)

        client = get_http_client()
        ts = "all" if timespan.lower() == "all" else f"{timespan}d"

        if region_key == "gc":
            groups = await _discover_event_groups(client)
            url = (
                f"{VLR_STATS_URL}/?event_group_id={groups['gc']}&event_id=all"
                f"&region=gc&country=all&min_rounds=50"
                f"&min_rating=1550&agent=all&map_id=all&timespan={ts}"
            )
            rows = await _fetch_rows(url, client)

        elif region_key in VCT_REGIONS:
            # Fetch VCT (top circuit) + Challengers (broader pool) in parallel,
            # then merge — deduplicating by player_id so each player appears once.
            groups = await _discover_event_groups(client)
            vct_url = (
                f"{VLR_STATS_URL}/?event_group_id={groups['vct']}&event_id=all"
                f"&region={region_key}&country=all&min_rounds=0"
                f"&min_rating=1550&agent=all&map_id=all&timespan={ts}"
            )
            chal_url = (
                f"{VLR_STATS_URL}/?event_group_id={groups['challengers']}&event_id=all"
                f"&region={region_key}&country=all&min_rounds=0"
                f"&min_rating=1550&agent=all&map_id=all&timespan={ts}"
            )
            vct_rows, chal_rows = await asyncio.gather(
                _fetch_rows(vct_url, client),
                _fetch_rows(chal_url, client),
            )
            # VCT entries take priority; Challengers fills in the rest
            seen_ids = {r["player_id"] for r in vct_rows if r["player_id"]}
            seen_names = {r["player"] for r in vct_rows}
            extras = [
                r for r in chal_rows
                if (r["player_id"] and r["player_id"] not in seen_ids)
                or (not r["player_id"] and r["player"] not in seen_names)
            ]
            rows = vct_rows + extras

        elif region_key in LA_COMBINED:
            # "la" is not a valid VLR.gg region — fetch las + lan in parallel and merge
            def _la_url(r):
                return (
                    f"{VLR_STATS_URL}/?event_group_id=all&event_id=all"
                    f"&region={r}&country=all&min_rounds=50"
                    f"&min_rating=1550&agent=all&map_id=all&timespan={ts}"
                )
            las_rows, lan_rows = await asyncio.gather(
                _fetch_rows(_la_url("las"), client),
                _fetch_rows(_la_url("lan"), client),
            )
            seen_ids = {r["player_id"] for r in las_rows if r["player_id"]}
            seen_names = {r["player"] for r in las_rows}
            extras = [
                r for r in lan_rows
                if (r["player_id"] and r["player_id"] not in seen_ids)
                or (not r["player_id"] and r["player"] not in seen_names)
            ]
            rows = las_rows + extras

        else:
            # Remaining regional circuits (kr, jp, br, la-s, la-n, oce, mn, col, cn)
            # Map region key to VLR.gg's actual param value where they differ
            vlr_region = VLR_REGION_MAP.get(region_key, region_key)
            url = (
                f"{VLR_STATS_URL}/?event_group_id=all&event_id=all"
                f"&region={vlr_region}&country=all&min_rounds=50"
                f"&min_rating=1550&agent=all&map_id=all&timespan={ts}"
            )
            rows = await _fetch_rows(url, client)

        data = {"data": {"status": 200, "segments": rows}}
        return data

    return await cache_manager.get_or_create_async(
        CACHE_TTL_STATS, build, "stats", region_key, timespan
    )
