"""
VLR.GG player-search scraper.

Powers the tracker → VLR direct-link feature: instead of bouncing
the user to vlr.gg's search page, we hit the same URL server-side,
parse the first few results, and return their player_id + name so
the frontend can redirect straight to /player/<id>/<slug>.
"""
import logging
import re
import urllib.parse

from selectolax.parser import HTMLParser

from utils.cache_manager import cache_manager
from utils.constants import VLR_BASE_URL
from utils.error_handling import handle_scraper_errors
from utils.http_client import fetch_with_retries

logger = logging.getLogger(__name__)


# Cache search results for 24h. Player IDs are stable and the search
# response is small, so caching aggressively saves the upstream a
# round trip whenever the same name is looked up twice.
CACHE_TTL_PLAYER_SEARCH = 24 * 60 * 60

# Each result is a /search/r/player/<id>/idx redirector. VLR resolves
# these to the canonical /player/<id>/<slug> page. We only need the
# numeric id; the slug part is ignored by VLR's router so any value
# (or no slug at all) routes to the right profile.
_HREF_ID_RE = re.compile(r"/search/r/player/(\d+)/")


@handle_scraper_errors
async def search_players_by_name(name: str, limit: int = 5) -> dict:
    """Return up to `limit` player matches for the given name.

    Returns:
        {"results": [{"player_id", "name", "profile_url"}, ...]}
    """
    name = (name or "").strip()
    if not name:
        return {"results": []}
    if limit <= 0:
        limit = 5
    if limit > 20:
        limit = 20

    cache_key = f"player_search:{name.lower()}:{limit}"
    cached = cache_manager.get(CACHE_TTL_PLAYER_SEARCH, cache_key)
    if cached is not None:
        return cached

    url = f"{VLR_BASE_URL}/search/?q={urllib.parse.quote(name)}&type=players"
    try:
        resp = await fetch_with_retries(url, timeout=10.0)
    except Exception as exc:
        logger.warning("[player_search] fetch failed for %r: %s", name, exc)
        return {"results": []}
    if resp.status_code >= 400:
        return {"results": []}

    html = HTMLParser(resp.text)
    results = []
    for node in html.css("a.search-item"):
        href = node.attributes.get("href") or ""
        m = _HREF_ID_RE.search(href)
        if not m:
            continue
        pid = m.group(1)
        title_node = node.css_first(".search-item-title")
        pname = title_node.text(strip=True) if title_node else ""
        if not pname:
            continue
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", pname).strip("-").lower() or "x"
        results.append({
            "player_id": pid,
            "name": pname,
            "profile_url": f"{VLR_BASE_URL}/player/{pid}/{slug}",
        })
        if len(results) >= limit:
            break

    payload = {"results": results}
    cache_manager.set_if_cacheable(
        CACHE_TTL_PLAYER_SEARCH, payload, cache_key
    )
    return payload
