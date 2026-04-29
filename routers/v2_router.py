"""
V2 API router — standardized responses, validation, Pydantic models.
"""
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from models import V2Response
from routers.shared_handlers import (
    get_event_matches_data,
    get_events_data,
    get_health_data,
    get_match_data,
    get_match_detail_data,
    get_news_data,
    get_player_data,
    get_player_matches_data,
    get_rankings_data,
    get_stats_data,
    get_team_data,
    get_team_matches_data,
    get_team_transactions_data,
)
from api.scrapers.stats import get_cached_stats, is_building, recently_failed, start_background_build
from api.scrapers.team_logos import (
    get_cached_team_logos,
    is_building_team_logos,
    recently_failed_team_logos,
    start_background_team_logos_build,
)
from api.scrapers.player_resilient import (
    get_cached_player,
    get_cached_player_matches,
    get_cached_match_detail,
    is_building_player,
    is_building_matches,
    is_building_match_detail,
    recently_failed_player,
    recently_failed_matches,
    recently_failed_match_detail,
    start_persistent_player_fetch,
    start_persistent_matches_fetch,
    start_persistent_match_detail_fetch,
)
from utils.constants import RATE_LIMIT, MAX_MATCH_QUERY_BOUND
from utils.error_handling import (
    validate_event_query,
    validate_match_query,
    validate_match_workload,
    validate_player_timespan,
    validate_region,
    validate_timespan,
    validate_id_param,
)

router = APIRouter(prefix="/v2", tags=["v2"])
limiter = Limiter(key_func=get_remote_address)


def _wrap_v2(scraper_result: dict) -> dict:
    """Wrap scraper result into the standardized V2 response shape."""
    if "data" in scraper_result:
        inner = scraper_result["data"]
        status_code = inner.get("status") if isinstance(inner, dict) else None
        if isinstance(status_code, int) and status_code >= 400:
            detail = inner.get("error", "Upstream request failed")
            raise HTTPException(status_code=status_code, detail=detail)
        return {
            "status": "success",
            "data": inner,
        }
    return {"status": "success", "data": scraper_result}


@router.get("/news", response_model=V2Response)
@limiter.limit(RATE_LIMIT)
async def v2_news(request: Request):
    """Get the latest Valorant esports news from VLR.GG."""
    result = await get_news_data()
    return _wrap_v2(result)


@router.get("/stats")
@limiter.limit(RATE_LIMIT)
async def v2_stats(
    request: Request,
    region: str = Query(..., description="Region shortname (na, eu, ap, la, etc.)"),
    timespan: str = Query(..., description="Timespan: 30, 60, 90, or all"),
    force: bool = Query(False, description="Force a fresh scrape, ignoring cache"),
    exclude_funhaver: bool = Query(False, description="(NA only) skip MrFunhaver event groups"),
):
    """
    Get player statistics for a region and timespan.
    Returns 200 with data when cached, or 202 {"status":"building"} while scraping.
    Poll the same endpoint every few seconds until you get 200.

    Set ?exclude_funhaver=1 (NA only) to drop MrFunhaver tournament data
    from the aggregation — useful when you want strictly VCT/VCL/Open
    Challengers stats without the T2 community circuit mixed in.
    """
    # Validate params eagerly so bad input still gets 400
    validate_region(region)
    validate_timespan(timespan)

    # exclude_funhaver only meaningful for NA — silently no-op elsewhere
    excl = bool(exclude_funhaver) and region == "na"

    # Force refresh: invalidate cache then fall through to build
    if force:
        from utils.constants import CACHE_TTL_STATS
        from utils.cache_manager import cache_manager
        from api.scrapers.stats import _cache_args
        cache_manager.invalidate(CACHE_TTL_STATS, *_cache_args(region, timespan, excl))
    else:
        # Cache hit → immediate response
        cached = get_cached_stats(region, timespan, excl)
        if cached is not None:
            return _wrap_v2(cached)

    # Recent build failed → return error so client stops polling
    if recently_failed(region, timespan, excl):
        raise HTTPException(
            status_code=502,
            detail=(
                f"The last scrape for region '{region}' failed or returned "
                f"no players. Try a different timespan, or wait a minute "
                f"and click refresh."
            ),
        )

    # Already building → tell client to poll
    if is_building(region, timespan, excl):
        return JSONResponse({"status": "building"}, status_code=202)

    # Cold cache → fire background scrape, tell client to poll
    start_background_build(region, timespan, excl)
    return JSONResponse({"status": "building"}, status_code=202)


@router.get("/rankings", response_model=V2Response)
@limiter.limit(RATE_LIMIT)
async def v2_rankings(
    request: Request,
    region: str = Query(..., description="Region shortname (na, eu, ap, la, etc.)"),
):
    """
    Get team rankings for a region.

    Region shortnames: na, eu, ap, la, la-s, la-n, oce, kr, mn, gc, br, cn, jp, col
    """
    result = await get_rankings_data(region)
    return _wrap_v2(result)


@router.get("/match", response_model=V2Response)
@limiter.limit(RATE_LIMIT)
async def v2_match(
    request: Request,
    q: str = Query(..., description="Match type: upcoming, upcoming_extended, live_score, results"),
    num_pages: int = Query(1, description="Number of pages to scrape", ge=1, le=MAX_MATCH_QUERY_BOUND),
    from_page: int = Query(None, description="Starting page number (1-based)", ge=1, le=MAX_MATCH_QUERY_BOUND),
    to_page: int = Query(None, description="Ending page number (1-based, inclusive)", ge=1, le=MAX_MATCH_QUERY_BOUND),
    max_retries: int = Query(3, description="Max retry attempts per page", ge=1, le=5),
    request_delay: float = Query(1.0, description="Delay between requests (seconds)", ge=0.5, le=5.0),
    timeout: int = Query(30, description="Request timeout (seconds)", ge=10, le=120),
):
    """
    Get match data by type.

    - **upcoming**: Upcoming matches from homepage
    - **upcoming_extended**: Upcoming matches from paginated /matches page
    - **live_score**: Live match scores with detail
    - **results**: Completed match results
    """
    validate_match_query(q)

    if q in {"upcoming_extended", "results"}:
        validate_match_workload(num_pages, from_page, to_page, max_retries, timeout)

    result = await get_match_data(
        q, num_pages, from_page, to_page, max_retries, request_delay, timeout
    )

    return _wrap_v2(result)


@router.get("/events", response_model=V2Response)
@limiter.limit(RATE_LIMIT)
async def v2_events(
    request: Request,
    q: str = Query(None, description="Event type: upcoming or completed"),
    page: int = Query(1, description="Page number (completed events only)", ge=1, le=100),
):
    """
    Get Valorant events.

    - **upcoming**: Currently active or scheduled future events
    - **completed**: Historical events that have finished
    - **omit q**: Both upcoming and completed events
    """
    validate_event_query(q)

    result = await get_events_data(q, page)

    return _wrap_v2(result)


@router.get("/match/details")
@limiter.limit(RATE_LIMIT)
async def v2_match_detail(
    request: Request,
    match_id: str = Query(..., description="VLR.GG match ID"),
):
    """
    Resilient match-detail endpoint with 202/poll fallback.
    Returns 200 with full match data when cached, or 202 {status:"building"}
    while a persistent background worker fetches the upstream pages.
    """
    validate_id_param(match_id, "match_id")

    cached = get_cached_match_detail(match_id)
    if cached is not None:
        return _wrap_v2(cached)

    if recently_failed_match_detail(match_id):
        raise HTTPException(
            status_code=502,
            detail=f"Recent attempts to fetch match {match_id} failed. Try again in ~1 minute.",
        )

    if not is_building_match_detail(match_id):
        start_persistent_match_detail_fetch(match_id)

    return JSONResponse({"status": "building"}, status_code=202)


@router.get("/health/upstream")
async def v2_health_upstream(request: Request):
    """Probe VLR.gg from this server. Reports on BOTH the direct path
    (Railway → vlr.gg) and the proxy path (Railway → Cloudflare Worker
    → vlr.gg). The proxy is what the actual scrapers fall through to
    when direct is blocked, so as long as it works the user sees data
    and the frontend should show 'ok'.

    Status semantics:
      ok       — proxy path works (frontend has real data)
      degraded — direct fails but proxy works (still ok for users)
      down     — neither path works
    """
    import time as _t
    import httpx as _httpx
    import urllib.parse as _urlp
    from utils.utils import headers as _headers
    started = _t.time()

    direct_status = None
    direct_error = None
    proxy_status = None
    proxy_error = None

    timeout = _httpx.Timeout(6)
    async with _httpx.AsyncClient(headers=_headers, timeout=timeout, follow_redirects=True) as cli:
        # Direct path
        try:
            r = await cli.get("https://www.vlr.gg/")
            direct_status = r.status_code
        except Exception as exc:
            direct_error = str(exc) or type(exc).__name__

        # Proxy path — same path the actual scrapers use when direct fails
        proxy_url = (
            "https://vlrgg-proxy.emirerdem2.workers.dev/?url="
            + _urlp.quote("https://www.vlr.gg/", safe="")
        )
        try:
            r = await cli.get(proxy_url)
            proxy_status = r.status_code
        except Exception as exc:
            proxy_error = str(exc) or type(exc).__name__

    # Aggregate verdict
    direct_ok = direct_status is not None and direct_status < 500
    proxy_ok = proxy_status is not None and proxy_status < 500
    if proxy_ok:
        status = "ok" if direct_ok else "degraded"
    else:
        status = "down"

    return {
        "status": status,
        "upstream_status": direct_status,  # legacy field — kept for compat
        "direct_status": direct_status,
        "direct_error": direct_error,
        "proxy_status": proxy_status,
        "proxy_error": proxy_error,
        "elapsed_ms": int((_t.time() - started) * 1000),
    }


@router.get("/player")
@limiter.limit(RATE_LIMIT)
async def v2_player(
    request: Request,
    id: str = Query(..., description="VLR.GG player ID"),
    timespan: str = Query("90d", description="Stats timespan: 30d, 60d, 90d, or all"),
):
    """
    Resilient player profile with 202/poll fallback.

    Returns 200 with data when cached. On cache miss, returns 202
    {status: "building"} and spawns a persistent background worker
    that retries the upstream fetch every few seconds for up to
    ~90s. Frontend polls until 200.
    """
    validate_id_param(id)
    validate_player_timespan(timespan)

    cached = get_cached_player(id, timespan)
    if cached is not None:
        return _wrap_v2(cached)

    if recently_failed_player(id, timespan):
        raise HTTPException(
            status_code=502,
            detail=f"Recent attempts to fetch player {id} failed. Try again in ~1 minute.",
        )

    if not is_building_player(id, timespan):
        start_persistent_player_fetch(id, timespan)

    return JSONResponse({"status": "building"}, status_code=202)


@router.get("/player/matches")
@limiter.limit(RATE_LIMIT)
async def v2_player_matches(
    request: Request,
    id: str = Query(..., description="VLR.GG player ID"),
    page: int = Query(1, description="Page number (1-based)", ge=1, le=100),
):
    """Resilient match history with 202/poll fallback (same pattern as /player)."""
    validate_id_param(id)

    cached = get_cached_player_matches(id, page)
    if cached is not None:
        return _wrap_v2(cached)

    if recently_failed_matches(id, page):
        raise HTTPException(
            status_code=502,
            detail=f"Recent attempts to fetch matches for player {id} failed. Try again in ~1 minute.",
        )

    if not is_building_matches(id, page):
        start_persistent_matches_fetch(id, page)

    return JSONResponse({"status": "building"}, status_code=202)


@router.get("/team", response_model=V2Response)
@limiter.limit(RATE_LIMIT)
async def v2_team(
    request: Request,
    id: str = Query(..., description="VLR.GG team ID"),
):
    """
    Get team profile.

    Includes roster, rating/ranking info, event placements, and total winnings.
    """
    validate_id_param(id)
    result = await get_team_data(id)
    return _wrap_v2(result)


@router.get("/team/matches", response_model=V2Response)
@limiter.limit(RATE_LIMIT)
async def v2_team_matches(
    request: Request,
    id: str = Query(..., description="VLR.GG team ID"),
    page: int = Query(1, description="Page number (1-based)", ge=1, le=100),
):
    """Get paginated match history for a team."""
    validate_id_param(id)
    result = await get_team_matches_data(id, page)
    return _wrap_v2(result)


@router.get("/team/transactions", response_model=V2Response)
@limiter.limit(RATE_LIMIT)
async def v2_team_transactions(
    request: Request,
    id: str = Query(..., description="VLR.GG team ID"),
):
    """Get roster transaction history for a team (joins, leaves, benchings)."""
    validate_id_param(id)
    result = await get_team_transactions_data(id)
    return _wrap_v2(result)


@router.get("/events/matches", response_model=V2Response)
@limiter.limit(RATE_LIMIT)
async def v2_event_matches(
    request: Request,
    event_id: str = Query(..., description="VLR.GG event ID"),
):
    """Get match list for a specific event with scores and VOD links."""
    validate_id_param(event_id, "event_id")
    result = await get_event_matches_data(event_id)
    return _wrap_v2(result)


@router.get("/team_logos")
@limiter.limit(RATE_LIMIT)
async def v2_team_logos(
    request: Request,
    region: str = Query(..., description="Region shortname (na, eu, ap, la, etc.)"),
):
    """
    Bulk team-logo lookup for a region.

    Returns a flat list of {player_id, team_id, team_name, team_tag,
    team_logo, region} rows covering every player on a top-N ranked
    team. The frontend uses this so stats rows show a logo INSTANTLY
    without waiting for individual /v2/player profile fetches.

    Pattern: 200 with data when cached, 202 {status:"building"} while
    the rankings + per-team fan-out runs (~30-90s on cold cache).
    """
    validate_region(region)

    cached = get_cached_team_logos(region)
    if cached is not None:
        return _wrap_v2(cached)

    if recently_failed_team_logos(region):
        raise HTTPException(
            status_code=502,
            detail=(
                f"Recent team-logo build for region '{region}' failed. "
                f"Try again in ~1 minute."
            ),
        )

    if not is_building_team_logos(region):
        start_background_team_logos_build(region)

    return JSONResponse({"status": "building"}, status_code=202)


@router.get("/health", response_model=V2Response)
async def v2_health():
    """Check API health and runtime readiness."""
    result = await get_health_data()
    return {"status": "success", "data": result}


