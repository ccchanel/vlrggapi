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
_BUILD_HARD_TIMEOUT = 240  # seconds — 25 event groups × up to 3 retries
                            # each, semaphore=8 → worst case ~10 batches
                            # × ~12s = 120s. 240s gives 2× headroom.

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

# Major regions where 0 players returned almost certainly means the scrape
# failed silently (empty cache poisoning) rather than a legitimate "no data".
# For these, an empty result is treated as a failure and NOT cached.
_MAJOR_REGIONS = {"na", "eu", "ap", "kr", "br", "cn", "jp"}

async def _background_build(region_key: str, timespan: str):
    key = _job_key(region_key, timespan)
    try:
        # Hard timeout so a single bad scrape can't hang for minutes
        result = await asyncio.wait_for(
            vlr_stats(region_key, timespan),
            timeout=_BUILD_HARD_TIMEOUT,
        )
        segments = (result or {}).get("data", {}).get("segments", [])
        # For MAJOR regions, 0 players is almost certainly a failure
        # (network blip, all sub-fetches timed out, etc). Don't let that
        # poison the cache for 4 hours. Force-invalidate and mark failed.
        if not segments and region_key in _MAJOR_REGIONS:
            logger.warning("Major region %s returned 0 players — invalidating cache", key)
            try:
                cache_manager.invalidate(CACHE_TTL_STATS, "stats", region_key, timespan)
            except Exception:
                pass
            _failed_builds[key] = time.time()
        else:
            # Empty results from niche regions (gc, la-n, la-s) ARE legitimate
            _failed_builds.pop(key, None)
            logger.info("Background stats build complete: %s (%d players)", key, len(segments))
    except asyncio.TimeoutError:
        logger.error("Background stats build TIMED OUT after %ds: %s", _BUILD_HARD_TIMEOUT, key)
        # Kill the inflight task — wait_for cancelling US doesn't kill
        # the underlying coalesced producer; we have to cancel it ourselves
        # or new requests will coalesce into the zombie task forever.
        try:
            cache_manager.invalidate(CACHE_TTL_STATS, "stats", region_key, timespan)
        except Exception:
            pass
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

# Concurrent requests — gentler concurrency keeps VLR.gg from
# rate-limiting us during a full-region scrape. Empirically, 20
# triggered occasional empty responses on subsequent regions; 8
# completes consistently.
_SEMAPHORE_LIMIT = 8

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


def _is_funhaver_event(name: str) -> bool:
    """MrFunhaver-branded NA tournament series. These often have older
    event_group_ids than the latest VCT/VCL/Ascension events, so they
    get cut off by the top-N cap in discovery — even though they're
    still actively producing recent matches in our timespan window.
    Detect them so we can pin them in alongside the top-N."""
    if not name:
        return False
    n = name.lower()
    return "funhaver" in n or "fun haver" in n or "mrfunhaver" in n


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

    # Country flag — VLR uses <i class="flag mod-us"> (or mod-gb, mod-kr, etc.)
    # Extract the 2-letter ISO country code from that class so the frontend
    # can render a flag emoji or icon under the player name.
    country = ""
    if player_cell:
        flag_node = player_cell.css_first("i.flag")
        if flag_node:
            cls = flag_node.attributes.get("class", "") or ""
            for token in cls.split():
                if token.startswith("mod-") and len(token) == 6:
                    country = token[4:].lower()
                    break

    return {
        "player": player_name,
        "org": org,
        "team_full": team_full,
        "player_id": player_id,
        "country": country,
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


# Maximum event groups to scrape per region. The dropdown lists
# years' worth of events; old ones contribute nothing for a 60d
# timespan because the player didn't compete in them recently
# anyway. Keeping only the most recent N events drops the per-scrape
# proxy load from ~80 sub-fetches to ~MAX_EVENT_GROUPS, which is the
# difference between 'often fails because proxies rate-limit' and
# 'completes in <60s every time'.
MAX_EVENT_GROUPS_OPEN = 25
MAX_EVENT_GROUPS_GC = 12


# ── EVENT-TIER WEIGHTS ────────────────────────────────────────────────
# Tier weight multiplies the contribution of a player's stats from
# this event when aggregating across multiple events. Higher tier =
# higher weight = those stats matter more.
#
# Goal: a T1 player in VCT looks better than a T2 player in VCL who
# looks better than a regional player in MrFunhaver — but all three
# are RECOGNISED by the scout score so up-and-coming talent surfaces.

TIER_VCT = 1.00          # VCT International League / Masters / Champions
TIER_VCL = 0.85          # VCL / Challengers / Ascension
TIER_FUNHAVER = 0.75     # MrFunhaver tournaments (close to VCL but lower)
TIER_OTHER = 0.50        # Anything else


def _classify_event_tier(name: str, region_key: str) -> float:
    """Return a multiplicative weight for stats from this event group.

    `region_key` is the user-facing region; we only honour MrFunhaver
    weighting in NA (where it's a recognised T2 circuit). In other
    regions a Funhaver event would be downgraded to TIER_OTHER.
    """
    if not name:
        return TIER_OTHER
    n = name.lower()

    # MrFunhaver-style tournaments — only count for NA
    if "funhaver" in n or "fun haver" in n or "mrfunhaver" in n:
        return TIER_FUNHAVER if region_key == "na" else TIER_OTHER

    # T2 markers take priority over the "Champions Tour" umbrella.
    # VLR names a lot of T2 events as "Champions Tour 2026: Challengers
    # <region>" — without this guard, the substring match below would
    # promote every Challengers player to VCT. Same for Ascension and
    # the "VCL <region>" naming. Also catch "Partner Series" — VLR's
    # tier-2 invitational that mixes Challengers + Partner orgs but is
    # NOT VCT proper. Players like kozzy/Proxh/Lime on Enterprise
    # Esports compete here and should keep their VCL tag.
    if (
        "challengers" in n
        or "vcl" in n
        or "ascension" in n
        or "promotion" in n
        or "partner series" in n
    ):
        return TIER_VCL

    # OFF//SEASON tournaments are mixed-tier exhibitions — real VCT
    # teams scrim alongside open-circuit teams. Bucketing them as VCT
    # falsely inflates lower-tier participants. Treat as OTHER.
    if "off//season" in n or "off-season" in n or "offseason" in n:
        return TIER_OTHER

    # T1: VCT International League, Masters, Champions, Lock-In, Kickoff
    if (
        "masters" in n
        or "vct champions" in n
        or "lock-in" in n or "lock in" in n
        or "kickoff" in n
        or "champions tour" in n  # umbrella — only T1 once T2 ruled out above
        or " vct" in f" {n}"
        or n.startswith("vct")
    ):
        return TIER_VCT

    return TIER_OTHER


async def _discover_event_group_ids(client) -> tuple[list, list, list]:
    """
    Fetch the VLR.gg stats dropdown and return three lists of (id, name)
    tuples, capped to the most-recent N events (highest numeric IDs):
      - open_ids:     non-GC, non-Funhaver events
      - gc_ids:       Game Changers events
      - funhaver_ids: MrFunhaver tournament series (NA T2 circuit)

    Older events almost never produce any rows under a 60/90-day
    timespan, so scraping the full ~80 wastes upstream calls and
    increases proxy rate-limit risk. Funhaver gets its own bucket so
    we can always include it for NA even when its event_group_id is
    older than the top-25 cutoff.
    """
    resp = await fetch_with_retries(
        f"{VLR_STATS_URL}/", client=client, timeout=10, max_retries=1
    )
    html = HTMLParser(resp.text)
    select = html.css_first("select[name='event_group_id']")
    open_pairs: list[tuple[str, str]] = []
    gc_pairs: list[tuple[str, str]] = []
    funhaver_pairs: list[tuple[str, str]] = []
    if select:
        for option in select.css("option"):
            val = option.attributes.get("value", "all")
            if val == "all":
                continue
            name = (option.text() or "").strip()
            if _is_gc_event(name):
                gc_pairs.append((val, name))
            elif _is_funhaver_event(name):
                funhaver_pairs.append((val, name))
            else:
                open_pairs.append((val, name))

    def _by_id_desc(pairs):
        try:
            return sorted(pairs, key=lambda x: int(x[0]), reverse=True)
        except ValueError:
            return pairs

    open_pairs = _by_id_desc(open_pairs)[:MAX_EVENT_GROUPS_OPEN]
    gc_pairs = _by_id_desc(gc_pairs)[:MAX_EVENT_GROUPS_GC]
    # Funhaver: keep all of them — there are typically only 1-3 active
    # series, and missing one means missing players like Zanks.
    funhaver_pairs = _by_id_desc(funhaver_pairs)
    logger.info(
        "Using %d open, %d GC, and %d Funhaver event groups",
        len(open_pairs), len(gc_pairs), len(funhaver_pairs)
    )
    return open_pairs, gc_pairs, funhaver_pairs


def _stats_url(event_group_id: str, region: str, timespan: str) -> str:
    ts = "all" if timespan.lower() == "all" else f"{timespan}d"
    # min_rating=0 is CRITICAL. VLR's min_rating filter is the
    # opponents'-team-ELO threshold (1550 = only matches where both
    # teams were rated 1550+). For top-tier players whose schedule is
    # entirely against ranked teams it has no effect, but for lower-
    # tier / VCL / Game Changers / Funhaver players it slices their
    # season into a tiny sub-sample of their tougher matches —
    # producing wildly inflated averages from small-sample outliers
    # (e.g. elul showed rating 1.22 / 266 ACS at min_rating=1550 vs
    # her real 1.04 / 228 across 408 rounds at min_rating=0). We
    # always want the full denominator so per-player rating/ACS
    # match what's on the player's profile page.
    return (
        f"{VLR_STATS_URL}/?event_group_id={event_group_id}&event_id=all"
        f"&region={region}&country=all&min_rounds=0"
        f"&min_rating=0&agent=all&map_id=all&timespan={ts}"
    )


async def _fetch_rows(url: str, client, semaphore: asyncio.Semaphore) -> list:
    """Fetch one event_group's stats rows. Retries up to 3 times when
    we get an empty result, because:
      - proxies sometimes return a VLR redirect (default /stats page,
        no rows match our region filter → 0 rows)
      - proxies sometimes return cached/transformed content
      - first call may have hit a rate-limited proxy

    Each retry rolls a fresh proxy randomization in _fetch_via_proxy
    (when direct fails / circuit breaker is tripped), so consecutive
    attempts hit different upstream paths.
    """
    async with semaphore:
        for attempt in range(3):
            try:
                resp = await fetch_with_retries(
                    url, client=client, timeout=15, max_retries=1
                )
                if resp.status_code != 200:
                    if attempt < 2:
                        await asyncio.sleep(0.4)
                        continue
                    return []
                html = HTMLParser(resp.text)
                rows = []
                for item in html.css("tbody tr"):
                    parsed = _parse_stats_row(item)
                    if parsed["player"]:
                        rows.append(parsed)
                # If we got rows OR this is the last attempt, accept it
                if rows or attempt >= 2:
                    return rows
                # Empty result from a 200 response — likely a redirect
                # to /stats (event_group_id mismatch) or cached page.
                # Retry through a different proxy path.
                logger.debug(
                    "Empty rows from %s on attempt %d — retrying", url, attempt + 1
                )
                await asyncio.sleep(0.4)
            except Exception as exc:
                logger.warning(
                    "Failed to fetch %s (attempt %d): %s", url, attempt + 1, exc
                )
                if attempt >= 2:
                    return []
                await asyncio.sleep(0.5)
        return []


def _to_float(s: str) -> float:
    """Parse a number out of a possibly-percentage-suffixed string."""
    if not s:
        return 0.0
    try:
        return float(str(s).rstrip("%").strip() or 0)
    except ValueError:
        return 0.0


# Fields that should be weighted by rounds_played when aggregating
# multiple per-event-group rows for the same player. These are all
# per-round metrics where summing makes no sense but a rounds-weighted
# average gives the correct overall.
_WEIGHTED_FIELDS = (
    "rating",
    "average_combat_score",
    "kill_deaths",
    "average_damage_per_round",
    "kills_per_round",
    "assists_per_round",
    "first_kills_per_round",
    "first_deaths_per_round",
)
_PERCENT_FIELDS = (
    "headshot_percentage",
    "clutch_success_percentage",
    "kill_assists_survived_traded",
)


def _merge(primary: list, secondary: list, secondary_tier: float = 1.0) -> list:
    """Merge two player lists with tier×rounds-weighted aggregation.

    Each event_group_id we scrape returns per-event stats for the
    players who competed in it. A player with rows in 5 events should
    show a tier-weighted aggregate that respects which tier the data
    came from — VCT performance contributes more than VCL, which
    contributes more than MrFunhaver / Other.

    `secondary_tier` is the tier weight (0.5–1.0) for the rows we're
    merging into the running primary. Each row's effective weight in
    the average is `rounds × tier_weight`. The `_acc_weight` field on
    each player accumulates the running total for chained merges.

    Aggregation rules:
      - rounds_played: SUM (real rounds, unweighted)
      - rating/ACS/K:D/ADR/KPR/APR/FKPR/FDPR: tier×rounds-weighted MEAN
      - HS%/Clutch%/KAST%: tier×rounds-weighted MEAN
      - agents: UNION across events
      - event_tier: MAX (best tier the player has competed in)
      - org/team_full: first non-empty
    """
    out = list(primary)
    by_id = {r["player_id"]: i for i, r in enumerate(out) if r.get("player_id")}
    by_name = {r["player"]: i for i, r in enumerate(out) if not r.get("player_id")}

    for r in secondary:
        pid = r.get("player_id")
        idx = by_id.get(pid) if pid else by_name.get(r.get("player"))
        if idx is None:
            # First time seeing this player — stamp tier metadata
            r = dict(r)
            new_rounds = _to_float(r.get("rounds_played"))
            r["_acc_weight"] = new_rounds * secondary_tier
            r["event_tier"] = round(secondary_tier, 2)
            out.append(r)
            if pid:
                by_id[pid] = len(out) - 1
            else:
                by_name[r["player"]] = len(out) - 1
            continue

        existing = out[idx]
        old_w = float(existing.get("_acc_weight", 0.0)) or _to_float(existing.get("rounds_played"))
        new_rounds = _to_float(r.get("rounds_played"))
        new_w = new_rounds * secondary_tier
        total_w = old_w + new_w
        if total_w <= 0:
            continue

        for f in _WEIGHTED_FIELDS:
            old_v = _to_float(existing.get(f))
            new_v = _to_float(r.get(f))
            avg = (old_v * old_w + new_v * new_w) / total_w
            existing[f] = f"{avg:.2f}"

        for f in _PERCENT_FIELDS:
            old_v = _to_float(existing.get(f))
            new_v = _to_float(r.get(f))
            avg = (old_v * old_w + new_v * new_w) / total_w
            had_pct = "%" in str(existing.get(f) or "") or "%" in str(r.get(f) or "")
            existing[f] = f"{round(avg)}%" if had_pct else f"{avg:.2f}"

        old_rounds = _to_float(existing.get("rounds_played"))
        existing["rounds_played"] = str(int(old_rounds + new_rounds))
        existing["_acc_weight"] = total_w
        existing["event_tier"] = round(max(float(existing.get("event_tier", 0.0)), secondary_tier), 2)

        agents_old = existing.get("agents") or []
        agents_new = r.get("agents") or []
        if agents_new:
            existing["agents"] = list(dict.fromkeys(list(agents_old) + list(agents_new)))

        for f in ("org", "team_full", "country"):
            if not existing.get(f) or existing.get(f) in {"", "N/A"}:
                if r.get(f) and r.get(f) not in {"", "N/A"}:
                    existing[f] = r[f]

    return out


async def _fetch_all_for_region(
    region: str,
    timespan: str,
    pairs_or_ids: list,
    client,
    region_key: str = None,
) -> list:
    """Scrape every event group in `pairs_or_ids` for the given region.

    `pairs_or_ids` may be a list of (id, name) tuples (preferred —
    enables tier weighting) OR plain id strings (legacy — falls back
    to TIER_OTHER for everything).
    """
    # Normalize to (id, name) pairs
    pairs = []
    for item in pairs_or_ids:
        if isinstance(item, tuple) and len(item) >= 2:
            pairs.append((item[0], item[1] or ""))
        else:
            pairs.append((str(item), ""))

    region_for_tier = region_key or region
    semaphore = asyncio.Semaphore(_SEMAPHORE_LIMIT)
    urls = [_stats_url(eid, region, timespan) for eid, _ in pairs]
    fetched = await asyncio.gather(*[_fetch_rows(u, client, semaphore) for u in urls])

    merged: list = []
    for (eid, name), rows in zip(pairs, fetched):
        tier = _classify_event_tier(name, region_for_tier)
        merged = _merge(merged, rows, secondary_tier=tier)
    # Strip the internal accumulator before returning
    for r in merged:
        r.pop("_acc_weight", None)
    return merged


async def _safe_discover(client) -> tuple[list, list, list]:
    """Discover event group IDs; on failure, return empty lists rather than raising."""
    try:
        return await _discover_event_group_ids(client)
    except Exception as exc:
        logger.warning("Event-group discovery failed (continuing with fallback): %s", exc)
        return [], [], []


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
        open_ids, gc_ids, funhaver_ids = await _safe_discover(client)

        # NA gets MrFunhaver pinned in alongside the top-N open events
        # — those events frequently have older event_group_ids that
        # would otherwise be cut by the cap, missing T2 NA players
        # like Zanks who only show up there. Other regions ignore
        # Funhaver entirely (per _classify_event_tier policy).
        if region_key == "na" and funhaver_ids:
            open_ids = list(open_ids) + list(funhaver_ids)

        if region_key == "gc":
            rows = await _fetch_all_for_region("gc", timespan, gc_ids, client, region_key="gc") if gc_ids else []
            if not rows:
                rows = await _fetch_single_all("gc", timespan, client)
            if not rows and (open_ids or gc_ids):
                all_pairs = list(open_ids) + list(gc_ids)
                rows = await _fetch_all_for_region("gc", timespan, all_pairs, client, region_key="gc")

        elif region_key in LA_COMBINED:
            if open_ids:
                las_rows, lan_rows = await asyncio.gather(
                    _fetch_all_for_region("las", timespan, open_ids, client, region_key="la"),
                    _fetch_all_for_region("lan", timespan, open_ids, client, region_key="la"),
                )
                rows = _merge(las_rows, lan_rows)
            else:
                las_rows, lan_rows = await asyncio.gather(
                    _fetch_single_all("las", timespan, client),
                    _fetch_single_all("lan", timespan, client),
                )
                rows = _merge(las_rows, lan_rows)

        else:
            vlr_region = VLR_REGION_MAP.get(region_key, region_key)
            if open_ids:
                rows = await _fetch_all_for_region(vlr_region, timespan, open_ids, client, region_key=region_key)
            else:
                rows = await _fetch_single_all(vlr_region, timespan, client)

        logger.info("vlr_stats(%s, %s): %d players", region_key, timespan, len(rows))
        return {"data": {"status": 200, "segments": rows}}

    return await cache_manager.get_or_create_async(
        CACHE_TTL_STATS, build, "stats", region_key, timespan
    )
