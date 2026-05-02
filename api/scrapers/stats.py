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

def _job_key(region_key: str, timespan: str, exclude_funhaver: bool = False) -> str:
    suffix = ":nofun" if exclude_funhaver else ""
    return f"stats:{region_key}:{timespan}{suffix}"

def _cache_args(region_key: str, timespan: str, exclude_funhaver: bool = False):
    base = ("stats", region_key, timespan)
    return base + ("nofun",) if exclude_funhaver else base

def get_cached_stats(region_key: str, timespan: str, exclude_funhaver: bool = False):
    """Return cached stats payload or None."""
    return cache_manager.get(CACHE_TTL_STATS, *_cache_args(region_key, timespan, exclude_funhaver))

def is_building(region_key: str, timespan: str, exclude_funhaver: bool = False) -> bool:
    """Return True if a background scrape is in progress."""
    return _job_key(region_key, timespan, exclude_funhaver) in _building

def recently_failed(region_key: str, timespan: str, exclude_funhaver: bool = False) -> bool:
    """Return True if a recent build failed (within backoff window)."""
    key = _job_key(region_key, timespan, exclude_funhaver)
    ts = _failed_builds.get(key)
    return bool(ts and (time.time() - ts) < _FAILURE_BACKOFF)

# Major regions where 0 players returned almost certainly means the scrape
# failed silently (empty cache poisoning) rather than a legitimate "no data".
# For these, an empty result is treated as a failure and NOT cached.
_MAJOR_REGIONS = {"na", "eu", "ap", "kr", "br", "cn", "jp"}

# Per-region "expected" country bias — the dominant country codes
# we'd expect in a healthy scrape. If the actual distribution comes
# back with too few of these (i.e. the scrape leaked into a global
# unfiltered result), we reject the build and don't cache the trash.
# Prevents the user from staring at stale "904 mostly-US players"
# under the China tab when VLR's region filter silently failed.
_REGION_EXPECTED_COUNTRIES = {
    "cn": {"cn", "hk", "mo", "tw"},
    "kr": {"kr"},
    "jp": {"jp"},
    "br": {"br"},
}
def _region_country_dominance(region_key: str, segments: list) -> float:
    expected = _REGION_EXPECTED_COUNTRIES.get(region_key)
    if not expected or not segments:
        return 1.0
    matches = sum(1 for s in segments if (s.get("country") or "").lower() in expected)
    return matches / len(segments)


async def _background_build(region_key: str, timespan: str, exclude_funhaver: bool = False):
    key = _job_key(region_key, timespan, exclude_funhaver)
    cargs = _cache_args(region_key, timespan, exclude_funhaver)
    try:
        # Hard timeout so a single bad scrape can't hang for minutes
        result = await asyncio.wait_for(
            vlr_stats(region_key, timespan, exclude_funhaver=exclude_funhaver),
            timeout=_BUILD_HARD_TIMEOUT,
        )
        segments = (result or {}).get("data", {}).get("segments", [])
        # Sanity: country-bias check for regions where we KNOW the
        # dominant country codes. CN was returning 904 mostly-US/TR
        # players from a leaked filter — if dominance falls below
        # ~25%, the filter clearly didn't take. Treat as failure.
        dominance = _region_country_dominance(region_key, segments)
        if segments and dominance < 0.25:
            logger.warning(
                "Region %s returned %d players but only %.0f%% match expected countries — leaked filter, invalidating",
                key, len(segments), dominance * 100,
            )
            cache_manager.invalidate(CACHE_TTL_STATS, *cargs)
            _failed_builds[key] = time.time()
            return
        # For MAJOR regions, 0 players is almost certainly a failure
        if not segments and region_key in _MAJOR_REGIONS:
            logger.warning("Major region %s returned 0 players — invalidating cache", key)
            try:
                cache_manager.invalidate(CACHE_TTL_STATS, *cargs)
            except Exception:
                pass
            _failed_builds[key] = time.time()
        else:
            _failed_builds.pop(key, None)
            logger.info("Background stats build complete: %s (%d players)", key, len(segments))
            # Post-scrape QA pass — scan for players with missing
            # critical fields (country, top agents, org tag) and try
            # to repair via per-player profile fetches. Runs in the
            # same task so the patched cache is in place before any
            # frontend re-fetch happens. Capped to avoid runaway cost.
            try:
                await _post_scrape_qa(region_key, timespan, exclude_funhaver, result, cargs)
            except Exception as exc:
                logger.warning("[qa] %s: post-scrape QA pass failed (non-fatal): %s", key, exc)
    except asyncio.TimeoutError:
        logger.error("Background stats build TIMED OUT after %ds: %s", _BUILD_HARD_TIMEOUT, key)
        try:
            cache_manager.invalidate(CACHE_TTL_STATS, *cargs)
        except Exception:
            pass
        _failed_builds[key] = time.time()
    except Exception as exc:
        logger.error("Background stats build failed for %s: %s", key, exc)
        _failed_builds[key] = time.time()
    finally:
        _building.discard(key)

async def _post_scrape_qa(region_key: str, timespan: str, exclude_funhaver: bool,
                          payload: dict, cargs: tuple):
    """Review a freshly-cached scrape, identify players with missing
    fields, and try to fill the gaps via per-player profile lookups.

    Anomalies we can repair (player_id required for profile lookup):
      · empty `country`         → /v2/player → country
      · empty `agents` list     → /v2/player → agent_stats top 3
      · empty / "N/A" `org`     → /v2/player → current_team.tag

    Anomalies we can't auto-repair (logged only):
      · empty `player_id`       (no profile to look up)
      · zero `rating` while rounds_played > 0  (parse miss; the profile
        endpoint doesn't carry per-timespan stats we'd need to backfill)

    Capped at MAX_REPAIRS per pass to avoid blasting VLR with hundreds
    of profile fetches if a scrape comes back wholesale broken — the
    real fix in that case is a re-scrape, not a profile-by-profile
    bandage.
    """
    segments = (payload or {}).get("data", {}).get("segments") or []
    if not segments:
        return

    key = _job_key(region_key, timespan, exclude_funhaver)
    repair_targets: list[tuple[int, str, list[str]]] = []
    unrepairable_no_id = 0
    unrepairable_zero_rating = 0
    for idx, seg in enumerate(segments):
        pid = (seg.get("player_id") or "").strip()
        # Track-only signals that we can't fix from /v2/player.
        try:
            rounds = int(seg.get("rounds_played") or 0)
        except (TypeError, ValueError):
            rounds = 0
        try:
            rating = float(seg.get("rating") or 0)
        except (TypeError, ValueError):
            rating = 0.0
        if not pid:
            unrepairable_no_id += 1
            continue
        if rating == 0 and rounds > 0:
            unrepairable_zero_rating += 1
        # Repairable signals.
        missing: list[str] = []
        if not (seg.get("country") or "").strip():
            missing.append("country")
        if not seg.get("agents") and rounds > 0:
            missing.append("agents")
        if (seg.get("org") or "").strip() in ("", "N/A"):
            missing.append("org")
        if missing:
            repair_targets.append((idx, pid, missing))

    if unrepairable_no_id or unrepairable_zero_rating:
        logger.info(
            "[qa] %s: %d segments without player_id (unrepairable), %d with zero rating + non-zero rounds (unrepairable)",
            key, unrepairable_no_id, unrepairable_zero_rating,
        )

    if not repair_targets:
        logger.info("[qa] %s: clean scan, %d segments, no missing fields", key, len(segments))
        return

    MAX_REPAIRS = 30
    overflow = max(0, len(repair_targets) - MAX_REPAIRS)
    repair_targets = repair_targets[:MAX_REPAIRS]
    if overflow:
        logger.warning(
            "[qa] %s: %d players have missing fields — capping repair pass at %d (consider a re-scrape)",
            key, len(repair_targets) + overflow, MAX_REPAIRS,
        )
    else:
        logger.info("[qa] %s: attempting to repair %d players with missing fields", key, len(repair_targets))

    # Late import — avoids the circular import at module load time.
    from .players import vlr_player

    semaphore = asyncio.Semaphore(4)  # gentler than the main scrape
    PROFILE_TIMEOUT = 15

    async def repair_one(idx: int, pid: str, missing: list[str]):
        async with semaphore:
            try:
                profile = await asyncio.wait_for(vlr_player(pid), timeout=PROFILE_TIMEOUT)
            except Exception as exc:
                logger.debug("[qa] %s: vlr_player(%s) failed: %s", key, pid, exc)
                return idx, None, missing
            profile_segs = (profile or {}).get("data", {}).get("segments") or []
            if not profile_segs:
                return idx, None, missing
            p = profile_segs[0]
            patch: dict = {}
            if "country" in missing:
                c = (p.get("country") or "").strip().lower()
                if c:
                    patch["country"] = c
            if "agents" in missing:
                agents: list[str] = []
                for a in (p.get("agent_stats") or [])[:3]:
                    name = a.get("agent") if isinstance(a, dict) else None
                    if name:
                        agents.append(name)
                if agents:
                    patch["agents"] = agents
            if "org" in missing:
                ct = p.get("current_team") or {}
                tag = (ct.get("tag") or "").strip()
                if tag:
                    patch["org"] = tag
            return idx, patch or None, missing

    results = await asyncio.gather(*(repair_one(i, p, m) for (i, p, m) in repair_targets))
    repaired_count = 0
    field_counts = {"country": 0, "agents": 0, "org": 0}
    for idx, patch, missing in results:
        if not patch:
            continue
        seg = segments[idx]
        for k_, v in patch.items():
            seg[k_] = v
            field_counts[k_] = field_counts.get(k_, 0) + 1
        repaired_count += 1

    if repaired_count:
        # Re-cache the patched payload so subsequent reads see the fix.
        # In-place mutation already updated the cached envelope (lists
        # and dicts are passed by reference) but call set() explicitly
        # to bump the cache's TTL anchor and survive any backend that
        # treats cache values as immutable.
        try:
            cache_manager.set(CACHE_TTL_STATS, payload, *cargs)
        except Exception as exc:
            logger.debug("[qa] %s: re-cache failed (mutation already applied): %s", key, exc)
        logger.info(
            "[qa] %s: repaired %d/%d players (country=%d agents=%d org=%d)",
            key, repaired_count, len(repair_targets),
            field_counts["country"], field_counts["agents"], field_counts["org"],
        )
    else:
        logger.info("[qa] %s: 0 repairs applied (all %d profile lookups failed or yielded no fix)",
                    key, len(repair_targets))


def start_background_build(region_key: str, timespan: str, exclude_funhaver: bool = False) -> bool:
    """Fire a background scrape task. Returns False if one is already running."""
    key = _job_key(region_key, timespan, exclude_funhaver)
    if key in _building:
        return False
    _building.add(key)
    asyncio.create_task(_background_build(region_key, timespan, exclude_funhaver))
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
MAX_EVENT_GROUPS_OPEN = 60
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
TIER_GC = 0.85           # Game Changers — legit pro circuit, treat ~= VCL
                         # (used when a stray GC event surfaces on a
                         # non-gc region request)
TIER_GC_MINOR = 0.80     # Game Changers regional leagues outside the
                         # four major franchise regions (LATAM, Oceania,
                         # MENA, Asia, etc.) when the user is on the gc
                         # region. Talent pool is shallower than EMEA /
                         # NA / APAC / Brazil so raw stats don't
                         # translate 1:1.
TIER_FUNHAVER = 0.75     # MrFunhaver tournaments (close to VCL but lower)
TIER_RIVALS = 0.65       # EU Rivals League / open + closed qualifiers —
                         # semi-pro circuit one rung below Challengers.
                         # Lower than Funhaver because the talent pool
                         # is broader / less curated. Region-
                         # disambiguated by the frontend so the label
                         # reads "RIVALS" on EU rows.
TIER_OTHER = 0.50        # Anything else


# GC sub-region detection. Match on the event name (already lowercased
# by the caller). The four major franchise regions Riot runs full GC
# leagues for keep TIER_VCT; everything else drops to TIER_GC_MINOR.
def _is_gc_major_region(event_name_lower: str) -> bool:
    n = event_name_lower
    padded = f" {n} "
    # EMEA — single distinctive token, safe to match anywhere.
    if "emea" in n:
        return True
    # North America. 'NA' as a bare token is dangerous (it appears in
    # 'mena', 'final', etc.) so require boundary.
    if "north america" in n or " na " in padded or " na:" in n:
        return True
    # APAC / Pacific.
    if "apac" in n or "pacific" in n:
        return True
    # Brazil (English + Portuguese spelling).
    if "brazil" in n or "brasil" in n:
        return True
    return False


def _classify_event_tier(name: str, region_key: str) -> float:
    """Return a multiplicative weight for stats from this event group.

    `region_key` is the user-facing region; we only honour MrFunhaver
    weighting in NA (where it's a recognised T2 circuit). In other
    regions a Funhaver event would be downgraded to TIER_OTHER.
    """
    if not name:
        return TIER_OTHER
    n = name.lower()

    # Game Changers — Riot's professional women's circuit. The main GC
    # league/stages ARE the top of their region (it's a self-contained
    # circuit with one championship), so for the gc region they weight
    # the same as VCT (1.0). Cash Cups are smaller side tournaments
    # held alongside the main league — those drop to OTHER (0.5).
    # This block goes FIRST so 'Champions Tour 2026: Game Changers'
    # isn't misread as T1 by the champions-tour fallback below.
    is_gc_event = (
        "game changers" in n
        or "gamechangers" in n.replace(" ", "")
        or " gc " in f" {n} "
        or "vctgc" in n.replace(" ", "")
    )
    if is_gc_event:
        # Cash Cups are smaller side events; demote to OTHER so they
        # don't inflate scores from a few low-stakes matches.
        if "cash cup" in n or "cashcup" in n.replace(" ", ""):
            return TIER_OTHER
        # In the GC region the main league IS the top tier of the
        # circuit. Within that, the four major franchise regions
        # (EMEA, NA, APAC/Pacific, Brazil) carry full TIER_VCT
        # weight; smaller regional GC leagues (LATAM, Oceania,
        # MENA, Asia) take a 20% haircut to TIER_GC_MINOR (0.80)
        # because the talent pool is shallower and raw stats from
        # those populations don't translate 1:1 with the majors.
        # The match is on the EVENT name, so it reflects which
        # league the team competed in (not where the player is from).
        if region_key == "gc":
            return TIER_VCT if _is_gc_major_region(n) else TIER_GC_MINOR
        # In other regions a stray GC event would still represent
        # T2-equivalent skill (TIER_VCL).
        return TIER_VCL

    # MrFunhaver-style tournaments — only count for NA
    if "funhaver" in n or "fun haver" in n or "mrfunhaver" in n:
        return TIER_FUNHAVER if region_key == "na" else TIER_OTHER

    # EU semi-pro circuit (region-scoped) — Rivals League and the open
    # qualifiers that feed Challengers. Lower-stakes than VCL but
    # higher than 'Other'. Match before VCL/VCT keyword checks so
    # 'Challengers Open Qualifier' lands here, not in TIER_VCL.
    if region_key == "eu":
        # Tier-3 EU circuit: Rivals League + qualifiers + open brackets
        # + play-ins + any open-circuit / off-season event that isn't
        # a real VCL stage. Match BEFORE the VCL/VCT keyword checks so
        # 'VCL EMEA Open Qualifier' / 'Challengers Open Qualifier' /
        # 'EMEA Open Bracket' all land here, not in TIER_VCL.
        is_qualifier = (
            "qualifier" in n
            or "open league" in n
            or "open bracket" in n
            or "open ladder" in n
            or "play-in" in n
            or " play in " in f" {n} "
            or "play-ins" in n
        )
        is_lcq = "last chance qualifier" in n or "last-chance qualifier" in n
        is_rivals = "rivals" in n
        if (is_qualifier and not is_lcq) or is_rivals:
            return TIER_RIVALS

    # T2 markers take priority over the "Champions Tour" umbrella.
    # VLR names a lot of T2 events as "Champions Tour 2026: Challengers
    # <region>" — without this guard, the substring match below would
    # promote every Challengers player to VCT. Same for Ascension and
    # the "VCL <region>" naming. Also catch "Partner Series" — VLR's
    # tier-2 invitational that mixes Challengers + Partner orgs but is
    # NOT VCT proper. Players like kozzy/Proxh/Lime on Enterprise
    # Esports compete here and should keep their VCL tag.
    # Note: dropped bare "promotion" because it matched too many open
    # / community "Promotion Cup" / "Promotion League" events that
    # shouldn't get the VCL tag. Real T2 promotion events are still
    # caught by 'vcl' / 'ascension' / 'challengers' substrings.
    if (
        "challengers" in n
        or "vcl" in n
        or "ascension" in n
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


def _stats_url(event_group_id: str, region: str, timespan: str, page: int = 1) -> str:
    # Frontend now passes "30", "60", "90", or "365" — append the
    # 'd' suffix VLR's URL expects. The legacy "all" value is still
    # tolerated for cached/stale clients that haven't reloaded yet.
    ts_lower = (timespan or "").lower()
    ts = "all" if ts_lower == "all" else f"{timespan}d"
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
    page_param = f"&page={page}" if page > 1 else ""
    return (
        f"{VLR_STATS_URL}/?event_group_id={event_group_id}&event_id=all"
        f"&region={region}&country=all&min_rounds=0"
        f"&min_rating=0&agent=all&map_id=all&timespan={ts}{page_param}"
    )


# VLR paginates the stats table at ~50 rows per page. Without paging we
# only get page 1 per event group, which left smaller regions (LATAM,
# Pacific minor leagues) with as few as ~64 unique players after dedup
# across event groups — not because the region is small, but because
# the first-page-50 of each event mostly overlaps. Walk pages until one
# returns short, capped so a single mis-served URL can't loop forever.
_STATS_PAGE_SIZE_THRESHOLD = 50  # VLR's default; if a page returns
                                  # fewer than this, we've hit the end.
_MAX_PAGES_PER_EVENT = 10        # 10 × 50 = 500 max per event, more
                                  # than any real VLR event has.


async def _fetch_one_page(url: str, client, semaphore: asyncio.Semaphore) -> list:
    """Fetch a single page of stats rows. Retries up to 3 times when
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


async def _fetch_rows(
    event_group_id: str,
    region: str,
    timespan: str,
    client,
    semaphore: asyncio.Semaphore,
) -> list:
    """Fetch every page of stats rows for a single event_group_id.

    VLR paginates at ~50 rows/page — without walking pages we'd only
    get the first 50 players per event, which especially hurts smaller
    regions (LATAM was capped at 64 unique players because the first
    page of each event group mostly overlapped). Walk pages until one
    returns fewer than _STATS_PAGE_SIZE_THRESHOLD or we hit the cap.

    Pages are fetched sequentially because we don't know up front how
    many there are; in practice most events stop after page 1-2 so
    this is cheap.
    """
    all_rows: list = []
    seen_players = set()
    for page in range(1, _MAX_PAGES_PER_EVENT + 1):
        url = _stats_url(event_group_id, region, timespan, page=page)
        page_rows = await _fetch_one_page(url, client, semaphore)
        if not page_rows:
            break
        # Dedup within this event in case VLR returns overlapping rows
        # across pages (rare but possible if the table re-orders mid-walk).
        new_rows = [r for r in page_rows if r["player"] not in seen_players]
        for r in new_rows:
            seen_players.add(r["player"])
        all_rows.extend(new_rows)
        if len(page_rows) < _STATS_PAGE_SIZE_THRESHOLD:
            break
    return all_rows


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
    """Merge two player lists with split-weight aggregation:

      - **Stats** (rating, ACS, K/D, ADR, KPR/APR/FKPR/FDPR, HS%,
        Clutch%, KAST%) use **pure rounds-weighted MEAN**, matching
        what VLR shows on each player's profile page. This keeps our
        watchlist + table numbers identical to what users see on
        vlr.gg, so a 1.03 rating on VLR reads as 1.03 here too.

      - **event_tier** uses **tier×rounds-weighted MEAN**. Cached as
        the rolling sum of (rounds_i × tier_i) divided by total
        rounds. The frontend's scout-score multiplier picks this up
        so tier still factors into ranking — just not into the raw
        stats display.

    `secondary_tier` is the tier weight (0.5-1.0) of the rows being
    merged in. `_acc_weight` accumulates rounds × tier across merges
    purely for the event_tier calc.

    Aggregation rules:
      - rounds_played: SUM (real rounds, unweighted)
      - rating/ACS/K:D/ADR/KPR/APR/FKPR/FDPR: rounds-weighted MEAN
      - HS%/Clutch%/KAST%: rounds-weighted MEAN
      - agents: UNION across events
      - event_tier: rounds-weighted MEAN of the tiers the player
        played in. Used to be MAX (best tier the player ever touched),
        but that inflated qualifier players who briefly cameo'd in a
        Challengers event to look like full VCL competitors.
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
            # Track VCL+ rounds separately so the frontend can label
            # "primarily VCL or not" off the actual round percentage,
            # not a tier-weighted mean (which folds OTHER and RIVALS
            # together unevenly). VCL+ = TIER_VCL or higher.
            is_vcl_plus = secondary_tier >= TIER_VCL
            r["_vcl_rounds"] = new_rounds if is_vcl_plus else 0
            r["vcl_round_pct"] = 1.0 if is_vcl_plus else 0.0
            out.append(r)
            if pid:
                by_id[pid] = len(out) - 1
            else:
                by_name[r["player"]] = len(out) - 1
            continue

        existing = out[idx]
        # Stats use pure rounds weighting (matches VLR).
        old_rounds = _to_float(existing.get("rounds_played"))
        new_rounds = _to_float(r.get("rounds_played"))
        total_rounds = old_rounds + new_rounds
        if total_rounds <= 0:
            continue

        for f in _WEIGHTED_FIELDS:
            old_v = _to_float(existing.get(f))
            new_v = _to_float(r.get(f))
            avg = (old_v * old_rounds + new_v * new_rounds) / total_rounds
            existing[f] = f"{avg:.2f}"

        for f in _PERCENT_FIELDS:
            old_v = _to_float(existing.get(f))
            new_v = _to_float(r.get(f))
            avg = (old_v * old_rounds + new_v * new_rounds) / total_rounds
            had_pct = "%" in str(existing.get(f) or "") or "%" in str(r.get(f) or "")
            existing[f] = f"{round(avg)}%" if had_pct else f"{avg:.2f}"

        existing["rounds_played"] = str(int(total_rounds))

        # event_tier still uses tier×rounds weighting — kept independent
        # of the stat aggregation above so the scout-score multiplier
        # on the frontend has tier context to multiply with.
        old_w_tier = float(existing.get("_acc_weight", 0.0)) or old_rounds
        new_w_tier = new_rounds * secondary_tier
        total_w_tier = old_w_tier + new_w_tier
        existing["_acc_weight"] = total_w_tier
        total_rounds_int = int(total_rounds)
        if total_rounds_int > 0:
            existing["event_tier"] = round(total_w_tier / total_rounds_int, 2)

        # VCL+ rounds tracker — independent of tier-weighted mean.
        # Frontend uses vcl_round_pct directly for the EU label
        # (≥80% VCL+ rounds → VCL, else T3) so a player with 25%
        # off-season + 75% VCL doesn't get misclassified by the
        # tier-mean math (which folds OTHER and RIVALS unevenly).
        old_vcl_rounds = _to_float(existing.get("_vcl_rounds", 0))
        new_vcl_rounds = new_rounds if secondary_tier >= TIER_VCL else 0
        total_vcl_rounds = old_vcl_rounds + new_vcl_rounds
        existing["_vcl_rounds"] = total_vcl_rounds
        if total_rounds_int > 0:
            existing["vcl_round_pct"] = round(total_vcl_rounds / total_rounds, 3)

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
    fetched = await asyncio.gather(
        *[_fetch_rows(eid, region, timespan, client, semaphore) for eid, _ in pairs]
    )

    merged: list = []
    for (eid, name), rows in zip(pairs, fetched):
        tier = _classify_event_tier(name, region_for_tier)
        merged = _merge(merged, rows, secondary_tier=tier)
    # Strip the internal accumulator before returning
    for r in merged:
        r.pop("_acc_weight", None)
        r.pop("_vcl_rounds", None)
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
    it tends to leak across regions (which is why we normally avoid it).
    Walks pages like the per-event-group path does."""
    semaphore = asyncio.Semaphore(1)
    return await _fetch_rows("all", region, timespan, client, semaphore)


@handle_scraper_errors
async def vlr_stats(region_key: str, timespan: str, exclude_funhaver: bool = False):
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
        # When the caller explicitly opts out via ?exclude_funhaver=1,
        # we skip pinning the Funhaver event groups so the response
        # contains only ranked / VCT / VCL data.
        if region_key == "na" and funhaver_ids and not exclude_funhaver:
            open_ids = list(open_ids) + list(funhaver_ids)

        if region_key == "gc":
            rows = await _fetch_all_for_region("gc", timespan, gc_ids, client, region_key="gc") if gc_ids else []
            if not rows:
                rows = await _fetch_single_all("gc", timespan, client)
            if not rows and (open_ids or gc_ids):
                all_pairs = list(open_ids) + list(gc_ids)
                rows = await _fetch_all_for_region("gc", timespan, all_pairs, client, region_key="gc")

        elif region_key in LA_COMBINED:
            # LATAM combined: walk per-event AND merge in event_group=all.
            # Per-event alone misses events outside the top-60 cap;
            # all-events-mode picks up the long tail. Region filter
            # (las/lan) is applied server-side both ways, so leaks are
            # bounded.
            if open_ids:
                las_rows, lan_rows, las_all, lan_all = await asyncio.gather(
                    _fetch_all_for_region("las", timespan, open_ids, client, region_key="la"),
                    _fetch_all_for_region("lan", timespan, open_ids, client, region_key="la"),
                    _fetch_single_all("las", timespan, client),
                    _fetch_single_all("lan", timespan, client),
                )
                rows = _merge(_merge(las_rows, lan_rows), _merge(las_all, lan_all))
            else:
                las_rows, lan_rows = await asyncio.gather(
                    _fetch_single_all("las", timespan, client),
                    _fetch_single_all("lan", timespan, client),
                )
                rows = _merge(las_rows, lan_rows)

        else:
            vlr_region = VLR_REGION_MAP.get(region_key, region_key)
            if open_ids:
                # For LATAM South / LATAM North standalone (and any
                # other small region where most top-60 events return 0
                # rows under the region filter) augment per-event with
                # event_group=all so we don't miss the long tail of
                # smaller leagues. Costs one extra request per region.
                if region_key in ("la-s", "la-n"):
                    per_event, all_events = await asyncio.gather(
                        _fetch_all_for_region(vlr_region, timespan, open_ids, client, region_key=region_key),
                        _fetch_single_all(vlr_region, timespan, client),
                    )
                    rows = _merge(per_event, all_events)
                else:
                    rows = await _fetch_all_for_region(vlr_region, timespan, open_ids, client, region_key=region_key)
            else:
                rows = await _fetch_single_all(vlr_region, timespan, client)

        logger.info("vlr_stats(%s, %s): %d players", region_key, timespan, len(rows))
        return {"data": {"status": 200, "segments": rows}}

    # Cache key includes the exclude_funhaver flag so the two
    # variants don't collide.
    cache_args = ("stats", region_key, timespan)
    if exclude_funhaver:
        cache_args = cache_args + ("nofun",)
    return await cache_manager.get_or_create_async(
        CACHE_TTL_STATS, build, *cache_args
    )
