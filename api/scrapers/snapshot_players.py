"""
Weekly player-stats snapshot for the admin movers report.

Pulls /v2/stats?timespan=90d for each VCT region, trims each segment
to the fields the frontend needs to recompute SKILL grades, and
inserts a single jsonb row per snapshot date into
public.player_snapshots.

The frontend's movers widget reads the two most recent rows and
diffs SKILL grades per player_id. Snapshots are tiny (a few hundred
KB per week × 12-week retention = ~5 MB ceiling).
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from typing import Any

import httpx

from api.scrapers.stats import vlr_stats

logger = logging.getLogger(__name__)


# Regions to snapshot. Keep aligned with the frontend's region picker —
# adding a new region here is harmless (just adds bytes to the row).
SNAPSHOT_REGIONS = ("na", "eu", "br", "kr", "ap", "jp", "cn")

# Fields the frontend movers widget needs. Anything else from the
# /v2/stats segment is dropped to keep the jsonb compact and to avoid
# leaking schema we don't actually use.
_KEEP_FIELDS = (
    "player",
    "player_id",
    "org",
    "country",
    "rating",
    "average_combat_score",
    "kill_deaths",
    "average_damage_per_round",
    "kills_per_round",
    "kill_assists_survived_traded",
    "first_kills_per_round",
    "first_deaths_per_round",
    "headshot_percentage",
    "clutch_success_percentage",
    "rounds_played",
    "main_rating",
    "main_average_combat_score",
    "main_kill_deaths",
    "main_rounds_played",
    "current_main_rounds_played",
    "event_tier",
    "agents",
    "team_placement",
)

# Retention: keep last 12 weekly rows. Older snapshots are deleted in
# the same pass so the table doesn't grow unbounded.
_RETENTION_ROWS = 12


def _trim_segment(seg: dict) -> dict:
    return {k: seg.get(k) for k in _KEEP_FIELDS if k in seg}


async def run_snapshot() -> dict:
    """Fetch all snapshot regions, build the payload, write to Supabase.
    Returns a summary dict with per-region row counts.
    """
    supabase_url = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
    service_role = (os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    if not supabase_url or not service_role:
        raise RuntimeError(
            "Snapshot needs SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY env vars."
        )

    payload: dict[str, Any] = {"regions": {}, "schema_version": 1}
    counts: dict[str, int] = {}

    for region in SNAPSHOT_REGIONS:
        try:
            result = await vlr_stats(region, "90d", exclude_funhaver=False)
            segments = (result or {}).get("data", {}).get("segments", []) or []
            trimmed = [_trim_segment(s) for s in segments if s.get("player_id")]
            payload["regions"][region] = trimmed
            counts[region] = len(trimmed)
        except Exception as exc:
            logger.warning("[snapshot] %s scrape failed: %s", region, exc)
            payload["regions"][region] = []
            counts[region] = 0

    today = _dt.date.today().isoformat()

    headers = {
        "apikey": service_role,
        "Authorization": f"Bearer {service_role}",
        "Content-Type": "application/json",
        # merge-duplicates makes re-running on the same day overwrite
        # rather than 409 on the primary-key conflict. Useful when
        # debugging: re-trigger the workflow without manual cleanup.
        "Prefer": "resolution=merge-duplicates,return=representation",
    }
    body = {"snapshot_date": today, "data": payload}

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            r = await client.post(
                f"{supabase_url}/rest/v1/player_snapshots",
                headers=headers,
                json=body,
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Supabase write failed: {exc}")
    if r.status_code >= 400:
        raise RuntimeError(
            f"Supabase returned {r.status_code}: {r.text[:300]}"
        )

    # Prune old snapshots — keep only the most recent _RETENTION_ROWS.
    # Best-effort: a prune failure does NOT roll back the insert, since
    # the snapshot is the important part. Old rows just stop costing
    # meaningful storage.
    pruned: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            list_resp = await client.get(
                f"{supabase_url}/rest/v1/player_snapshots",
                headers={
                    "apikey": service_role,
                    "Authorization": f"Bearer {service_role}",
                },
                params={
                    "select": "snapshot_date",
                    "order": "snapshot_date.desc",
                },
            )
            if list_resp.status_code < 400:
                rows = list_resp.json() or []
                stale = rows[_RETENTION_ROWS:]
                for s in stale:
                    d = s.get("snapshot_date")
                    if not d:
                        continue
                    del_resp = await client.delete(
                        f"{supabase_url}/rest/v1/player_snapshots",
                        headers={
                            "apikey": service_role,
                            "Authorization": f"Bearer {service_role}",
                        },
                        params={"snapshot_date": f"eq.{d}"},
                    )
                    if del_resp.status_code < 400:
                        pruned.append(d)
    except Exception as exc:
        logger.warning("[snapshot] prune failed: %s", exc)

    total = sum(counts.values())
    return {
        "snapshot_date": today,
        "total_players": total,
        "regions": counts,
        "pruned": pruned,
    }
