"""
Weekly snapshot of Supabase tables that hold user-curated data
(scout_data, vods) → R2.

Why: scout_data is irreplaceable — watchlists, notes, tracker entries.
Supabase Hobby has a 7-day rolling backup, so this gives us 12 weekly
snapshots stored independently of Supabase's own infra. ~$0 cost
(R2 free tier covers it; rows are tiny JSON).

Layout in R2:
    backups/scout_data/YYYY-MM-DD.json
    backups/vods/YYYY-MM-DD.json

Retention: keeps the most recent 12 weekly snapshots per table; older
ones are deleted on each run. If the prune step fails for any reason
the upload still succeeds — old backups don't expire automatically,
they just stop costing meaningful storage.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from typing import Any

import httpx

from api.scrapers.r2_uploads import _get_client

logger = logging.getLogger(__name__)


# Tables we snapshot. Order doesn't matter; each is an independent file.
_BACKUP_TABLES = ("scout_data", "vods")
# Rolling retention. Twelve weeks ≈ a quarter of a year — enough that
# you can roll back through a season's worth of bad commits without
# storage bloat.
_RETENTION_WEEKS = 12


def _supabase_select_all(table: str) -> list[dict[str, Any]]:
    """Read every row from a Supabase table via the REST API + Service
    Role key. Pages in 1000-row chunks (Supabase's default page size)
    so very large tables don't truncate.
    """
    url = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
    key = (os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    if not url or not key:
        raise RuntimeError(
            "Backup needs SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY env vars."
        )

    out: list[dict[str, Any]] = []
    page_size = 1000
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Range-Unit": "items",
    }
    with httpx.Client(timeout=30.0) as client:
        offset = 0
        while True:
            r = client.get(
                f"{url}/rest/v1/{table}",
                headers={**headers, "Range": f"{offset}-{offset + page_size - 1}"},
                params={"select": "*"},
            )
            if r.status_code >= 400:
                raise RuntimeError(
                    f"Supabase /rest/v1/{table} returned {r.status_code}: {r.text[:200]}"
                )
            chunk = r.json() or []
            out.extend(chunk)
            if len(chunk) < page_size:
                break
            offset += page_size
    return out


def _r2_put_json(bucket: str, key: str, payload: dict | list) -> int:
    """PUT a JSON-serializable payload to R2. Returns body size in bytes."""
    body = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
    client = _get_client()
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json; charset=utf-8",
        # No public access intended for backups — bucket policy controls
        # who can read; we just upload with private ACL by default.
        CacheControl="no-store",
    )
    return len(body)


def _r2_list_prefix(bucket: str, prefix: str) -> list[dict]:
    """List objects under a prefix. Returns [{Key, LastModified}, ...]
    sorted by Key ascending (which is YYYY-MM-DD asc, so oldest first).
    """
    client = _get_client()
    keys: list[dict] = []
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        r = client.list_objects_v2(**kw)
        for obj in r.get("Contents") or []:
            keys.append({"Key": obj["Key"], "LastModified": obj.get("LastModified")})
        if not r.get("IsTruncated"):
            break
        token = r.get("NextContinuationToken")
    keys.sort(key=lambda x: x["Key"])
    return keys


def _r2_delete(bucket: str, key: str) -> None:
    client = _get_client()
    client.delete_object(Bucket=bucket, Key=key)


def run_backup() -> dict:
    """Snapshot every table in _BACKUP_TABLES to R2 + prune old snapshots
    beyond _RETENTION_WEEKS. Returns a structured summary so the caller
    can surface counts to logs / response payload.
    """
    bucket = (os.environ.get("R2_BUCKET") or "").strip()
    if not bucket:
        raise RuntimeError("R2_BUCKET env var not set.")

    today = _dt.date.today().isoformat()
    summary = {"date": today, "tables": [], "pruned": []}

    for table in _BACKUP_TABLES:
        rows = _supabase_select_all(table)
        prefix = f"backups/{table}/"
        key = f"{prefix}{today}.json"
        size = _r2_put_json(bucket, key, rows)
        summary["tables"].append({
            "table": table,
            "key": key,
            "rows": len(rows),
            "bytes": size,
        })

        # Prune: keep only the most recent _RETENTION_WEEKS snapshots
        # under this prefix. Best-effort — a delete failure here doesn't
        # roll back the upload above, since the snapshot is the
        # important part.
        try:
            existing = _r2_list_prefix(bucket, prefix)
            stale = existing[:-_RETENTION_WEEKS] if len(existing) > _RETENTION_WEEKS else []
            for s in stale:
                try:
                    _r2_delete(bucket, s["Key"])
                    summary["pruned"].append(s["Key"])
                except Exception as exc:
                    logger.warning("[backup] prune failed for %s: %s", s["Key"], exc)
        except Exception as exc:
            logger.warning("[backup] list-for-prune failed under %s: %s", prefix, exc)

    return summary
