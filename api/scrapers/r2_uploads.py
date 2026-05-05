"""
Cloudflare R2 presigned-URL minting for the VOD self-host pipeline.

Admins on the frontend pick a video file and call /v2/vods/upload-url
to get a short-lived (15 min) PUT URL pointing at R2. The browser
uploads directly to that URL; the file body never touches Railway.
After upload, the public r2.dev URL is what we store in vods.video_url
and what the <video> player streams from.

Env vars (set on Railway):
    R2_ACCESS_KEY_ID      — Account API token Access Key ID
    R2_SECRET_ACCESS_KEY  — Account API token Secret Access Key
    R2_ACCOUNT_ID         — Cloudflare account ID
    R2_ENDPOINT           — https://<account>.r2.cloudflarestorage.com
    R2_BUCKET             — bucket name
    R2_PUBLIC_URL         — public base URL (https://pub-xxxx.r2.dev)
"""
import logging
import os
import re
import time
import uuid

logger = logging.getLogger(__name__)

# Lazy import: boto3 is only needed when /v2/vods/upload-url is hit, and
# we don't want to crash startup if R2 envs aren't configured yet.
_boto3_client = None


def _get_client():
    global _boto3_client
    if _boto3_client is not None:
        return _boto3_client
    import boto3
    from botocore.config import Config
    endpoint = os.environ.get("R2_ENDPOINT", "").strip()
    access_key = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
    if not endpoint or not access_key or not secret_key:
        raise RuntimeError(
            "R2 env vars not configured: need R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY"
        )
    # R2 requires SigV4 with the 'auto' region.
    _boto3_client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )
    return _boto3_client


# Allowed video MIME types — we'll only mint URLs for media the
# <video> tag can actually play.
_ALLOWED_MIME = {
    "video/mp4",
    "video/webm",
    "video/quicktime",
    "video/x-matroska",  # mkv (browser support varies, but allow)
}

# Hard cap on file size (10 GB). The user is on R2's free 10 GB tier,
# but a single 10 GB upload is a reasonable upper bound for a full
# 4-hour 1080p VOD.
_MAX_BYTES = 10 * 1024 * 1024 * 1024


def _safe_filename(raw: str) -> str:
    """Strip path traversal, normalise whitespace, keep extension."""
    if not raw:
        return f"vod-{uuid.uuid4().hex}"
    # Drop any path components a malicious uploader might have included
    raw = raw.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    # Replace anything that's not [A-Za-z0-9._-] with _
    raw = re.sub(r"[^A-Za-z0-9._-]", "_", raw)
    if not raw or raw in {".", ".."}:
        raw = f"vod-{uuid.uuid4().hex}.mp4"
    return raw[:128]


def mint_upload_url(filename: str, content_type: str, size_bytes: int) -> dict:
    """Return a presigned PUT URL + the eventual public read URL.

    Object key format:  vods/<unix-ts>-<rand6>-<safe-filename>
    Keeps uploads chronologically sortable and avoids collisions when
    two admins upload at the same time.
    """
    bucket = os.environ.get("R2_BUCKET", "").strip()
    public_base = os.environ.get("R2_PUBLIC_URL", "").strip().rstrip("/")
    if not bucket or not public_base:
        raise RuntimeError("R2_BUCKET and R2_PUBLIC_URL must be set")

    if content_type not in _ALLOWED_MIME:
        raise ValueError(
            f"Unsupported content type '{content_type}'. Use mp4 / webm / mov / mkv."
        )
    if size_bytes <= 0:
        raise ValueError("size_bytes must be positive")
    if size_bytes > _MAX_BYTES:
        raise ValueError(
            f"File too large ({size_bytes} bytes). Max {_MAX_BYTES // (1024*1024*1024)} GB."
        )

    safe = _safe_filename(filename)
    object_key = f"vods/{int(time.time())}-{uuid.uuid4().hex[:6]}-{safe}"

    client = _get_client()
    # 60-minute window. Old comment claimed 15 min was "enough for 10 GB
    # at 100 Mbps" but the math is wrong: 10 GB ≈ 13.3 min at 100 Mbps
    # with zero margin, and most home upload links are 30-50 Mbps. Real
    # case: 3.88 GB upload 400'd because the URL expired mid-PUT.
    # Bumping to 60 min gives a typical 30 Mbps connection room for ~13
    # GB; still short enough that a leaked URL has a small blast radius.
    upload_url = client.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": bucket,
            "Key": object_key,
            "ContentType": content_type,
        },
        ExpiresIn=60 * 60,
        HttpMethod="PUT",
    )

    return {
        "upload_url": upload_url,
        "public_url": f"{public_base}/{object_key}",
        "object_key": object_key,
        "expires_in": 60 * 60,
    }


# ── Multipart upload (for files > R2's 5 GB single-PUT limit) ────────
# R2's PutObject caps a single request at 5 GB. Anything bigger has to
# be uploaded as multiple parts under one UploadId. We mint presigned
# URLs for each part PUT so the file body still never touches Railway.
#
# Flow:
#   1. POST /v2/vods/upload-init → returns {upload_id, object_key,
#      part_size, num_parts, public_url}
#   2. For part N in 1..num_parts:
#        POST /v2/vods/upload-part {object_key, upload_id, part_number}
#          → returns {upload_url}
#        PUT that URL with the part bytes; capture the ETag header
#   3. POST /v2/vods/upload-complete {object_key, upload_id, parts}
#      where parts = [{PartNumber, ETag}, ...]
#
# Part-size policy: each part must be ≥5 MiB except the final one, and
# we have a hard cap of 10,000 parts per upload. Default to 32 MiB
# parts which covers ~320 GB before bumping into the 10K cap.

_MULTIPART_THRESHOLD = 4 * 1024 * 1024 * 1024  # 4 GiB — R2 single-PUT
                                                # cap is 5 GB, leaving
                                                # 1 GB margin so a
                                                # progress-bar overshoot
                                                # never trips it
_DEFAULT_PART_SIZE = 32 * 1024 * 1024           # 32 MiB
_MIN_PART_SIZE = 5 * 1024 * 1024                # R2/S3 minimum
_MAX_PARTS = 10_000                              # R2/S3 max parts


def should_use_multipart(size_bytes: int) -> bool:
    return size_bytes >= _MULTIPART_THRESHOLD


def init_multipart_upload(filename: str, content_type: str, size_bytes: int) -> dict:
    """Start a multipart upload and return the metadata the client needs
    to drive the per-part PUTs."""
    bucket = os.environ.get("R2_BUCKET", "").strip()
    public_base = os.environ.get("R2_PUBLIC_URL", "").strip().rstrip("/")
    if not bucket or not public_base:
        raise RuntimeError("R2_BUCKET and R2_PUBLIC_URL must be set")
    if content_type not in _ALLOWED_MIME:
        raise ValueError(
            f"Unsupported content type '{content_type}'. Use mp4 / webm / mov / mkv."
        )
    if size_bytes <= 0:
        raise ValueError("size_bytes must be positive")
    if size_bytes > _MAX_BYTES:
        raise ValueError(
            f"File too large ({size_bytes} bytes). Max {_MAX_BYTES // (1024*1024*1024)} GB."
        )

    # Pick a part size that keeps num_parts under _MAX_PARTS, never goes
    # below _MIN_PART_SIZE, and defaults to _DEFAULT_PART_SIZE for the
    # common case.
    part_size = _DEFAULT_PART_SIZE
    if size_bytes // part_size > _MAX_PARTS:
        # Round up to the next MB boundary
        part_size = max(_MIN_PART_SIZE, ((size_bytes // _MAX_PARTS) + 1024 * 1024 - 1) & ~(1024 * 1024 - 1))

    num_parts = (size_bytes + part_size - 1) // part_size

    safe = _safe_filename(filename)
    object_key = f"vods/{int(time.time())}-{uuid.uuid4().hex[:6]}-{safe}"

    client = _get_client()
    resp = client.create_multipart_upload(
        Bucket=bucket,
        Key=object_key,
        ContentType=content_type,
    )

    return {
        "upload_id": resp["UploadId"],
        "object_key": object_key,
        "public_url": f"{public_base}/{object_key}",
        "part_size": part_size,
        "num_parts": num_parts,
    }


def mint_part_upload_url(object_key: str, upload_id: str, part_number: int) -> dict:
    """Presigned PUT URL for a single part of an in-progress multipart
    upload. Same 60-min window as the single-PUT path."""
    bucket = os.environ.get("R2_BUCKET", "").strip()
    if not bucket:
        raise RuntimeError("R2_BUCKET must be set")
    if not object_key or not upload_id:
        raise ValueError("object_key and upload_id are required")
    if part_number < 1 or part_number > _MAX_PARTS:
        raise ValueError(f"part_number must be 1..{_MAX_PARTS}")

    client = _get_client()
    upload_url = client.generate_presigned_url(
        "upload_part",
        Params={
            "Bucket": bucket,
            "Key": object_key,
            "UploadId": upload_id,
            "PartNumber": part_number,
        },
        ExpiresIn=60 * 60,
        HttpMethod="PUT",
    )
    return {"upload_url": upload_url}


def complete_multipart_upload(object_key: str, upload_id: str, parts: list) -> dict:
    """Finalize the multipart upload. `parts` is a list of
    {PartNumber, ETag} dicts in part-number order."""
    bucket = os.environ.get("R2_BUCKET", "").strip()
    public_base = os.environ.get("R2_PUBLIC_URL", "").strip().rstrip("/")
    if not bucket or not public_base:
        raise RuntimeError("R2_BUCKET and R2_PUBLIC_URL must be set")
    if not object_key or not upload_id:
        raise ValueError("object_key and upload_id are required")
    if not parts:
        raise ValueError("parts list is empty")

    # Defensive sort + shape check — boto3 demands ascending PartNumber
    # and complains noisily if ETags don't include the surrounding quotes
    # that S3/R2 returns by default.
    cleaned = []
    for p in parts:
        try:
            n = int(p["PartNumber"])
            etag = str(p["ETag"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Bad part entry {p}: {exc}")
        if not etag.startswith('"'):
            etag = f'"{etag}"'
        cleaned.append({"PartNumber": n, "ETag": etag})
    cleaned.sort(key=lambda x: x["PartNumber"])

    client = _get_client()
    client.complete_multipart_upload(
        Bucket=bucket,
        Key=object_key,
        UploadId=upload_id,
        MultipartUpload={"Parts": cleaned},
    )
    return {"public_url": f"{public_base}/{object_key}"}


def abort_multipart_upload(object_key: str, upload_id: str) -> dict:
    """Best-effort abort so a cancelled upload doesn't leave R2
    accruing storage for orphaned parts."""
    bucket = os.environ.get("R2_BUCKET", "").strip()
    if not bucket:
        raise RuntimeError("R2_BUCKET must be set")
    client = _get_client()
    client.abort_multipart_upload(
        Bucket=bucket,
        Key=object_key,
        UploadId=upload_id,
    )
    return {"aborted": True}


# ── Team-logo mirror (server-side upload) ─────────────────────────────
# Solves the recurring "owcdn.net logos go dead" problem. VLR serves
# logos from owcdn.net, which replaces / removes assets when teams
# rebrand. We mirror each unique logo URL to R2 once, then frontend
# routes through R2 forever. Uploads go server-side here (not via
# presigned URL) because (a) the byte payload is tiny (~50 KB), and
# (b) we control the source URL, so SSRF / size-bound concerns of
# user-uploaded content don't apply.

import hashlib

_LOGO_PREFIX = "team-logos/"
# Hard cap per logo so a misclassified URL pointing at a huge file
# doesn't blow up R2 storage or hold the request open.
_MAX_LOGO_BYTES = 5 * 1024 * 1024  # 5 MB
# Allowed extensions / content types — VLR logos are PNG / WebP / JPG
# in practice. SVG is allowed too since some Liquipedia fallbacks ship
# SVG; we serve them with image/svg+xml.
_LOGO_EXT_BY_CT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/svg+xml": ".svg",
    "image/gif": ".gif",
}


def _logo_ext_from(url: str, content_type: str) -> str:
    """Pick a file extension. Prefer content-type, fall back to URL."""
    ct = (content_type or "").split(";", 1)[0].strip().lower()
    if ct in _LOGO_EXT_BY_CT:
        return _LOGO_EXT_BY_CT[ct]
    # Fall back to the URL extension if it's recognisable.
    m = re.search(r"\.(png|jpe?g|webp|svg|gif)(?:[?#]|$)", url, flags=re.I)
    if m:
        ext = m.group(1).lower()
        if ext == "jpeg":
            ext = "jpg"
        return f".{ext}"
    return ".png"


def mirror_logo_to_r2(vlr_url: str, image_bytes: bytes, content_type: str) -> dict:
    """Upload a single team logo to R2. Returns a dict ready to upsert
    into the public.team_logo_mirror Supabase table.

    Raises ValueError on size limit, RuntimeError on R2 misconfig.
    The caller is responsible for fetching `image_bytes` from VLR's
    CDN (so we don't have to recreate the proxy / cookie logic here).
    """
    if not vlr_url:
        raise ValueError("vlr_url required")
    if not image_bytes:
        raise ValueError("image_bytes empty")
    if len(image_bytes) > _MAX_LOGO_BYTES:
        raise ValueError(
            f"Logo too large ({len(image_bytes)} bytes, max {_MAX_LOGO_BYTES})"
        )

    bucket = os.environ.get("R2_BUCKET", "").strip()
    public_base = os.environ.get("R2_PUBLIC_URL", "").strip().rstrip("/")
    if not bucket or not public_base:
        raise RuntimeError("R2_BUCKET and R2_PUBLIC_URL must be set")

    # Stable key from URL hash so identical URLs across calls dedupe.
    digest = hashlib.md5(vlr_url.encode("utf-8")).hexdigest()
    ext = _logo_ext_from(vlr_url, content_type)
    object_key = f"{_LOGO_PREFIX}{digest}{ext}"

    # Pick a sensible Content-Type for the upload — mirror what we
    # downloaded, fall back to image/png.
    upload_ct = (content_type or "").split(";", 1)[0].strip().lower()
    if upload_ct not in _LOGO_EXT_BY_CT:
        upload_ct = "image/png"

    client = _get_client()
    client.put_object(
        Bucket=bucket,
        Key=object_key,
        Body=image_bytes,
        ContentType=upload_ct,
        # CacheControl: 30 days — logos are rarely re-cut, and our
        # key is derived from the source URL so an updated logo gets
        # a fresh URL anyway.
        CacheControl="public, max-age=2592000, immutable",
    )

    return {
        "vlr_url": vlr_url,
        "r2_key": object_key,
        "r2_url": f"{public_base}/{object_key}",
        "content_type": upload_ct,
        "size_bytes": len(image_bytes),
    }
