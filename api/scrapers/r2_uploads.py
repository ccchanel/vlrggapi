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
    # 15-minute window — enough for a 10 GB upload at 100 Mbps and
    # short enough to bound exposure if the URL leaks somehow.
    upload_url = client.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": bucket,
            "Key": object_key,
            "ContentType": content_type,
        },
        ExpiresIn=15 * 60,
        HttpMethod="PUT",
    )

    return {
        "upload_url": upload_url,
        "public_url": f"{public_base}/{object_key}",
        "object_key": object_key,
        "expires_in": 15 * 60,
    }
