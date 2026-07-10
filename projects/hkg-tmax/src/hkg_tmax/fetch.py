from __future__ import annotations

import mimetypes
import ssl
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import httpx

from .storage import RawSnapshot, store_raw_bytes


class FetchError(RuntimeError):
    """Raised when a remote source cannot be safely archived."""


@dataclass(frozen=True)
class FetchPolicy:
    timeout_seconds: float = 60.0
    user_agent: str = "HKG-Tmax-Research/0.1"
    follow_redirects: bool = True
    max_bytes: int = 512 * 1024 * 1024
    max_attempts: int = 1
    retry_sleep_seconds: float = 0.0


def httpx_verify_context() -> bool | ssl.SSLContext:
    """Use OS trust roots when truststore is installed, while keeping TLS verification on."""

    try:
        truststore: Any = import_module("truststore")
    except ModuleNotFoundError:
        return True
    return cast(ssl.SSLContext, truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT))


_SCRIPT_SUFFIXES = {"php", "aspx", "asp", "jsp", "cgi", "do", "action", "html", "htm"}


def infer_extension(url: str, content_type: str | None) -> str:
    clean_type = (content_type or "").split(";", 1)[0].strip().lower()
    known = {
        "application/json": "json",
        "application/geo+json": "geojson",
        "text/csv": "csv",
        "application/csv": "csv",
        "text/plain": "txt",
        "text/html": "html",
        "application/pdf": "pdf",
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/gif": "gif",
        "application/x-netcdf": "nc",
        "application/netcdf": "nc",
        "application/x-grib": "grib",
        "application/grib": "grib",
        "application/zip": "zip",
        "application/gzip": "gz",
    }
    if clean_type in known:
        return known[clean_type]

    suffix = Path(urlparse(url).path).suffix.lower().lstrip(".")
    if suffix and suffix not in _SCRIPT_SUFFIXES:
        return suffix
    guessed = mimetypes.guess_extension(clean_type) if clean_type else None
    return (guessed or ".bin").lstrip(".")


def _redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    protected = {"authorization", "cookie", "x-api-key", "api-key"}
    return {
        key: ("<redacted>" if key.lower() in protected else value)
        for key, value in headers.items()
    }


def fetch_and_archive(
    *,
    url: str,
    source_id: str,
    raw_root: Path,
    policy: FetchPolicy | None = None,
    request_headers: Mapping[str, str] | None = None,
) -> RawSnapshot:
    policy = policy or FetchPolicy()
    if policy.max_attempts < 1:
        raise ValueError("FetchPolicy.max_attempts must be >= 1")
    if policy.retry_sleep_seconds < 0:
        raise ValueError("FetchPolicy.retry_sleep_seconds must be >= 0")
    headers = {"User-Agent": policy.user_agent, **dict(request_headers or {})}

    response: httpx.Response | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            with httpx.Client(
                timeout=policy.timeout_seconds,
                follow_redirects=policy.follow_redirects,
                headers=headers,
                verify=httpx_verify_context(),
            ) as client:
                response = client.get(url)
            break
        except httpx.HTTPError as exc:
            if attempt == policy.max_attempts:
                raise FetchError(
                    f"Request failed for {source_id} at {url} "
                    f"after {attempt} attempt(s): {exc}"
                ) from exc
            if policy.retry_sleep_seconds > 0:
                time.sleep(policy.retry_sleep_seconds)
    if response is None:
        raise FetchError(f"Request failed for {source_id} at {url}: no response")

    retrieved_at = datetime.now(UTC)
    if response.status_code < 200 or response.status_code >= 300:
        raise FetchError(
            f"HTTP {response.status_code} for {source_id} at {url}; "
            "non-success payload was not accepted as source data"
        )

    content = response.content
    if not content:
        raise FetchError(f"Empty payload for {source_id} at {url}")
    if len(content) > policy.max_bytes:
        raise FetchError(
            f"Payload exceeds max_bytes for {source_id}: {len(content)} > {policy.max_bytes}"
        )

    content_type = response.headers.get("content-type")
    extension = infer_extension(str(response.url), content_type)
    metadata = {
        "requested_url": url,
        "final_url": str(response.url),
        "request_method": "GET",
        "request_headers": _redact_headers(headers),
        "http_status": response.status_code,
        "response_headers": dict(response.headers),
        "content_type": content_type,
        "extension_inferred": extension,
    }
    return store_raw_bytes(
        raw_root,
        source_id=source_id,
        content=content,
        retrieved_at=retrieved_at,
        extension=extension,
        metadata=metadata,
    )
