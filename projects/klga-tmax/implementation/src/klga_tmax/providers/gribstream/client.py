from __future__ import annotations

from datetime import datetime, timezone
import gzip
import json
import random
import threading
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from klga_tmax.providers.gribstream.config import GribStreamSettings
from klga_tmax.providers.gribstream.models import GribStreamChunk, GribStreamRawResponse

PARSER_VERSION = "gribstream_single_cutoff_v2"
RETRYABLE_STATUS_CODES = {500, 502, 503, 504}
STOP_STATUS_CODES = {401, 403, 429}


class GribStreamRequestError(RuntimeError):
    def __init__(self, response: GribStreamRawResponse) -> None:
        super().__init__(response.error_message or response.error_type or "GribStream request failed")
        self.response = response


class OneThreadRateLimiter:
    def __init__(self, *, spacing_seconds: float) -> None:
        self.spacing_seconds = max(0.0, spacing_seconds)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self._next_allowed - now)
            if wait_seconds:
                time.sleep(wait_seconds)
                now = time.monotonic()
            self._next_allowed = max(now, self._next_allowed) + self.spacing_seconds


def retry_after_seconds(headers: Any) -> float | None:
    raw = headers.get("Retry-After") if hasattr(headers, "get") else None
    if not raw:
        return None
    raw = str(raw).strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        retry_dt = parsedate_to_datetime(raw)
    except Exception:
        return None
    if retry_dt.tzinfo is None:
        retry_dt = retry_dt.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_dt.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds())


def _raw_path(settings: GribStreamSettings, chunk: GribStreamChunk) -> Path:
    raw_root = settings.artifact_root / "raw"
    if chunk.endpoint_type != "timeseries":
        raw_root = raw_root / chunk.endpoint_type
    return (
        raw_root
        / chunk.model_id
        / f"{chunk.target_start_date.isoformat()}_{chunk.target_end_date.isoformat()}_{chunk.request_sha256[:12]}.ndjson.gz"
    )


def _write_gzip_atomic(path: Path, body: bytes) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    with gzip.GzipFile(filename="", mode="wb", fileobj=tmp_path.open("wb"), mtime=0) as gzip_file:
        gzip_file.write(body)
    tmp_path.replace(path)
    return path.stat().st_size


class GribStreamTimeseriesClient:
    def __init__(
        self,
        settings: GribStreamSettings,
        *,
        rate_limiter: OneThreadRateLimiter | None = None,
    ) -> None:
        if not settings.api_token:
            raise ValueError("GribStream API token is required")
        self.settings = settings
        self.rate_limiter = rate_limiter or OneThreadRateLimiter(
            spacing_seconds=settings.spacing_seconds
        )

    def fetch_chunk(self, chunk: GribStreamChunk) -> GribStreamRawResponse:
        url = f"{self.settings.base_url.rstrip('/')}/{chunk.model_id}/{chunk.endpoint_type}"
        headers = {
            "Accept": "application/ndjson",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.api_token}",
            "User-Agent": self.settings.user_agent,
        }
        attempts = max(1, self.settings.max_retries + 1)
        last_response: GribStreamRawResponse | None = None
        raw_path = _raw_path(self.settings, chunk)
        redacted_url = url
        body_bytes = json.dumps(chunk.request_payload, sort_keys=True).encode("utf-8")
        for attempt in range(1, attempts + 1):
            self.rate_limiter.acquire()
            retrieved_at = datetime.now(timezone.utc)
            request = Request(url, data=body_bytes, method="POST", headers=headers)
            try:
                with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                    body = response.read()
                    content_type = response.headers.get("Content-Type")
                    status = int(response.status)
                    _write_gzip_atomic(raw_path, body)
                    return GribStreamRawResponse(
                        chunk=chunk,
                        endpoint_url_redacted=redacted_url,
                        retrieved_at_utc=retrieved_at,
                        http_status=status,
                        content_type=content_type,
                        response_body_sha256=response.headers.get("ETag", "").strip('"') or _sha256_bytes(body),
                        response_size_bytes=len(body),
                        raw_storage_uri=str(raw_path),
                        attempts=attempt,
                    )
            except HTTPError as exc:
                body = exc.read()
                content_type = exc.headers.get("Content-Type")
                status = int(exc.code)
                result = GribStreamRawResponse(
                    chunk=chunk,
                    endpoint_url_redacted=redacted_url,
                    retrieved_at_utc=retrieved_at,
                    http_status=status,
                    content_type=content_type,
                    response_body_sha256=_sha256_bytes(body),
                    response_size_bytes=len(body),
                    raw_storage_uri=str(raw_path),
                    attempts=attempt,
                    error_type=f"HTTP_{status}",
                    error_message=body.decode("utf-8", errors="replace")[:2000],
                )
                if status in STOP_STATUS_CODES:
                    retry_after = retry_after_seconds(exc.headers)
                    if retry_after:
                        time.sleep(retry_after)
                    raise GribStreamRequestError(result)
                if status not in RETRYABLE_STATUS_CODES or attempt == attempts:
                    raise GribStreamRequestError(result)
                last_response = result
                _sleep_before_retry(attempt, retry_after_seconds(exc.headers))
            except (TimeoutError, URLError, OSError) as exc:
                result = GribStreamRawResponse(
                    chunk=chunk,
                    endpoint_url_redacted=redacted_url,
                    retrieved_at_utc=retrieved_at,
                    http_status=None,
                    content_type=None,
                    response_body_sha256=_sha256_bytes(b""),
                    response_size_bytes=0,
                    raw_storage_uri=str(raw_path),
                    attempts=attempt,
                    error_type="TRANSPORT_ERROR",
                    error_message=str(exc),
                )
                if attempt == attempts:
                    raise GribStreamRequestError(result) from exc
                last_response = result
                _sleep_before_retry(attempt, None)
        if last_response is not None:
            raise GribStreamRequestError(last_response)
        raise RuntimeError("GribStream request loop exited without a response")


def _sha256_bytes(body: bytes) -> str:
    from klga_tmax.ingestion.hash_keys import sha256_hex

    return sha256_hex(body)


def _sleep_before_retry(attempt: int, retry_after: float | None) -> None:
    sleep_seconds = retry_after if retry_after is not None else min(60.0, 2.0 * (2 ** max(0, attempt - 1)) + random.uniform(0.0, 0.5))
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
