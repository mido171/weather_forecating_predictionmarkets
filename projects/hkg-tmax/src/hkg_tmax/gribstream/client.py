"""Secret-safe, one-thread GribStream HTTP client."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable

import httpx


TRANSIENT_HTTP_STATUS = {429, 500, 502, 503, 504}
PERMANENT_HTTP_STATUS = {400, 401, 403, 404}
QUERY_ACCEPT_NDJSON = "application/ndjson"


def canonical_request_json(payload: dict[str, Any]) -> str:
    """Return stable JSON used for request identity."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def request_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_request_json(payload).encode("utf-8")).hexdigest()


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_now_iso() -> str:
    return utc_now().isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(fs_path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fs_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name != "nt" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved.lstrip("\\")
    return "\\\\?\\" + resolved


def ensure_directory(path: Path) -> None:
    os.makedirs(fs_path(path), exist_ok=True)


def append_jsonl(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    ensure_directory(path.parent)
    with open(fs_path(path), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")


def parse_retry_after(value: str | None) -> float | None:
    """Parse Retry-After seconds or HTTP-date values."""

    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return max(float(text), 0.0)
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max((parsed.astimezone(UTC) - utc_now()).total_seconds(), 0.0)


def sanitize_text(value: str, token: str | None = None, *, limit: int = 240) -> str:
    sanitized = value.replace("\r", " ").replace("\n", " ")[:limit]
    if token:
        sanitized = sanitized.replace(token, "[REDACTED_GRIBSTREAM_API_KEY]")
    return sanitized


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 3
    min_interval_seconds: float = 12.0
    default_rate_limit_pause_seconds: float = 300.0
    min_rate_limit_pause_seconds: float = 180.0
    max_retry_delay_seconds: float = 1800.0
    transient_base_delay_seconds: float = 5.0
    transient_max_delay_seconds: float = 60.0


@dataclass(frozen=True)
class ResponseManifest:
    provider: str
    dataset: str
    endpoint: str
    request_sha256: str
    object_path: Path
    byte_size: int
    sha256: str
    content_type: str
    retrieved_at_utc: str
    row_count: int
    http_status: int
    attempt_count: int


class GribStreamRequestError(RuntimeError):
    """Raised after a request fails under the retry policy."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        attempt_count: int = 0,
        error_class: str = "GribStreamRequestError",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.attempt_count = attempt_count
        self.error_class = error_class


class OneThreadRateLimiter:
    """Process-local limiter for single-threaded GribStream access."""

    def __init__(
        self,
        min_interval_seconds: float,
        *,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.min_interval_seconds = max(min_interval_seconds, 0.0)
        self._sleeper = sleeper
        self._monotonic = monotonic
        self._next_allowed_at = 0.0

    def wait(self) -> float:
        now = self._monotonic()
        delay = max(self._next_allowed_at - now, 0.0)
        if delay > 0:
            self._sleeper(delay)
        self._next_allowed_at = self._monotonic() + self.min_interval_seconds
        return delay


def retry_delay_seconds(
    *,
    status_code: int | None,
    retry_after: str | None,
    attempt_number: int,
    config: RetryConfig,
    rng: random.Random | None = None,
) -> float:
    """Compute a bounded retry delay; 429 defaults to a 3-5 minute pause."""

    if status_code == 429:
        parsed = parse_retry_after(retry_after)
        delay = parsed if parsed is not None else config.default_rate_limit_pause_seconds
        delay = max(delay, config.min_rate_limit_pause_seconds)
        return min(delay, config.max_retry_delay_seconds)
    jitter = (rng or random).uniform(0.0, 1.0)
    exponential = config.transient_base_delay_seconds * (2 ** max(attempt_number - 1, 0))
    return min(exponential + jitter, config.transient_max_delay_seconds, config.max_retry_delay_seconds)


class GribStreamClient:
    """Direct HTTP client that stores GribStream query responses as NDJSON gzip."""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = "https://gribstream.com",
        retry_config: RetryConfig | None = None,
        http_client: httpx.Client | None = None,
        event_log_path: Path | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not token:
            raise ValueError("GRIBSTREAM_API_KEY is required for authenticated queries")
        self._token = token
        self.base_url = base_url.rstrip("/")
        self.retry_config = retry_config or RetryConfig()
        self._client = http_client or httpx.Client(timeout=httpx.Timeout(30.0, read=120.0))
        self._owns_client = http_client is None
        self._limiter = OneThreadRateLimiter(
            self.retry_config.min_interval_seconds,
            sleeper=sleeper,
        )
        self._event_log_path = event_log_path
        self._sleeper = sleeper

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> GribStreamClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def post_runs_to_gzip(
        self,
        *,
        dataset: str,
        payload: dict[str, Any],
        output_path: Path,
        request_hash: str | None = None,
        accept: str = QUERY_ACCEPT_NDJSON,
    ) -> ResponseManifest:
        """POST a `/runs` request and write one immutable NDJSON gzip object."""

        req_hash = request_hash or request_sha256(payload)
        url = f"{self.base_url}/api/v2/{dataset}/runs"
        ensure_directory(output_path.parent)
        part_path = output_path.with_suffix(output_path.suffix + ".part")
        if os.path.exists(fs_path(part_path)):
            os.unlink(fs_path(part_path))
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": accept,
            "Accept-Encoding": "gzip",
        }
        last_error: GribStreamRequestError | None = None
        max_attempts = max(self.retry_config.max_attempts, 1)
        for attempt in range(1, max_attempts + 1):
            rate_wait = self._limiter.wait()
            started = time.perf_counter()
            try:
                response = self._client.post(url, json=payload, headers=headers)
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                last_error = GribStreamRequestError(
                    sanitize_text(str(exc), self._token),
                    attempt_count=attempt,
                    error_class=type(exc).__name__,
                )
                append_jsonl(
                    self._event_log_path,
                    {
                        "event": "network_exception",
                        "request_sha256": req_hash,
                        "attempt": attempt,
                        "elapsed_ms": elapsed_ms,
                        "rate_wait_seconds": round(rate_wait, 3),
                        "error_class": type(exc).__name__,
                        "error_message": sanitize_text(str(exc), self._token),
                        "updated_at_utc": utc_now_iso(),
                    },
                )
                self._sleep_before_retry(None, None, attempt)
                continue

            elapsed_ms = int((time.perf_counter() - started) * 1000)
            content_type = response.headers.get("content-type", "")
            append_jsonl(
                self._event_log_path,
                {
                    "event": "http_response",
                    "request_sha256": req_hash,
                    "attempt": attempt,
                    "http_status": response.status_code,
                    "content_type": content_type,
                    "elapsed_ms": elapsed_ms,
                    "rate_wait_seconds": round(rate_wait, 3),
                    "retry_after": response.headers.get("retry-after", ""),
                    "updated_at_utc": utc_now_iso(),
                },
            )
            if response.status_code == 200:
                row_count = self._write_response_lines(response, part_path)
                os.replace(fs_path(part_path), fs_path(output_path))
                return ResponseManifest(
                    provider="GribStream",
                    dataset=dataset,
                    endpoint="runs",
                    request_sha256=req_hash,
                    object_path=output_path,
                    byte_size=os.path.getsize(fs_path(output_path)),
                    sha256=sha256_file(output_path),
                    content_type=content_type or accept,
                    retrieved_at_utc=utc_now_iso(),
                    row_count=row_count,
                    http_status=response.status_code,
                    attempt_count=attempt,
                )

            body = sanitize_text(response.text, self._token)
            if response.status_code in PERMANENT_HTTP_STATUS:
                raise GribStreamRequestError(
                    f"Permanent GribStream HTTP {response.status_code}: {body}",
                    status_code=response.status_code,
                    attempt_count=attempt,
                    error_class=f"HTTP_{response.status_code}",
                )
            if response.status_code not in TRANSIENT_HTTP_STATUS:
                raise GribStreamRequestError(
                    f"Unhandled GribStream HTTP {response.status_code}: {body}",
                    status_code=response.status_code,
                    attempt_count=attempt,
                    error_class=f"HTTP_{response.status_code}",
                )
            last_error = GribStreamRequestError(
                f"Transient GribStream HTTP {response.status_code}: {body}",
                status_code=response.status_code,
                attempt_count=attempt,
                error_class=f"HTTP_{response.status_code}",
            )
            self._sleep_before_retry(response.status_code, response.headers.get("retry-after"), attempt)

        if last_error is not None:
            raise last_error
        raise GribStreamRequestError("GribStream request failed without response", attempt_count=max_attempts)

    def _write_response_lines(self, response: httpx.Response, output_path: Path) -> int:
        row_count = 0
        ensure_directory(output_path.parent)
        with open(fs_path(output_path), "wb") as raw_handle:
            with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as gzip_handle:
                for line in response.iter_lines():
                    if not line:
                        continue
                    stripped = line.strip()
                    if not stripped:
                        continue
                    gzip_handle.write(stripped.encode("utf-8"))
                    gzip_handle.write(b"\n")
                    row_count += 1
        return row_count

    def _sleep_before_retry(
        self,
        status_code: int | None,
        retry_after: str | None,
        attempt: int,
    ) -> None:
        if attempt >= max(self.retry_config.max_attempts, 1):
            return
        delay = retry_delay_seconds(
            status_code=status_code,
            retry_after=retry_after,
            attempt_number=attempt,
            config=self.retry_config,
        )
        append_jsonl(
            self._event_log_path,
            {
                "event": "retry_sleep",
                "attempt": attempt,
                "http_status": status_code,
                "sleep_seconds": round(delay, 3),
                "updated_at_utc": utc_now_iso(),
            },
        )
        self._sleeper(delay)
