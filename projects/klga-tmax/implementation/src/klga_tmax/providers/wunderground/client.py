from __future__ import annotations

from datetime import date, datetime, timezone
import gzip
import json
import random
import threading
import time
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from klga_tmax.ingestion.hash_keys import sha256_hex
from klga_tmax.providers.wunderground.config import WundergroundSettings
from klga_tmax.providers.wunderground.models import Units, WundergroundRawDayResponse

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
PERMANENT_STATUS_CODES = {400, 401, 403, 404}
PARSER_VERSION = "weathercom_historical_observations_v1"


class RateLimiter:
    """Small process-local limiter shared by WU worker threads."""

    def __init__(self, *, permits_per_minute: int) -> None:
        self._interval_seconds = 60.0 / max(1, permits_per_minute)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self._next_allowed - now)
            if wait_seconds:
                time.sleep(wait_seconds)
                now = time.monotonic()
            self._next_allowed = max(now, self._next_allowed) + self._interval_seconds


def weathercom_location_id(wunderground_station_id: str) -> str:
    normalized = wunderground_station_id.strip().upper()
    if not normalized:
        raise ValueError("wunderground_station_id is required")
    if ":" in normalized:
        return normalized
    return f"{normalized}:9:US"


def build_weathercom_url(
    *,
    base_url: str,
    weathercom_location_id_value: str,
    api_key: str,
    units: str,
    start_date: date,
    end_date: date,
) -> str:
    query = urlencode(
        {
            "apiKey": api_key,
            "units": units,
            "startDate": start_date.strftime("%Y%m%d"),
            "endDate": end_date.strftime("%Y%m%d"),
        }
    )
    return (
        f"{base_url.rstrip('/')}/v1/location/{weathercom_location_id_value}"
        f"/observations/historical.json?{query}"
    )


def _redact_precisely(url: str) -> str:
    if "apiKey=" not in url:
        return url
    prefix, rest = url.split("apiKey=", 1)
    if "&" in rest:
        _, suffix = rest.split("&", 1)
        return f"{prefix}apiKey=REDACTED&{suffix}"
    return f"{prefix}apiKey=REDACTED"


def _retry_after_seconds(headers: Any) -> float | None:
    raw = None
    try:
        raw = headers.get("Retry-After")
    except AttributeError:
        return None
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


def _decode_body(body_bytes: bytes, headers: Any) -> str:
    encoding = ""
    try:
        encoding = headers.get("Content-Encoding", "")
    except AttributeError:
        encoding = ""
    if encoding and encoding.lower() == "gzip":
        body_bytes = gzip.decompress(body_bytes)
    return body_bytes.decode("utf-8", errors="replace")


def _content_type(headers: Any) -> str | None:
    try:
        return headers.get("Content-Type")
    except AttributeError:
        return None


def _json_payload(body_text: str) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if not body_text:
        return None, "EMPTY_BODY", "provider returned an empty response body"
    try:
        payload = json.loads(body_text)
    except json.JSONDecodeError as exc:
        return None, "JSON_PARSE_ERROR", str(exc)
    if not isinstance(payload, dict):
        return None, "JSON_NOT_OBJECT", "provider JSON response was not an object"
    return payload, None, None


class WundergroundHistoricalClient:
    def __init__(self, settings: WundergroundSettings, *, rate_limiter: RateLimiter | None = None) -> None:
        if not settings.api_key:
            raise ValueError("Wunderground API key is required")
        self.settings = settings
        self.rate_limiter = rate_limiter or RateLimiter(
            permits_per_minute=settings.rate_limit_per_minute
        )

    def fetch_station_day(
        self,
        station_id: str,
        local_date: date,
        units: Units = "e",
    ) -> WundergroundRawDayResponse:
        return self.fetch_station_range(
            station_id=station_id,
            start_local_date=local_date,
            end_local_date=local_date,
            units=units,
        )

    def fetch_station_range(
        self,
        station_id: str,
        start_local_date: date,
        end_local_date: date,
        units: Units = "e",
    ) -> WundergroundRawDayResponse:
        if end_local_date < start_local_date:
            raise ValueError("end_local_date must be on or after start_local_date")
        if units not in {"e", "m"}:
            raise ValueError("units must be 'e' or 'm'")
        location_id = weathercom_location_id(station_id)
        url = build_weathercom_url(
            base_url=self.settings.base_url,
            weathercom_location_id_value=location_id,
            api_key=self.settings.api_key or "",
            units=units,
            start_date=start_local_date,
            end_date=end_local_date,
        )
        return self._fetch_url(
            station_id=station_id,
            wunderground_station_id=station_id,
            weathercom_location_id_value=location_id,
            start_local_date=start_local_date,
            end_local_date=end_local_date,
            units=units,
            url=url,
        )

    def fetch_registry_station_range(
        self,
        *,
        canonical_station_id: str,
        wunderground_station_id: str,
        start_local_date: date,
        end_local_date: date,
        units: Units = "e",
    ) -> WundergroundRawDayResponse:
        if end_local_date < start_local_date:
            raise ValueError("end_local_date must be on or after start_local_date")
        if units not in {"e", "m"}:
            raise ValueError("units must be 'e' or 'm'")
        location_id = weathercom_location_id(wunderground_station_id)
        url = build_weathercom_url(
            base_url=self.settings.base_url,
            weathercom_location_id_value=location_id,
            api_key=self.settings.api_key or "",
            units=units,
            start_date=start_local_date,
            end_date=end_local_date,
        )
        return self._fetch_url(
            station_id=canonical_station_id,
            wunderground_station_id=wunderground_station_id,
            weathercom_location_id_value=location_id,
            start_local_date=start_local_date,
            end_local_date=end_local_date,
            units=units,
            url=url,
        )

    def _fetch_url(
        self,
        *,
        station_id: str,
        wunderground_station_id: str,
        weathercom_location_id_value: str,
        start_local_date: date,
        end_local_date: date,
        units: Units,
        url: str,
    ) -> WundergroundRawDayResponse:
        attempts = max(1, self.settings.max_retries + 1)
        last_response: WundergroundRawDayResponse | None = None
        redacted_url = _redact_precisely(url)

        for attempt in range(1, attempts + 1):
            self.rate_limiter.acquire()
            retrieved_at = datetime.now(timezone.utc)
            request = Request(
                url,
                method="GET",
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "User-Agent": self.settings.user_agent,
                },
            )
            try:
                with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                    body_bytes = response.read()
                    body_text = _decode_body(body_bytes, response.headers)
                    payload, parse_error_type, parse_error_message = _json_payload(body_text)
                    status = int(response.status)
                    result = WundergroundRawDayResponse(
                        station_id=station_id,
                        wunderground_station_id=wunderground_station_id,
                        weathercom_location_id=weathercom_location_id_value,
                        start_local_date=start_local_date,
                        end_local_date=end_local_date,
                        units=units,
                        endpoint_url_redacted=redacted_url,
                        retrieved_at_utc=retrieved_at,
                        http_status=status,
                        content_type=_content_type(response.headers),
                        response_body_text=body_text,
                        response_body_sha256=sha256_hex(body_text),
                        response_size_bytes=len(body_bytes),
                        payload_json=payload,
                        attempts=attempt,
                        error_type=parse_error_type,
                        error_message=parse_error_message,
                    )
                    if result.success or status in PERMANENT_STATUS_CODES or attempt == attempts:
                        return result
                    last_response = result
            except HTTPError as exc:
                body_bytes = exc.read()
                body_text = _decode_body(body_bytes, exc.headers)
                payload, parse_error_type, parse_error_message = _json_payload(body_text)
                status = int(exc.code)
                error_type = parse_error_type or f"HTTP_{status}"
                error_message = parse_error_message or body_text[:1000]
                result = WundergroundRawDayResponse(
                    station_id=station_id,
                    wunderground_station_id=wunderground_station_id,
                    weathercom_location_id=weathercom_location_id_value,
                    start_local_date=start_local_date,
                    end_local_date=end_local_date,
                    units=units,
                    endpoint_url_redacted=redacted_url,
                    retrieved_at_utc=retrieved_at,
                    http_status=status,
                    content_type=_content_type(exc.headers),
                    response_body_text=body_text,
                    response_body_sha256=sha256_hex(body_text),
                    response_size_bytes=len(body_bytes),
                    payload_json=payload,
                    attempts=attempt,
                    error_type=error_type,
                    error_message=error_message,
                )
                if status in PERMANENT_STATUS_CODES or status not in RETRYABLE_STATUS_CODES or attempt == attempts:
                    return result
                last_response = result
                retry_after = _retry_after_seconds(exc.headers)
                self._sleep_before_retry(attempt, retry_after)
                continue
            except (TimeoutError, URLError, OSError) as exc:
                result = WundergroundRawDayResponse(
                    station_id=station_id,
                    wunderground_station_id=wunderground_station_id,
                    weathercom_location_id=weathercom_location_id_value,
                    start_local_date=start_local_date,
                    end_local_date=end_local_date,
                    units=units,
                    endpoint_url_redacted=redacted_url,
                    retrieved_at_utc=retrieved_at,
                    http_status=None,
                    content_type=None,
                    response_body_text="",
                    response_body_sha256=sha256_hex(""),
                    response_size_bytes=0,
                    payload_json=None,
                    attempts=attempt,
                    error_type="IO_EXCEPTION",
                    error_message=str(exc),
                )
                if attempt == attempts:
                    return result
                last_response = result
                self._sleep_before_retry(attempt, None)
                continue

            self._sleep_before_retry(attempt, None)

        if last_response is not None:
            return last_response
        raise RuntimeError("Wunderground fetch loop exited without a response")

    def _sleep_before_retry(self, attempt: int, retry_after: float | None) -> None:
        if retry_after is not None:
            sleep_seconds = retry_after
        else:
            base = 1.5 * (2 ** max(0, attempt - 1))
            sleep_seconds = min(60.0, base + random.uniform(0.0, 0.25))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
