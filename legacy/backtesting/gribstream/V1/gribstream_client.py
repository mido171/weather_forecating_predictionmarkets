from __future__ import annotations

import csv
import io
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import requests

from .config import (
    GRIBSTREAM_API_TOKEN_ENV,
    GRIBSTREAM_BASE_URL,
    REQUEST_BACKOFF_SECONDS,
    REQUEST_CONNECT_TIMEOUT_SECONDS,
    REQUEST_MAX_RETRIES,
    REQUEST_READ_TIMEOUT_SECONDS,
    RETRYABLE_STATUS_CODES,
    parse_utc,
)
from .model_catalog import VariableSpec

LOGGER = logging.getLogger(__name__)
UTC = timezone.utc
FIXED_COLUMNS = {"forecasted_at", "forecasted_time", "lat", "lon", "name", "member"}


@dataclass(frozen=True)
class ParsedTimeseriesRow:
    forecasted_at_utc: datetime
    forecasted_time_utc: datetime
    lat: float
    lon: float
    coord_name: str | None
    variable: VariableSpec
    member: int | None
    value_native: float


@dataclass(frozen=True)
class FetchTimeseriesResult:
    rows: list[ParsedTimeseriesRow]
    http_status: int
    attempts: int


class GribstreamRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        attempts: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.attempts = attempts


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return max((parsed - datetime.now(tz=UTC)).total_seconds(), 0.0)


def _token_from_env() -> str:
    token = str(os.environ.get(GRIBSTREAM_API_TOKEN_ENV) or "").strip()
    if not token:
        raise GribstreamRequestError(
            f"Missing required environment variable {GRIBSTREAM_API_TOKEN_ENV}"
        )
    return token


def _parse_variable_header(header: str) -> VariableSpec:
    parts = header.split("|")
    while len(parts) < 3:
        parts.append("")
    return VariableSpec(parts[0], parts[1], parts[2])


def _sort_rows(rows: list[ParsedTimeseriesRow]) -> list[ParsedTimeseriesRow]:
    return sorted(
        rows,
        key=lambda row: (
            row.forecasted_time_utc,
            row.forecasted_at_utc,
            row.variable.name,
            row.variable.level,
            row.variable.info,
            -1 if row.member is None else row.member,
        ),
    )


def _parse_csv_body(text: str) -> list[ParsedTimeseriesRow]:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise GribstreamRequestError("Gribstream timeseries CSV response missing header")
    variable_headers = [field for field in reader.fieldnames if field not in FIXED_COLUMNS]
    rows: list[ParsedTimeseriesRow] = []
    for raw_row in reader:
        if not raw_row:
            continue
        forecasted_at_utc = parse_utc(str(raw_row["forecasted_at"]))
        forecasted_time_utc = parse_utc(str(raw_row["forecasted_time"]))
        lat = float(raw_row["lat"])
        lon = float(raw_row["lon"])
        coord_name = str(raw_row.get("name") or "").strip() or None
        member_text = str(raw_row.get("member") or "").strip()
        member = int(member_text) if member_text else None
        for variable_header in variable_headers:
            cell_text = str(raw_row.get(variable_header) or "").strip()
            if not cell_text:
                continue
            value_native = float(cell_text)
            if math.isnan(value_native):
                continue
            rows.append(
                ParsedTimeseriesRow(
                    forecasted_at_utc=forecasted_at_utc,
                    forecasted_time_utc=forecasted_time_utc,
                    lat=lat,
                    lon=lon,
                    coord_name=coord_name,
                    variable=_parse_variable_header(variable_header),
                    member=member,
                    value_native=value_native,
                )
            )
    return _sort_rows(rows)


class GribstreamClient:
    def __init__(
        self,
        *,
        token: str | None = None,
        base_url: str = GRIBSTREAM_BASE_URL,
        max_retries: int = REQUEST_MAX_RETRIES,
        connect_timeout_seconds: int = REQUEST_CONNECT_TIMEOUT_SECONDS,
        read_timeout_seconds: int = REQUEST_READ_TIMEOUT_SECONDS,
        backoff_seconds: float = REQUEST_BACKOFF_SECONDS,
        session: requests.Session | None = None,
    ) -> None:
        self._token = token or _token_from_env()
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._timeout = (connect_timeout_seconds, read_timeout_seconds)
        self._backoff_seconds = backoff_seconds
        self._session = session or requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "Accept": "text/csv",
                "Accept-Encoding": "gzip",
            }
        )

    def fetch_timeseries_with_meta(
        self,
        model_code: str,
        payload: dict[str, Any],
    ) -> FetchTimeseriesResult:
        url = f"{self._base_url}/api/v2/{model_code}/timeseries"
        attempts = 0
        last_status = 0
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            attempts = attempt
            try:
                response = self._session.post(url, json=payload, timeout=self._timeout)
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    break
                sleep_seconds = self._backoff_seconds * (2 ** (attempt - 1)) + random.uniform(0.0, 0.25)
                LOGGER.warning(
                    "Gribstream request retry model=%s attempt=%d/%d reason=%s sleep=%.2fs",
                    model_code,
                    attempt,
                    self._max_retries,
                    exc.__class__.__name__,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
                continue
            last_status = response.status_code
            if response.status_code == 200:
                return FetchTimeseriesResult(
                    rows=_parse_csv_body(response.text),
                    http_status=response.status_code,
                    attempts=attempts,
                )
            if response.status_code not in RETRYABLE_STATUS_CODES or attempt >= self._max_retries:
                message = response.text.strip()
                raise GribstreamRequestError(
                    f"Gribstream timeseries request failed model={model_code} "
                    f"status={response.status_code} body={message[:500]}",
                    status_code=response.status_code,
                    attempts=attempts,
                )
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            sleep_seconds = retry_after
            if sleep_seconds is None:
                sleep_seconds = self._backoff_seconds * (2 ** (attempt - 1)) + random.uniform(0.0, 0.25)
            LOGGER.warning(
                "Gribstream request retry model=%s attempt=%d/%d status=%d sleep=%.2fs",
                model_code,
                attempt,
                self._max_retries,
                response.status_code,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
        if last_error is not None:
            raise GribstreamRequestError(
                f"Gribstream timeseries request failed model={model_code} error={last_error}",
                attempts=attempts,
            ) from last_error
        raise GribstreamRequestError(
            f"Gribstream timeseries request failed model={model_code} status={last_status}",
            status_code=last_status,
            attempts=attempts,
        )

    def fetch_timeseries(self, model_code: str, payload: dict[str, Any]) -> list[ParsedTimeseriesRow]:
        return self.fetch_timeseries_with_meta(model_code, payload).rows


_DEFAULT_CLIENT: GribstreamClient | None = None


def default_client() -> GribstreamClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = GribstreamClient()
    return _DEFAULT_CLIENT


def fetch_timeseries(model_code: str, payload: dict[str, Any]) -> list[ParsedTimeseriesRow]:
    return default_client().fetch_timeseries(model_code, payload)
