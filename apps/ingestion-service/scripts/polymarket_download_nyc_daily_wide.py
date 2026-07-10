#!/usr/bin/env python3
"""
Backfill Polymarket "Highest temperature in NYC" daily wide CSVs.

Output format:
  - One CSV per day
  - Minute-grid rows (UTC timestamps)
  - Bucket-side wide columns:
      <bucket_label>__YES, <bucket_label>__NO
  - Forward-filled "latest known price" per minute
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from zoneinfo import ZoneInfo


GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"
DATA_BASE_URL = "https://data-api.polymarket.com"

DEFAULT_TIMEZONE = "America/New_York"
DEFAULT_INTERVAL = "1m"
DEFAULT_FIDELITY_MINUTES = 1
DEFAULT_PRICE_SOURCE = "auto"
DEFAULT_MAX_CONCURRENCY = 1
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_TIMEOUT = (10, 60)
DEFAULT_MAX_RETRIES = 1
DEFAULT_RETRY_BACKOFF_SECONDS = 0.6
DEFAULT_TRADES_PAGE_LIMIT = 10000
DEFAULT_SUFFICIENCY_MIN_POINTS = 60

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
YES_LABELS = {"yes", "y"}
NO_LABELS = {"no", "n"}
TITLE_DATE_PATTERN = re.compile(
    r"\bon\s+([A-Za-z]+)\s+(\d{1,2})(?:,\s*(\d{4}))?\b",
    re.IGNORECASE,
)
FIDELITY_MIN_PATTERN = re.compile(r"minimum 'fidelity'.*?is\s+(\d+)", re.IGNORECASE)
RANGE_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:-|to)\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
OR_BELOW_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)\s*[^0-9]*or\s*below", re.IGNORECASE)
OR_ABOVE_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)\s*[^0-9]*or\s*(?:above|higher|more)", re.IGNORECASE)
SINGLE_NUMBER_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)")

logger = logging.getLogger("polymarket_backfill")


class ApiError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class PublicApiClient:
    def __init__(
        self,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: Tuple[int, int] = DEFAULT_TIMEOUT,
        backoff_base_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    ) -> None:
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_base_seconds = backoff_base_seconds
        self._thread_local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {"User-Agent": "weather-forecasting-predictionmarkets/polymarket-backfill"}
            )
            self._thread_local.session = session
        return session

    def _sleep_backoff(self, attempt: int, retry_after: Optional[str]) -> None:
        if retry_after:
            try:
                sleep_s = float(retry_after)
                if sleep_s > 0:
                    time.sleep(min(sleep_s, 60.0))
                    return
            except Exception:
                pass
        exp = self.backoff_base_seconds * (2 ** attempt)
        jitter = random.random() * 0.35
        time.sleep(min(exp + jitter, 60.0))

    def get_json(
        self,
        base_url: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        allow_404: bool = False,
    ) -> Any:
        if not path.startswith("/"):
            path = "/" + path
        url = f"{base_url.rstrip('/')}{path}"
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._session().get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise ApiError(f"request failed for {url}: {exc}") from exc
                self._sleep_backoff(attempt, None)
                continue

            if response.status_code == 404 and allow_404:
                return None

            if response.status_code in RETRYABLE_STATUS_CODES:
                if attempt >= self.max_retries:
                    raise ApiError(
                        f"HTTP {response.status_code} for {url}",
                        status_code=response.status_code,
                        response_text=response.text,
                    )
                self._sleep_backoff(attempt, response.headers.get("Retry-After"))
                continue

            if not (200 <= response.status_code < 300):
                raise ApiError(
                    f"HTTP {response.status_code} for {url} params={params}",
                    status_code=response.status_code,
                    response_text=response.text,
                )

            try:
                return response.json()
            except json.JSONDecodeError as exc:
                raise ApiError(f"invalid JSON from {url}: {exc}") from exc

        raise ApiError(f"request failed for {url}: {last_error}")


@dataclass
class BucketMarket:
    bucket_label: str
    column_label: str
    condition_id: str
    yes_token_id: str
    no_token_id: str
    market_id: str
    market_slug: str
    market_question: str


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso_utc(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def to_epoch_seconds(ts: dt.datetime) -> int:
    return int(ts.timestamp())


def epoch_to_iso_utc(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")


def floor_to_minute(epoch_seconds: int) -> int:
    return int(epoch_seconds) - (int(epoch_seconds) % 60)


def parse_seed_slug(seed_event: str) -> str:
    raw = seed_event.strip()
    if not raw:
        raise ValueError("--seed-event cannot be empty")
    if "://" not in raw:
        return raw.strip("/")

    parsed = urlparse(raw)
    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        raise ValueError(f"unable to parse seed slug from URL: {raw}")
    if "event" in parts:
        idx = parts.index("event")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return parts[-1]


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        return [value]
    return [value]


def normalize_bucket_label(label: str) -> str:
    return " ".join(str(label).strip().split())


def price_to_cents(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except Exception:
        return None
    if f < 0:
        return None
    if f <= 1.0 + 1e-9:
        cents = f * 100.0
    else:
        cents = f
    return round(cents, 2)


def format_price(value: Optional[float]) -> str:
    if value is None:
        return ""
    text = f"{value:.2f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def parse_title_date(
    title: Optional[str],
    *,
    fallback_year: Optional[int],
) -> Optional[dt.date]:
    if not title:
        return None
    m = TITLE_DATE_PATTERN.search(title)
    if not m:
        return None
    month_text = m.group(1)
    day_text = m.group(2)
    year_text = m.group(3)

    year = int(year_text) if year_text else fallback_year
    if year is None:
        return None
    month_candidates = [month_text]
    if month_text.endswith("."):
        month_candidates.append(month_text[:-1])
    for fmt in ("%b", "%B"):
        for month_candidate in month_candidates:
            try:
                month_num = dt.datetime.strptime(month_candidate, fmt).month
                return dt.date(year, month_num, int(day_text))
            except ValueError:
                continue
    return None


def parse_yes_no_from_tokens(market: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    yes_token: Optional[str] = None
    no_token: Optional[str] = None

    tokens = as_list(market.get("tokens"))
    if tokens and isinstance(tokens[0], dict):
        for token_obj in tokens:
            if not isinstance(token_obj, dict):
                continue
            outcome = str(
                token_obj.get("outcome")
                or token_obj.get("name")
                or token_obj.get("label")
                or ""
            ).strip().lower()
            token_id = (
                token_obj.get("token_id")
                or token_obj.get("tokenId")
                or token_obj.get("id")
                or token_obj.get("asset")
                or token_obj.get("outcomeTokenId")
            )
            if token_id is None:
                continue
            token_text = str(token_id)
            if outcome in YES_LABELS and not yes_token:
                yes_token = token_text
            if outcome in NO_LABELS and not no_token:
                no_token = token_text

    if yes_token and no_token:
        return yes_token, no_token

    outcomes = [str(x) for x in as_list(market.get("outcomes"))]
    token_ids = [str(x) for x in as_list(market.get("clobTokenIds"))]
    if len(token_ids) >= 2:
        if outcomes and len(outcomes) == len(token_ids):
            for idx, outcome in enumerate(outcomes):
                outcome_norm = outcome.strip().lower()
                if outcome_norm in YES_LABELS and not yes_token:
                    yes_token = token_ids[idx]
                elif outcome_norm in NO_LABELS and not no_token:
                    no_token = token_ids[idx]
        if not yes_token and not no_token and len(token_ids) == 2:
            yes_token = token_ids[0]
            no_token = token_ids[1]

    return yes_token, no_token


def parse_bucket_bounds(label: str) -> Tuple[float, float]:
    text = label or ""
    m_range = RANGE_PATTERN.search(text)
    if m_range:
        lo = float(m_range.group(1))
        hi = float(m_range.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi

    m_below = OR_BELOW_PATTERN.search(text)
    if m_below:
        hi = float(m_below.group(1))
        return float("-inf"), hi

    m_above = OR_ABOVE_PATTERN.search(text)
    if m_above:
        lo = float(m_above.group(1))
        return lo, float("inf")

    m_single = SINGLE_NUMBER_PATTERN.search(text)
    if m_single:
        x = float(m_single.group(1))
        return x, x
    return float("inf"), float("inf")


def bucket_sort_key(market: Dict[str, Any], label: str) -> Tuple[float, float, str]:
    lo = market.get("lowerBound")
    hi = market.get("upperBound")
    try:
        lo_v = float(lo) if lo is not None else None
        hi_v = float(hi) if hi is not None else None
        if lo_v is not None or hi_v is not None:
            if lo_v is None:
                lo_v = float("-inf")
            if hi_v is None:
                hi_v = float("inf")
            return lo_v, hi_v, label
    except Exception:
        pass
    parsed_lo, parsed_hi = parse_bucket_bounds(label)
    return parsed_lo, parsed_hi, label


def resolve_series_slug(seed_event: Dict[str, Any]) -> Optional[str]:
    slug = seed_event.get("seriesSlug")
    if slug:
        return str(slug)

    for market in as_list(seed_event.get("markets")):
        if isinstance(market, dict) and market.get("seriesSlug"):
            return str(market.get("seriesSlug"))

    nested_series = seed_event.get("series")
    if isinstance(nested_series, dict):
        nested_slug = nested_series.get("slug")
        if nested_slug:
            return str(nested_slug)
    for series_obj in as_list(nested_series):
        if isinstance(series_obj, dict) and series_obj.get("slug"):
            return str(series_obj.get("slug"))
    return None


def get_seed_event_by_slug(client: PublicApiClient, slug: str) -> Dict[str, Any]:
    payload = client.get_json(
        GAMMA_BASE_URL,
        f"/events/slug/{slug}",
        allow_404=True,
    )
    if payload is None:
        raise RuntimeError(f"seed event slug not found: {slug}")
    if isinstance(payload, list):
        if not payload:
            raise RuntimeError(f"seed event slug not found: {slug}")
        payload = payload[0]
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected seed event payload type for slug {slug}: {type(payload)}")
    return payload


def resolve_series_record(
    client: PublicApiClient,
    series_slug: str,
) -> Dict[str, Any]:
    records = client.get_json(
        GAMMA_BASE_URL,
        "/series",
        params={"slug": series_slug},
    )
    if not isinstance(records, list):
        raise RuntimeError(f"unexpected /series payload type: {type(records)}")

    exact = [r for r in records if isinstance(r, dict) and str(r.get("slug", "")) == series_slug]
    if exact:
        return exact[0]

    if records:
        return records[0]

    raise RuntimeError(f"series not found for slug={series_slug}")


def fetch_series_events(client: PublicApiClient, series_id: str) -> List[Dict[str, Any]]:
    payload = client.get_json(GAMMA_BASE_URL, f"/series/{series_id}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected /series/{{id}} payload type: {type(payload)}")
    events = payload.get("events")
    if not isinstance(events, list):
        return []
    return [e for e in events if isinstance(e, dict)]


def event_target_date(
    event_obj: Dict[str, Any],
    tz: ZoneInfo,
) -> Optional[dt.date]:
    end_dt = parse_iso_utc(event_obj.get("endDate"))
    if end_dt is not None:
        return end_dt.astimezone(tz).date()

    start_dt = parse_iso_utc(event_obj.get("startDate"))
    fallback_year = start_dt.year if start_dt else None
    parsed_title_date = parse_title_date(event_obj.get("title"), fallback_year=fallback_year)
    if parsed_title_date is not None:
        return parsed_title_date
    if start_dt is not None:
        return start_dt.astimezone(tz).date()
    return None


def select_events_for_year(
    events: List[Dict[str, Any]],
    *,
    year: int,
    tz: ZoneInfo,
    date_from: Optional[dt.date],
    date_to: Optional[dt.date],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    by_date: Dict[dt.date, Dict[str, Any]] = {}

    for event in events:
        target = event_target_date(event, tz)
        if target is None:
            warnings.append(f"event_missing_target_date event_id={event.get('id')} slug={event.get('slug')}")
            continue
        if target.year != year:
            continue
        if date_from is not None and target < date_from:
            continue
        if date_to is not None and target > date_to:
            continue

        existing = by_date.get(target)
        if existing is None:
            by_date[target] = event
            continue

        existing_start = parse_iso_utc(existing.get("startDate"))
        incoming_start = parse_iso_utc(event.get("startDate"))
        if existing_start is None or (incoming_start is not None and incoming_start > existing_start):
            warnings.append(
                f"duplicate_event_date_replaced date={target.isoformat()} "
                f"old_event_id={existing.get('id')} new_event_id={event.get('id')}"
            )
            by_date[target] = event
        else:
            warnings.append(
                f"duplicate_event_date_ignored date={target.isoformat()} "
                f"event_id={event.get('id')}"
            )

    ordered = [by_date[d] for d in sorted(by_date.keys())]
    return ordered, warnings


def fetch_event_detail(client: PublicApiClient, event_id: str, event_slug: Optional[str]) -> Dict[str, Any]:
    detail = client.get_json(GAMMA_BASE_URL, f"/events/{event_id}", allow_404=True)
    if isinstance(detail, dict):
        return detail
    if event_slug:
        detail = client.get_json(GAMMA_BASE_URL, f"/events/slug/{event_slug}", allow_404=True)
        if isinstance(detail, list):
            detail = detail[0] if detail else None
        if isinstance(detail, dict):
            return detail
    raise RuntimeError(f"unable to fetch event detail for id={event_id} slug={event_slug}")


def extract_bucket_markets(event_detail: Dict[str, Any]) -> Tuple[List[BucketMarket], List[str]]:
    warnings: List[str] = []
    markets = as_list(event_detail.get("markets"))
    parsed_markets: List[Tuple[Tuple[float, float, str], BucketMarket]] = []

    for market in markets:
        if not isinstance(market, dict):
            continue
        bucket_label_raw = (
            market.get("groupItemTitle")
            or market.get("question")
            or market.get("slug")
            or ""
        )
        bucket_label = normalize_bucket_label(str(bucket_label_raw))
        if not bucket_label:
            warnings.append(f"market_missing_bucket_label market_id={market.get('id')}")
            continue

        condition_id = str(market.get("conditionId") or "").strip()
        yes_token_id, no_token_id = parse_yes_no_from_tokens(market)
        if not condition_id:
            warnings.append(f"market_missing_condition_id bucket={bucket_label}")
        if not yes_token_id or not no_token_id:
            warnings.append(f"market_missing_yes_no_token bucket={bucket_label}")

        column_label = bucket_label
        sort_key = bucket_sort_key(market, bucket_label)
        parsed_markets.append(
            (
                sort_key,
                BucketMarket(
                    bucket_label=bucket_label,
                    column_label=column_label,
                    condition_id=condition_id,
                    yes_token_id=str(yes_token_id or ""),
                    no_token_id=str(no_token_id or ""),
                    market_id=str(market.get("id") or ""),
                    market_slug=str(market.get("slug") or ""),
                    market_question=str(market.get("question") or ""),
                ),
            )
        )

    parsed_markets.sort(key=lambda x: x[0])
    ordered = [x[1] for x in parsed_markets]

    # Ensure stable unique column labels.
    seen: Dict[str, int] = {}
    for market in ordered:
        base = market.column_label
        count = seen.get(base, 0)
        if count > 0:
            market.column_label = f"{base} ({count + 1})"
        seen[base] = count + 1

    return ordered, warnings


def resolve_day_window(
    event_detail: Dict[str, Any],
    *,
    target_date: dt.date,
    tz: ZoneInfo,
) -> Tuple[int, int, List[str]]:
    warnings: List[str] = []
    start_dt = parse_iso_utc(event_detail.get("startDate"))
    end_dt = parse_iso_utc(event_detail.get("endDate"))
    if start_dt is not None and end_dt is not None and end_dt > start_dt:
        return to_epoch_seconds(start_dt), to_epoch_seconds(end_dt), warnings

    warnings.append("event_missing_or_invalid_start_endDate_using_timezone_fallback")
    day_start_local = dt.datetime.combine(target_date, dt.time(0, 0, 0), tzinfo=tz)
    day_end_local = day_start_local + dt.timedelta(days=1)
    return (
        to_epoch_seconds(day_start_local.astimezone(dt.timezone.utc)),
        to_epoch_seconds(day_end_local.astimezone(dt.timezone.utc)),
        warnings,
    )


def build_minute_grid(start_ts: int, end_ts: int) -> List[int]:
    if end_ts <= start_ts:
        return []
    current = floor_to_minute(start_ts)
    out: List[int] = []
    while current < end_ts:
        out.append(current)
        current += 60
    return out


def forward_fill_prices(minute_grid: List[int], points: List[Tuple[int, float]]) -> List[Optional[float]]:
    sorted_points = sorted(points, key=lambda x: x[0])
    out: List[Optional[float]] = []
    idx = 0
    latest: Optional[float] = None
    for minute_ts in minute_grid:
        while idx < len(sorted_points) and sorted_points[idx][0] <= minute_ts:
            latest = sorted_points[idx][1]
            idx += 1
        out.append(latest)
    return out


def parse_history_points(payload: Any, *, start_ts: int, end_ts: int) -> List[Tuple[int, float]]:
    if not isinstance(payload, dict):
        return []
    history = payload.get("history")
    if not isinstance(history, list):
        return []
    by_ts: Dict[int, float] = {}
    for item in history:
        if not isinstance(item, dict):
            continue
        t_raw = item.get("t")
        p_raw = item.get("p")
        try:
            t = int(t_raw)
        except Exception:
            continue
        if t < start_ts or t >= end_ts:
            continue
        cents = price_to_cents(p_raw)
        if cents is None:
            continue
        by_ts[t] = cents
    return sorted(by_ts.items(), key=lambda x: x[0])


def fetch_prices_history_for_token(
    client: PublicApiClient,
    token_id: str,
    *,
    start_ts: int,
    end_ts: int,
    interval: str,
    fidelity_minutes: int,
) -> Tuple[List[Tuple[int, float]], int, List[str]]:
    warnings: List[str] = []
    params: Dict[str, Any] = {
        "market": token_id,
        "startTs": int(start_ts),
        "endTs": int(end_ts),
        "interval": interval,
        "fidelity": int(fidelity_minutes),
    }
    used_fidelity = int(fidelity_minutes)
    try:
        payload = client.get_json(CLOB_BASE_URL, "/prices-history", params=params)
        points = parse_history_points(payload, start_ts=start_ts, end_ts=end_ts)
        return points, used_fidelity, warnings
    except ApiError as exc:
        if exc.status_code == 400 and exc.response_text:
            m = FIDELITY_MIN_PATTERN.search(exc.response_text)
            if m:
                min_fidelity = int(m.group(1))
                if min_fidelity > used_fidelity:
                    used_fidelity = min_fidelity
                    warnings.append(
                        f"prices_history_fidelity_adjusted token={token_id} "
                        f"from={fidelity_minutes} to={used_fidelity}"
                    )
                    params["fidelity"] = used_fidelity
                    payload = client.get_json(CLOB_BASE_URL, "/prices-history", params=params)
                    points = parse_history_points(payload, start_ts=start_ts, end_ts=end_ts)
                    return points, used_fidelity, warnings
        raise


def is_prices_history_sufficient(points: List[Tuple[int, float]]) -> bool:
    if not points:
        return False
    return len(points) >= DEFAULT_SUFFICIENCY_MIN_POINTS


def fetch_all_trades_for_condition(
    client: PublicApiClient,
    condition_id: str,
) -> List[Dict[str, Any]]:
    all_trades: List[Dict[str, Any]] = []
    offset = 0
    for _page in range(100):
        params = {
            "market": condition_id,
            "limit": DEFAULT_TRADES_PAGE_LIMIT,
            "offset": offset,
            "takerOnly": "false",
        }
        payload = client.get_json(DATA_BASE_URL, "/trades", params=params)
        if isinstance(payload, dict):
            batch = payload.get("trades") if isinstance(payload.get("trades"), list) else []
        elif isinstance(payload, list):
            batch = payload
        else:
            batch = []
        if not batch:
            return all_trades
        typed_batch = [x for x in batch if isinstance(x, dict)]
        all_trades.extend(typed_batch)
        if len(batch) < DEFAULT_TRADES_PAGE_LIMIT:
            return all_trades
        offset += DEFAULT_TRADES_PAGE_LIMIT
    raise RuntimeError("Trade history exceeded the 100-page safety budget")


def map_trade_to_token(
    trade: Dict[str, Any],
    *,
    yes_token_id: str,
    no_token_id: str,
) -> Optional[str]:
    asset = str(trade.get("asset") or "").strip()
    if asset and asset == yes_token_id:
        return yes_token_id
    if asset and asset == no_token_id:
        return no_token_id

    outcome = str(trade.get("outcome") or "").strip().lower()
    if outcome in YES_LABELS:
        return yes_token_id
    if outcome in NO_LABELS:
        return no_token_id

    outcome_idx = trade.get("outcomeIndex")
    try:
        idx = int(outcome_idx)
        if idx == 0:
            return yes_token_id
        if idx == 1:
            return no_token_id
    except Exception:
        pass
    return None


def build_token_points_from_trades(
    trades: List[Dict[str, Any]],
    *,
    yes_token_id: str,
    no_token_id: str,
    start_ts: int,
    end_ts: int,
) -> Dict[str, List[Tuple[int, float]]]:
    by_token_ts: Dict[str, Dict[int, float]] = {yes_token_id: {}, no_token_id: {}}
    for trade in trades:
        try:
            ts = int(trade.get("timestamp"))
        except Exception:
            continue
        if ts < start_ts or ts >= end_ts:
            continue
        token = map_trade_to_token(trade, yes_token_id=yes_token_id, no_token_id=no_token_id)
        if not token:
            continue
        price_cents = price_to_cents(trade.get("price"))
        if price_cents is None:
            continue
        by_token_ts[token][ts] = price_cents
    return {
        token: sorted(ts_map.items(), key=lambda x: x[0])
        for token, ts_map in by_token_ts.items()
    }


def validate_existing_csv(
    csv_path: Path,
    *,
    expected_columns: List[str],
    expected_rows: int,
) -> Tuple[bool, str]:
    if not csv_path.exists():
        return False, "missing_file"
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return False, "empty_file"
            if header != expected_columns:
                return False, "header_mismatch"

            row_count = 0
            prev_ts: Optional[int] = None
            for row in reader:
                row_count += 1
                if len(row) != len(expected_columns):
                    return False, f"row_column_count_mismatch_at_row={row_count}"
                ts_text = row[0]
                parsed = parse_iso_utc(ts_text)
                if parsed is None:
                    return False, f"invalid_timestamp_at_row={row_count}"
                ts = to_epoch_seconds(parsed)
                if prev_ts is not None and ts - prev_ts != 60:
                    return False, f"timestamp_gap_not_60s_at_row={row_count}"
                prev_ts = ts
            if row_count != expected_rows:
                return False, f"row_count_mismatch expected={expected_rows} actual={row_count}"
    except Exception as exc:
        return False, f"csv_validation_exception={exc}"
    return True, "ok"


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def write_csv_atomic(path: Path, rows: Iterable[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def build_csv_columns(bucket_markets: List[BucketMarket]) -> List[str]:
    columns = ["timestamp"]
    for bucket in bucket_markets:
        columns.append(f"{bucket.column_label}__YES")
        columns.append(f"{bucket.column_label}__NO")
    return columns


def process_event_day(
    *,
    client: PublicApiClient,
    event_summary: Dict[str, Any],
    date_key: dt.date,
    tz: ZoneInfo,
    args: argparse.Namespace,
    day_csv_path: Path,
    failed_report_path: Path,
) -> Dict[str, Any]:
    warnings: List[str] = []
    event_id = str(event_summary.get("id") or "")
    event_slug = str(event_summary.get("slug") or "")
    event_start = event_summary.get("startDate")
    event_end = event_summary.get("endDate")

    detail = fetch_event_detail(client, event_id, event_slug)
    bucket_markets, bucket_warnings = extract_bucket_markets(detail)
    warnings.extend(bucket_warnings)

    if not bucket_markets:
        raise RuntimeError("no_bucket_markets")

    missing_fields = [
        b.bucket_label
        for b in bucket_markets
        if not b.condition_id or not b.yes_token_id or not b.no_token_id
    ]
    if missing_fields:
        raise RuntimeError(f"missing_condition_or_token_ids_for_buckets={missing_fields}")

    start_ts, end_ts, window_warnings = resolve_day_window(
        detail,
        target_date=date_key,
        tz=tz,
    )
    warnings.extend(window_warnings)
    if end_ts <= start_ts:
        raise RuntimeError(f"invalid_day_window start={start_ts} end={end_ts}")

    minute_grid = build_minute_grid(start_ts, end_ts)
    if not minute_grid:
        raise RuntimeError("empty_minute_grid")

    columns = build_csv_columns(bucket_markets)
    expected_rows = len(minute_grid)

    if day_csv_path.exists() and not args.force:
        ok, reason = validate_existing_csv(
            day_csv_path,
            expected_columns=columns,
            expected_rows=expected_rows,
        )
        if ok:
            return {
                "date": date_key.isoformat(),
                "event_id": event_id,
                "event_slug": event_slug,
                "startDate": event_start,
                "endDate": event_end,
                "output_csv_path": str(day_csv_path),
                "status": "skipped_existing",
                "price_source_used": "existing",
                "num_buckets": len(bucket_markets),
                "num_tokens": len(bucket_markets) * 2,
                "minutes_in_day": expected_rows,
                "total_points_fetched": 0,
                "warnings": warnings + [f"skipped_existing reason={reason}"],
            }
        warnings.append(f"existing_csv_invalid_rebuilding reason={reason}")

    unique_tokens: List[str] = []
    for bucket in bucket_markets:
        unique_tokens.append(bucket.yes_token_id)
        unique_tokens.append(bucket.no_token_id)
    unique_tokens = sorted(set(unique_tokens))

    price_source_mode = args.price_source
    price_source_used = ""
    per_token_points: Dict[str, List[Tuple[int, float]]] = {}
    total_points_fetched = 0

    if price_source_mode in {"prices_history", "auto"}:
        futures = {}
        with ThreadPoolExecutor(max_workers=max(1, int(args.max_concurrency))) as pool:
            for token_id in unique_tokens:
                fut = pool.submit(
                    fetch_prices_history_for_token,
                    client,
                    token_id,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    interval=args.interval,
                    fidelity_minutes=int(args.fidelity_minutes),
                )
                futures[fut] = token_id
            for fut in as_completed(futures):
                token_id = futures[fut]
                points, used_fidelity, token_warnings = fut.result()
                warnings.extend(token_warnings)
                per_token_points[token_id] = points
                total_points_fetched += len(points)
                logger.debug(
                    "prices-history token=%s points=%s fidelity=%s",
                    token_id,
                    len(points),
                    used_fidelity,
                )

        insufficient_tokens = [
            token for token, pts in per_token_points.items() if not is_prices_history_sufficient(pts)
        ]
        if insufficient_tokens and price_source_mode == "auto":
            warnings.append(
                "prices_history_insufficient_switching_to_trades tokens="
                + ",".join(insufficient_tokens)
            )
            per_token_points = {}
            total_points_fetched = 0
            price_source_used = "trades"
        else:
            price_source_used = "prices_history"

    if price_source_mode == "trades" or price_source_used == "trades":
        per_token_points = {token_id: [] for token_id in unique_tokens}
        futures = {}
        with ThreadPoolExecutor(max_workers=max(1, int(args.max_concurrency))) as pool:
            for bucket in bucket_markets:
                fut = pool.submit(fetch_all_trades_for_condition, client, bucket.condition_id)
                futures[fut] = bucket
            for fut in as_completed(futures):
                bucket = futures[fut]
                trades = fut.result()
                token_points = build_token_points_from_trades(
                    trades,
                    yes_token_id=bucket.yes_token_id,
                    no_token_id=bucket.no_token_id,
                    start_ts=start_ts,
                    end_ts=end_ts,
                )
                yes_pts = token_points.get(bucket.yes_token_id, [])
                no_pts = token_points.get(bucket.no_token_id, [])
                per_token_points[bucket.yes_token_id] = yes_pts
                per_token_points[bucket.no_token_id] = no_pts
                total_points_fetched += len(yes_pts) + len(no_pts)
                logger.debug(
                    "trades condition=%s yes_points=%s no_points=%s",
                    bucket.condition_id,
                    len(yes_pts),
                    len(no_pts),
                )
        price_source_used = "trades"

    if not price_source_used:
        price_source_used = "prices_history"

    token_filled: Dict[str, List[Optional[float]]] = {}
    for token_id in unique_tokens:
        token_filled[token_id] = forward_fill_prices(
            minute_grid,
            per_token_points.get(token_id, []),
        )

    if any(len(token_filled[token]) != expected_rows for token in unique_tokens):
        raise RuntimeError("forward_fill_length_mismatch")

    rows: List[List[str]] = []
    rows.append(columns)
    for idx, minute_ts in enumerate(minute_grid):
        row = [epoch_to_iso_utc(minute_ts)]
        for bucket in bucket_markets:
            row.append(format_price(token_filled[bucket.yes_token_id][idx]))
            row.append(format_price(token_filled[bucket.no_token_id][idx]))
        rows.append(row)

    actual_row_count = len(rows) - 1
    if actual_row_count != expected_rows:
        raise RuntimeError(f"csv_row_count_mismatch expected={expected_rows} actual={actual_row_count}")
    if len(rows[0]) != 1 + (2 * len(bucket_markets)):
        raise RuntimeError("csv_column_count_mismatch")
    for i in range(2, len(rows)):
        prev_parsed = parse_iso_utc(rows[i - 1][0])
        cur_parsed = parse_iso_utc(rows[i][0])
        if prev_parsed is None or cur_parsed is None:
            raise RuntimeError(f"invalid_timestamp_after_build row={i}")
        prev_ts = to_epoch_seconds(prev_parsed)
        cur_ts = to_epoch_seconds(cur_parsed)
        if cur_ts - prev_ts != 60:
            raise RuntimeError(f"timestamp_not_strict_60s row={i}")

    write_csv_atomic(day_csv_path, rows)
    if failed_report_path.exists():
        failed_report_path.unlink(missing_ok=True)

    return {
        "date": date_key.isoformat(),
        "event_id": event_id,
        "event_slug": event_slug,
        "startDate": event_start,
        "endDate": event_end,
        "output_csv_path": str(day_csv_path),
        "status": "completed",
        "price_source_used": price_source_used,
        "num_buckets": len(bucket_markets),
        "num_tokens": len(bucket_markets) * 2,
        "minutes_in_day": expected_rows,
        "total_points_fetched": int(total_points_fetched),
        "warnings": warnings,
    }


def parse_optional_date(value: Optional[str], *, arg_name: str) -> Optional[dt.date]:
    if value is None:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid {arg_name} (expected YYYY-MM-DD): {value}") from exc


def configure_logging(level: str) -> None:
    log_level = getattr(logging, level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"invalid log level: {level}")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill Polymarket NYC highest temperature daily wide CSVs."
    )
    parser.add_argument("--seed-event", required=True, help="Seed event slug or full Polymarket URL.")
    parser.add_argument("--year", required=True, type=int, help="Target year (e.g., 2025).")
    parser.add_argument("--out-dir", required=True, help="Output directory root.")

    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE, help="Fallback timezone.")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help="CLOB prices-history interval.")
    parser.add_argument(
        "--fidelity-minutes",
        type=int,
        default=DEFAULT_FIDELITY_MINUTES,
        help="CLOB prices-history fidelity minutes (default 1; auto-adjusts if API requires higher).",
    )
    parser.add_argument(
        "--price-source",
        default=DEFAULT_PRICE_SOURCE,
        choices=("prices_history", "trades", "auto"),
        help="Historical price source mode.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help="Maximum token/market fetch concurrency per day.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if daily CSV already exists and validates.",
    )
    parser.add_argument(
        "--write-manifest",
        dest="write_manifest",
        action="store_true",
        default=True,
        help="Write index manifest JSON (default true).",
    )
    parser.add_argument(
        "--no-write-manifest",
        dest="write_manifest",
        action="store_false",
        help="Disable writing index manifest JSON.",
    )
    parser.add_argument("--log-level", default=DEFAULT_LOG_LEVEL, help="Python log level.")

    # Optional convenience window for partial year runs (defaults to full year).
    parser.add_argument("--date-from", default=None, help="Inclusive start date YYYY-MM-DD.")
    parser.add_argument("--date-to", default=None, help="Inclusive end date YYYY-MM-DD.")

    return parser.parse_args(argv)


def iter_dates_inclusive(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    tz = ZoneInfo(args.timezone)
    seed_slug = parse_seed_slug(args.seed_event)
    date_from = parse_optional_date(args.date_from, arg_name="--date-from")
    date_to = parse_optional_date(args.date_to, arg_name="--date-to")
    if date_from and date_to and date_to < date_from:
        raise ValueError("--date-to must be >= --date-from")

    year_root = Path(args.out_dir).resolve() / str(args.year)
    days_dir = year_root / "days"
    manifest_path = year_root / f"index_{args.year}.json"
    days_dir.mkdir(parents=True, exist_ok=True)

    client = PublicApiClient()

    logger.info("Resolving seed event slug=%s", seed_slug)
    seed_event = get_seed_event_by_slug(client, seed_slug)
    series_slug = resolve_series_slug(seed_event)
    if not series_slug:
        raise RuntimeError(f"unable to resolve series slug from seed event slug={seed_slug}")

    logger.info("Resolving series slug=%s", series_slug)
    series_record = resolve_series_record(client, series_slug)
    series_id = str(series_record.get("id") or "")
    if not series_id:
        raise RuntimeError(f"series record missing id for slug={series_slug}")

    logger.info("Fetching events for series id=%s slug=%s", series_id, series_slug)
    series_events = fetch_series_events(client, series_id)
    selected_events, selection_warnings = select_events_for_year(
        series_events,
        year=args.year,
        tz=tz,
        date_from=date_from,
        date_to=date_to,
    )
    target_start = date_from if date_from is not None else dt.date(args.year, 1, 1)
    target_end = date_to if date_to is not None else dt.date(args.year, 12, 31)
    target_dates = list(iter_dates_inclusive(target_start, target_end))

    event_by_date: Dict[dt.date, Dict[str, Any]] = {}
    for event in selected_events:
        d = event_target_date(event, tz)
        if d is not None:
            event_by_date[d] = event

    logger.info(
        "Selected %s events; target dates=%s (from %s to %s)",
        len(event_by_date),
        len(target_dates),
        target_start,
        target_end,
    )
    manifest_days: Dict[str, Dict[str, Any]] = {}
    if manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for entry in existing_manifest.get("days", []):
                if isinstance(entry, dict) and entry.get("date"):
                    manifest_days[str(entry["date"])] = entry
        except Exception:
            pass

    for idx, date_key in enumerate(target_dates, start=1):
        day_name = date_key.isoformat()
        day_csv_path = days_dir / f"{day_name}.csv"
        failed_report_path = days_dir / f"{day_name}_FAILED.json"
        event_summary = event_by_date.get(date_key)
        if event_summary is None:
            logger.warning("[%s/%s] missing event for date=%s", idx, len(target_dates), day_name)
            failure = {
                "date": day_name,
                "event_id": "",
                "event_slug": "",
                "error": "missing_event_for_date",
                "failed_at_utc": utc_now_iso(),
            }
            write_json_atomic(failed_report_path, failure)
            manifest_days[day_name] = {
                "date": day_name,
                "event_id": "",
                "event_slug": "",
                "status": "failed",
                "price_source_used": None,
                "output_csv_path": str(day_csv_path),
                "warnings": ["missing_event_for_date"],
            }
            if args.write_manifest:
                manifest = {
                    "generated_at_utc": utc_now_iso(),
                    "series_id": series_id,
                    "series_slug": series_slug,
                    "series_title": series_record.get("title"),
                    "seed_slug": seed_slug,
                    "year": args.year,
                    "date_from": date_from.isoformat() if date_from else None,
                    "date_to": date_to.isoformat() if date_to else None,
                    "selection_warnings": selection_warnings,
                    "days": [manifest_days[d] for d in sorted(manifest_days.keys())],
                }
                write_json_atomic(manifest_path, manifest)
            continue

        logger.info("[%s/%s] processing date=%s event_id=%s slug=%s",
                    idx, len(target_dates), day_name, event_summary.get("id"), event_summary.get("slug"))

        try:
            day_entry = process_event_day(
                client=client,
                event_summary=event_summary,
                date_key=date_key,
                tz=tz,
                args=args,
                day_csv_path=day_csv_path,
                failed_report_path=failed_report_path,
            )
            manifest_days[day_name] = day_entry
            logger.info(
                "Completed date=%s status=%s source=%s rows=%s buckets=%s",
                day_name,
                day_entry.get("status"),
                day_entry.get("price_source_used"),
                day_entry.get("minutes_in_day"),
                day_entry.get("num_buckets"),
            )
        except Exception as exc:
            logger.exception("Failed date=%s event_id=%s", day_name, event_summary.get("id"))
            failure = {
                "date": day_name,
                "event_id": str(event_summary.get("id") or ""),
                "event_slug": str(event_summary.get("slug") or ""),
                "error": str(exc),
                "failed_at_utc": utc_now_iso(),
            }
            write_json_atomic(failed_report_path, failure)
            manifest_days[day_name] = {
                "date": day_name,
                "event_id": str(event_summary.get("id") or ""),
                "event_slug": str(event_summary.get("slug") or ""),
                "status": "failed",
                "price_source_used": None,
                "output_csv_path": str(day_csv_path),
                "warnings": [str(exc)],
            }

        if args.write_manifest:
            manifest = {
                "generated_at_utc": utc_now_iso(),
                "series_id": series_id,
                "series_slug": series_slug,
                "series_title": series_record.get("title"),
                "seed_slug": seed_slug,
                "year": args.year,
                "date_from": date_from.isoformat() if date_from else None,
                "date_to": date_to.isoformat() if date_to else None,
                "selection_warnings": selection_warnings,
                "days": [manifest_days[d] for d in sorted(manifest_days.keys())],
            }
            write_json_atomic(manifest_path, manifest)

    logger.info("Done. Output root: %s", year_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
