#!/usr/bin/env python3
"""
Download minute-by-minute Kalshi bucket prices for KXHIGHMIA events and export per-day CSVs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dateutil import parser as date_parser

from kalshi_api import DEFAULT_BASE_URL, KalshiApiError, KalshiClient


MONTH_ABBR = {
    1: "JAN",
    2: "FEB",
    3: "MAR",
    4: "APR",
    5: "MAY",
    6: "JUN",
    7: "JUL",
    8: "AUG",
    9: "SEP",
    10: "OCT",
    11: "NOV",
    12: "DEC",
}

DEFAULT_START_DATE = "2025-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_SERIES = "KXHIGHMIA"
DEFAULT_OUT_DIR = "data/kalshi_backtest_data"
DEFAULT_MAX_RETRIES = 6
DEFAULT_SLEEP_ON_RATE_LIMIT = True
DEFAULT_SKIP_EXISTING = True

MAX_POINTS_TOTAL = 9000
MAX_TICKERS_PER_BATCH = 100
HISTORICAL_CHUNK_MINUTES = 1440


@dataclass
class ManifestEntry:
    date: str
    event_ticker: str
    market_tickers: List[str]
    start_time: Optional[str]
    end_time: Optional[str]
    start_ts: Optional[int]
    end_ts: Optional[int]
    rows_written: int
    errors: List[str]


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso_to_epoch(ts: Optional[str]) -> Optional[int]:
    if not ts:
        return None
    try:
        parsed = date_parser.isoparse(ts)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return int(parsed.timestamp())


def _epoch_to_iso(ts: int) -> str:
    return dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _event_tickers(series: str, day: dt.date) -> Tuple[str, str]:
    dd = f"{day.day:02d}"
    mon = MONTH_ABBR[day.month]
    yy = f"{day.year % 100:02d}"
    primary = f"{series.upper()}-{dd}{mon}{yy}"
    alternate = f"{series.upper()}-{yy}{mon}{dd}"
    return primary, alternate


def _daterange(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def _extract_markets(event_resp: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    event_obj = event_resp.get("event") or {}
    markets = event_obj.get("markets")
    if isinstance(markets, list) and markets:
        return event_obj, markets

    markets = event_resp.get("markets")
    if isinstance(markets, list) and markets:
        return event_obj, markets

    return event_obj, []


def _parse_event_title_date(title: Optional[str]) -> Optional[dt.date]:
    if not title:
        return None
    # Example: "Highest temperature in Miami on Oct 24, 2025?"
    m = re.search(r"\bon\s+([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})\b", title)
    if not m:
        return None
    try:
        return date_parser.parse(f"{m.group(1)} {m.group(2)} {m.group(3)}").date()
    except Exception:
        return None


def _event_matches_day(event_resp: Dict[str, Any], day: dt.date) -> Optional[bool]:
    event_obj, markets = _extract_markets(event_resp)

    # Prefer explicit date parsed from the title, since KXHIGHMIA tickers are ambiguous
    # (e.g. "KXHIGHMIA-24OCT25" refers to Oct 25, 2024, not Oct 24, 2025).
    title_day = _parse_event_title_date(event_obj.get("title"))
    if title_day is not None:
        return title_day == day

    # Fallback: accept if the day falls within the market open/close window (UTC dates).
    open_times = [t for t in (_market_open_epoch(m) for m in markets) if t is not None]
    close_times = [t for t in (_market_close_epoch(m) for m in markets) if t is not None]
    if not open_times or not close_times:
        return None
    start = dt.datetime.fromtimestamp(min(open_times), tz=dt.timezone.utc).date()
    end = dt.datetime.fromtimestamp(max(close_times), tz=dt.timezone.utc).date()
    return start <= day <= end


def _resolve_event_for_day(series: str, day: dt.date, client: KalshiClient) -> Tuple[Optional[str], Optional[Dict[str, Any]], List[str]]:
    """
    KXHIGHMIA tickers are ambiguous between DDMONYY and YYMONDD forms.

    We resolve by:
      1) trying both candidates
      2) selecting the one whose event title date matches `day` (preferred)
      3) otherwise selecting the one whose open/close window spans `day`

    Returns (event_ticker, event_resp, errors).
    """
    errors: List[str] = []
    ddmonyy, yymondd = _event_tickers(series, day)

    candidates: List[Tuple[str, Dict[str, Any], Optional[bool]]] = []
    # On dates like 2024-10-24, DDMONYY and YYMONDD are identical strings ("24OCT24").
    # Deduplicate to avoid double-fetching and spurious "multiple_event_matches".
    for t in dict.fromkeys([yymondd, ddmonyy]):
        try:
            resp = client.get_event(t)
        except KalshiApiError as exc:
            if exc.status_code == 404:
                continue
            raise
        match = _event_matches_day(resp, day)
        candidates.append((t, resp, match))

    if not candidates:
        errors.append("event_not_found")
        return None, None, errors

    exact = [c for c in candidates if c[2] is True]
    if len(exact) == 1:
        return exact[0][0], exact[0][1], errors
    if len(exact) > 1:
        # Extremely unlikely; choose deterministically (prefer YYMONDD).
        errors.append("multiple_event_matches")
        return exact[0][0], exact[0][1], errors

    span = [c for c in candidates if c[2] is None or c[2] is True]
    if len(span) == 1:
        if span[0][2] is None:
            errors.append("event_date_unverified")
        return span[0][0], span[0][1], errors

    # We could not verify; fail closed rather than silently download the wrong day.
    errors.append("event_date_mismatch_unresolved")
    return None, None, errors


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _subtitle_sort_key(subtitle: str) -> Tuple[float, float, str]:
    if not subtitle:
        return (math.inf, math.inf, "")
    s = subtitle.strip()

    or_below = re.search(r"([-+]?\d+(?:\.\d+)?)\s*°?\s*or\s*(?:below|less)", s, re.IGNORECASE)
    if or_below:
        x = float(or_below.group(1))
        return (x, float("-inf"), s)

    to_match = re.search(
        r"([-+]?\d+(?:\.\d+)?)\s*°?\s*(?:to|\-|–)\s*([-+]?\d+(?:\.\d+)?)",
        s,
        re.IGNORECASE,
    )
    if to_match:
        a = float(to_match.group(1))
        b = float(to_match.group(2))
        return (a, b, s)

    or_above = re.search(r"([-+]?\d+(?:\.\d+)?)\s*°?\s*or\s*(?:above|more)", s, re.IGNORECASE)
    if or_above:
        y = float(or_above.group(1))
        return (y, float("inf"), s)

    return (math.inf, math.inf, s)


def _market_sort_key(market: Dict[str, Any]) -> Tuple[float, float, str]:
    floor_val = _parse_float(market.get("floor_strike"))
    cap_val = _parse_float(market.get("cap_strike"))
    subtitle = market.get("subtitle") or ""

    if floor_val is not None or cap_val is not None:
        if floor_val is None and cap_val is not None:
            return (cap_val, float("-inf"), subtitle)
        if floor_val is not None and cap_val is None:
            return (floor_val, float("inf"), subtitle)
        if floor_val is not None and cap_val is not None:
            return (floor_val, cap_val, subtitle)

    return _subtitle_sort_key(subtitle)


def _format_cents(value: Decimal) -> str:
    s = format(value, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    return s


def _price_cents_from_candle(candle: Dict[str, Any]) -> Optional[str]:
    price = candle.get("price") or {}
    mean_dollars = price.get("mean_dollars")
    if mean_dollars is not None:
        try:
            dec = Decimal(str(mean_dollars)) * Decimal("100")
        except Exception:
            return None
        return _format_cents(dec)

    mean_cents = price.get("mean")
    if mean_cents is None:
        return None
    try:
        dec = Decimal(str(mean_cents))
    except Exception:
        return None
    return _format_cents(dec)


def _extract_candles_by_market(
    resp: Dict[str, Any],
    fallback_tickers: Iterable[str],
) -> Dict[str, List[Dict[str, Any]]]:
    fallback_list = list(fallback_tickers)
    out: Dict[str, List[Dict[str, Any]]] = {}

    markets_block = resp.get("markets")
    if isinstance(markets_block, list):
        for market in markets_block:
            if not isinstance(market, dict):
                continue
            mt = market.get("market_ticker") or market.get("ticker")
            if not mt and len(fallback_list) == 1:
                mt = fallback_list[0]
            if not mt:
                continue
            candles = market.get("candlesticks") or []
            if isinstance(candles, list):
                out.setdefault(mt, []).extend(candles)
        return out

    items = resp.get("candlesticks")
    if items is None:
        return out

    if isinstance(items, dict):
        items = [items]

    if not isinstance(items, list):
        return out

    for item in items:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("candlesticks"), list):
            mt = item.get("market_ticker") or item.get("ticker")
            if not mt and len(fallback_list) == 1:
                mt = fallback_list[0]
            if not mt:
                continue
            out.setdefault(mt, []).extend(item.get("candlesticks") or [])
        else:
            mt = item.get("market_ticker") or item.get("ticker")
            if not mt and len(fallback_list) == 1:
                mt = fallback_list[0]
            if not mt:
                continue
            out.setdefault(mt, []).append(item)

    return out


def _chunk_list(values: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


def _chunk_time_ranges(start_ts: int, end_ts: int, max_minutes: int) -> Iterable[Tuple[int, int]]:
    if start_ts > end_ts:
        return
    chunk_s = max(60, int(max_minutes) * 60)
    t0 = int(start_ts)
    while t0 <= end_ts:
        t1 = min(end_ts, t0 + chunk_s - 1)
        yield t0, t1
        t0 = t1 + 1


def _log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts}  {msg}")


def _cutoff_epoch(cutoff_resp: Dict[str, Any]) -> Optional[int]:
    raw = cutoff_resp.get("market_settled_ts")
    if raw is None:
        raw = cutoff_resp.get("cutoff_ts")
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str):
        if raw.isdigit():
            return int(raw)
        parsed = _parse_iso_to_epoch(raw)
        if parsed is not None:
            return parsed
    return None


def _market_settle_epoch(market: Dict[str, Any]) -> Optional[int]:
    for field in ("settlement_ts", "close_time", "latest_expiration_time", "expiration_time"):
        ts = _parse_iso_to_epoch(market.get(field))
        if ts is not None:
            return ts
    return None


def _market_open_epoch(market: Dict[str, Any]) -> Optional[int]:
    return _parse_iso_to_epoch(market.get("open_time"))


def _market_close_epoch(market: Dict[str, Any]) -> Optional[int]:
    return _parse_iso_to_epoch(market.get("close_time"))


def _write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _prepare_out_dir(out_dir: str) -> Path:
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def _fetch_live_candles(
    client: KalshiClient,
    market_tickers: List[str],
    start_ts: int,
    end_ts: int,
    errors: List[str],
) -> Dict[str, Dict[int, str]]:
    prices: Dict[str, Dict[int, str]] = {t: {} for t in market_tickers}
    for ticker_group in _chunk_list(market_tickers, MAX_TICKERS_PER_BATCH):
        group_size = max(1, len(ticker_group))
        max_minutes = max(1, MAX_POINTS_TOTAL // group_size)
        for t0, t1 in _chunk_time_ranges(start_ts, end_ts, max_minutes):
            try:
                resp = client.get_batch_candlesticks(
                    ticker_group,
                    start_ts=t0,
                    end_ts=t1,
                    period_interval=1,
                )
            except Exception as exc:
                errors.append(f"batch_candlesticks_failed {ticker_group} {t0}-{t1}: {exc}")
                continue
            grouped = _extract_candles_by_market(resp, ticker_group)
            for mt, candles in grouped.items():
                for candle in candles:
                    end_ts_val = candle.get("end_period_ts")
                    if end_ts_val is None:
                        continue
                    try:
                        end_ts_int = int(end_ts_val)
                    except Exception:
                        continue
                    price = _price_cents_from_candle(candle)
                    if price is None:
                        continue
                    prices.setdefault(mt, {})[end_ts_int] = price
    return prices


def _fetch_historical_candles(
    client: KalshiClient,
    market_tickers: List[str],
    start_ts: int,
    end_ts: int,
    errors: List[str],
) -> Dict[str, Dict[int, str]]:
    prices: Dict[str, Dict[int, str]] = {t: {} for t in market_tickers}
    for mt in market_tickers:
        for t0, t1 in _chunk_time_ranges(start_ts, end_ts, HISTORICAL_CHUNK_MINUTES):
            try:
                resp = client.get_historical_candlesticks(
                    mt,
                    start_ts=t0,
                    end_ts=t1,
                    period_interval=1,
                )
            except Exception as exc:
                errors.append(f"historical_candlesticks_failed {mt} {t0}-{t1}: {exc}")
                break
            grouped = _extract_candles_by_market(resp, [mt])
            candles = grouped.get(mt, [])
            for candle in candles:
                end_ts_val = candle.get("end_period_ts")
                if end_ts_val is None:
                    continue
                try:
                    end_ts_int = int(end_ts_val)
                except Exception:
                    continue
                price = _price_cents_from_candle(candle)
                if price is None:
                    continue
                prices.setdefault(mt, {})[end_ts_int] = price
    return prices


def _write_csv(
    path: Path,
    header: List[str],
    rows: List[List[str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    os.replace(tmp, path)


def _build_rows(
    market_order: List[Dict[str, Any]],
    prices_by_market: Dict[str, Dict[int, str]],
) -> Tuple[List[str], List[List[str]]]:
    columns = ["timestamp"]
    market_tickers: List[str] = []
    for m in market_order:
        subtitle = m.get("subtitle") or m.get("title") or m.get("ticker") or ""
        columns.append(subtitle)
        market_tickers.append(m.get("ticker") or "")

    row_map: Dict[int, Dict[str, str]] = {}
    for m in market_order:
        ticker = m.get("ticker") or ""
        if not ticker:
            continue
        prices = prices_by_market.get(ticker, {})
        col_name = m.get("subtitle") or m.get("title") or m.get("ticker") or ""
        for ts, price in prices.items():
            row = row_map.setdefault(int(ts), {})
            row[col_name] = price

    rows: List[List[str]] = []
    for ts in sorted(row_map.keys()):
        row_data = row_map[ts]
        row = [_epoch_to_iso(ts)]
        for col_name in columns[1:]:
            row.append(row_data.get(col_name, ""))
        rows.append(row)

    return columns, rows


def _load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Kalshi KXHIGHMIA minute bucket prices.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="End date YYYY-MM-DD")
    parser.add_argument("--series", default=DEFAULT_SERIES, help="Kalshi series ticker (default: KXHIGHMIA)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--base-url", default=None, help="Override Kalshi API base URL")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--sleep-on-rate-limit", action="store_true", default=DEFAULT_SLEEP_ON_RATE_LIMIT)
    parser.add_argument("--no-sleep-on-rate-limit", action="store_false", dest="sleep_on_rate_limit")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=DEFAULT_SKIP_EXISTING,
        help="Skip days whose output CSV already exists and manifest shows success (default: true)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Do not skip existing output CSVs",
    )
    args = parser.parse_args()

    start_date = dt.date.fromisoformat(args.start_date)
    end_date = dt.date.fromisoformat(args.end_date)
    if end_date < start_date:
        raise SystemExit("--end-date must be >= --start-date")

    out_dir = _prepare_out_dir(args.out_dir)
    manifest_path = out_dir / "manifest.json"

    client = KalshiClient(
        base_url=args.base_url or DEFAULT_BASE_URL,
        max_retries=args.max_retries,
        sleep_on_rate_limit=args.sleep_on_rate_limit,
    )

    _log("Fetching historical cutoff...")
    cutoff_resp = client.get_cutoff()
    cutoff_ts = _cutoff_epoch(cutoff_resp)
    _log(f"Cutoff market_settled_ts={cutoff_ts}")

    manifest = _load_manifest(manifest_path)
    manifest.setdefault("series", args.series)
    manifest.setdefault("generated_at_utc", _utc_now_iso())
    manifest["start_date"] = args.start_date
    manifest["end_date"] = args.end_date
    manifest["out_dir"] = str(out_dir)
    manifest["cutoff_ts"] = cutoff_ts
    manifest.setdefault("dates", {})

    for day in _daterange(start_date, end_date):
        day_key = day.isoformat()
        out_name = f"KMIA_{day.strftime('%Y%m%d')}.csv"
        out_path = out_dir / out_name
        if args.skip_existing and out_path.exists():
            prior = (manifest.get("dates") or {}).get(day_key) or {}
            prior_errors = prior.get("errors") or []
            prior_rows = prior.get("rows_written") or 0
            if isinstance(prior_errors, list) and len(prior_errors) == 0 and int(prior_rows) > 0:
                _log(f"Skipping existing {out_path} (manifest rows_written={prior_rows})")
                continue
        errors: List[str] = []
        event_ticker, event_resp, resolve_errors = _resolve_event_for_day(args.series, day, client)
        errors.extend(resolve_errors)
        if not event_ticker or not event_resp:
            _log(f"Unable to resolve event ticker for {day_key}; errors={errors}")
            manifest["dates"][day_key] = ManifestEntry(
                date=day_key,
                event_ticker=event_ticker or "",
                market_tickers=[],
                start_time=None,
                end_time=None,
                start_ts=None,
                end_ts=None,
                rows_written=0,
                errors=errors,
            ).__dict__
            _write_manifest(manifest_path, manifest)
            continue

        _log(f"Processing {day_key} resolved_event={event_ticker}")

        _, markets = _extract_markets(event_resp)
        if not markets:
            _log("No markets from event; attempting historical markets endpoint")
            try:
                markets = client.get_historical_markets(event_ticker)
            except Exception as exc:
                errors.append(f"historical_markets_failed: {exc}")

        if not markets:
            errors.append("no_markets_found")
            _log(f"No markets found for {event_ticker}")
            manifest["dates"][day_key] = ManifestEntry(
                date=day_key,
                event_ticker=event_ticker,
                market_tickers=[],
                start_time=None,
                end_time=None,
                start_ts=None,
                end_ts=None,
                rows_written=0,
                errors=errors,
            ).__dict__
            _write_manifest(manifest_path, manifest)
            continue

        market_objs: List[Dict[str, Any]] = []
        for m in markets:
            if not isinstance(m, dict):
                continue
            ticker = m.get("ticker") or m.get("market_ticker") or ""
            if not ticker:
                continue
            m["ticker"] = ticker
            market_objs.append(m)

        if not market_objs:
            errors.append("no_valid_market_objects")
            _log(f"No valid market objects for {event_ticker}")
            manifest["dates"][day_key] = ManifestEntry(
                date=day_key,
                event_ticker=event_ticker,
                market_tickers=[],
                start_time=None,
                end_time=None,
                start_ts=None,
                end_ts=None,
                rows_written=0,
                errors=errors,
            ).__dict__
            _write_manifest(manifest_path, manifest)
            continue

        market_objs.sort(key=_market_sort_key)
        market_tickers = [m.get("ticker") for m in market_objs if m.get("ticker")]

        open_times = [t for t in (_market_open_epoch(m) for m in market_objs) if t is not None]
        close_times = [t for t in (_market_close_epoch(m) for m in market_objs) if t is not None]

        if not open_times or not close_times:
            errors.append("missing_open_or_close_time")
            _log(f"Missing open/close time for {event_ticker}")
            manifest["dates"][day_key] = ManifestEntry(
                date=day_key,
                event_ticker=event_ticker,
                market_tickers=market_tickers,
                start_time=None,
                end_time=None,
                start_ts=None,
                end_ts=None,
                rows_written=0,
                errors=errors,
            ).__dict__
            _write_manifest(manifest_path, manifest)
            continue

        start_ts = min(open_times)
        end_ts = max(close_times)
        start_iso = _epoch_to_iso(start_ts)
        end_iso = _epoch_to_iso(end_ts)

        live_markets: List[str] = []
        historical_markets: List[str] = []
        for m in market_objs:
            ticker = m.get("ticker")
            if not ticker:
                continue
            settle_ts = _market_settle_epoch(m)
            if cutoff_ts is not None and settle_ts is not None and settle_ts <= cutoff_ts:
                historical_markets.append(ticker)
            else:
                live_markets.append(ticker)

        prices_by_market: Dict[str, Dict[int, str]] = {}
        if live_markets:
            _log(f"Fetching live candlesticks: {len(live_markets)} markets")
            live_prices = _fetch_live_candles(client, live_markets, start_ts, end_ts, errors)
            prices_by_market.update(live_prices)

        if historical_markets:
            _log(f"Fetching historical candlesticks: {len(historical_markets)} markets")
            hist_prices = _fetch_historical_candles(client, historical_markets, start_ts, end_ts, errors)
            prices_by_market.update(hist_prices)

        header, rows = _build_rows(market_objs, prices_by_market)
        rows_written = len(rows)

        out_name = f"KMIA_{day.strftime('%Y%m%d')}.csv"
        out_path = out_dir / out_name
        _write_csv(out_path, header, rows)

        _log(f"Wrote {out_path} rows={rows_written}")

        manifest["dates"][day_key] = ManifestEntry(
            date=day_key,
            event_ticker=event_ticker,
            market_tickers=market_tickers,
            start_time=start_iso,
            end_time=end_iso,
            start_ts=start_ts,
            end_ts=end_ts,
            rows_written=rows_written,
            errors=errors,
        ).__dict__
        manifest["generated_at_utc"] = _utc_now_iso()
        _write_manifest(manifest_path, manifest)

    _log("Download complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
