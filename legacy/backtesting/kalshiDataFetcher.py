#!/usr/bin/env python3
"""
kalshiDataFetcher.py

Goal
----
For a given Miami day (KXHIGHMIA series), produce ONE JSON FILE for that day.
Inside that JSON:
  - one object per bucket market (each bucket is a binary market within the daily event)
  - inside each bucket object: a minute-by-minute series of YES prices from market open until settlement.

Where the data comes from (Kalshi Trade API v2)
------------------------------------------------
Base URL for public market data (no auth required for many endpoints):
  https://api.elections.kalshi.com/trade-api/v2

We use these endpoints:
  1) Get Event (to list bucket markets)
     GET /events/{event_ticker}?with_nested_markets=true

  2) Get Market Candlesticks (minute candles)
     GET /series/{series_ticker}/markets/{market_ticker}/candlesticks
       ?start_ts=<unix seconds>
       &end_ts=<unix seconds>
       &period_interval=1

How we define the "YES price" per minute
----------------------------------------
Kalshi candlesticks include a `price` object which is a "PriceDistribution".
According to Kalshi's API model docs:
  - price.close_dollars = last traded YES contract price during the candlestick period
  - price.previous_dollars = last traded YES contract price *before* the candlestick period

So for each 1-minute candlestick we output:
  YES_PRICE = price.close_dollars if present
          else price.previous_dollars if present
          else null

This produces a minute-by-minute series that carries forward the last traded
price (typical for charting). Minutes with no trades and no prior trade stay null.

Output format (per day)
-----------------------
The output JSON for a day looks like:

{
  "location": "Miami, FL",
  "series_ticker": "KXHIGHMIA",
  "date": "YYYY-MM-DD",
  "event_ticker": "KXHIGHMIA-26JAN18",
  "strike_date": "...",
  "generated_at_utc": "...",
  "period_interval_minutes": 1,
  "buckets": [
    {
      "market_ticker": "KXHIGHMIA-26JAN18-B82.5",
      "title": "...",
      "subtitle": "...",
      "floor_strike": 123,
      "cap_strike": 123,
      "open_time": "...",
      "close_time": "...",
      "settlement_ts": "...",
      "result": "yes" | "no" | null,
      "yes_prices_1m": [
        {"end_period_ts": 1739726400, "yes_price_dollars": "0.5600"},
        ...
      ]
    },
    ...
  ]
}

Usage
-----
Install dependencies:
  pip install requests cryptography

Fetch a single day:
  python backtesting/kalshiDataFetcher.py --date 2026-01-18 --out-dir backtesting/out/

Fetch a range (inclusive):
  python backtesting/kalshiDataFetcher.py --date-from 2026-01-01 --date-to 2026-01-18 --out-dir backtesting/out/

Optional: if you ever get HTTP 401, you can sign requests with API keys:
  python backtesting/kalshiDataFetcher.py --date 2026-01-18 \
    --api-key-id <YOUR_KEY_ID> --private-key-path /path/to/private.key \
    --host https://api.kalshi.com \
    --out-dir backtesting/out/
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP, getcontext
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

# High precision for converting cents->dollars, etc.
getcontext().prec = 28


def parse_iso8601_z(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def iso8601_to_epoch_seconds(ts: Optional[str]) -> Optional[int]:
    if not ts:
        return None
    return int(parse_iso8601_z(ts).timestamp())


def epoch_seconds_now() -> int:
    return int(time.time())


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def ceil_to_multiple(ts: int, step: int) -> int:
    return ceil_div(ts, step) * step


def floor_to_multiple(ts: int, step: int) -> int:
    return (ts // step) * step


def try_zoneinfo(tz_name: str):
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(tz_name)
    except Exception:
        return None


def _as_decimal(v: Any) -> Optional[Decimal]:
    if v is None:
        return None
    if isinstance(v, Decimal):
        return v
    if isinstance(v, (int, float)):
        return Decimal(str(v))
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return None
        return Decimal(s)
    return None


def dollars_from_dollars_or_cents(dollars_field: Any, cents_field: Any) -> Optional[str]:
    d = _as_decimal(dollars_field)
    if d is not None:
        return format(d.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP), "f")

    c = _as_decimal(cents_field)
    if c is None:
        return None

    d2 = (c / Decimal("100")).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    return format(d2, "f")


@dataclass
class KalshiAuth:
    api_key_id: str
    private_key_pem_bytes: bytes

    def __post_init__(self) -> None:
        try:
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding
        except Exception as exc:
            raise RuntimeError(
                "cryptography is required for authenticated requests. Install it with:\n"
                "  pip install cryptography\n"
                f"Original import error: {exc}"
            )

        self._hashes = hashes
        self._padding = padding
        self._serialization = serialization
        self._default_backend = default_backend

        self._private_key = self._serialization.load_pem_private_key(
            self.private_key_pem_bytes,
            password=None,
            backend=self._default_backend(),
        )

    def sign(self, timestamp_ms: str, method: str, path: str) -> str:
        path_wo_query = path.split("?", 1)[0]
        msg = f"{timestamp_ms}{method.upper()}{path_wo_query}".encode("utf-8")

        sig = self._private_key.sign(
            msg,
            self._padding.PSS(
                mgf=self._padding.MGF1(self._hashes.SHA256()),
                salt_length=self._padding.PSS.DIGEST_LENGTH,
            ),
            self._hashes.SHA256(),
        )
        return base64.b64encode(sig).decode("utf-8")

    def headers_for(self, method: str, path: str) -> Dict[str, str]:
        ts_ms = str(int(time.time() * 1000))
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "KALSHI-ACCESS-SIGNATURE": self.sign(ts_ms, method, path),
        }


class KalshiClient:
    def __init__(
        self,
        host: str,
        base_path: str,
        auth: Optional[KalshiAuth] = None,
        timeout_s: int = 30,
        max_retries: int = 6,
    ) -> None:
        self.host = host.rstrip("/")
        self.base_path = base_path.rstrip("/")
        self.auth = auth
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.session = requests.Session()

    def _full_url(self, path: str) -> str:
        return f"{self.host}{path}"

    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        method = method.upper()
        url = self._full_url(path)

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                headers: Dict[str, str] = {}
                if self.auth is not None:
                    headers.update(self.auth.headers_for(method, path))

                resp = self.session.request(
                    method,
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout_s,
                )

                if resp.status_code == 401 and self.auth is None:
                    raise RuntimeError(
                        f"HTTP 401 Unauthorized for {method} {url}. "
                        "Provide --api-key-id and --private-key-path (and likely --host https://api.kalshi.com)."
                    )

                if resp.status_code in (429, 500, 502, 503, 504):
                    sleep_s = min(10.0, 0.5 * (2 ** attempt)) + random.random() * 0.25
                    time.sleep(sleep_s)
                    continue

                if not (200 <= resp.status_code < 300):
                    raise RuntimeError(
                        f"HTTP {resp.status_code} for {method} {url} params={params}\n"
                        f"Response: {resp.text[:2000]}"
                    )

                return resp.json()

            except Exception as exc:
                last_err = exc
                if attempt >= self.max_retries:
                    break
                sleep_s = min(10.0, 0.5 * (2 ** attempt)) + random.random() * 0.25
                time.sleep(sleep_s)

        raise RuntimeError(f"Request failed after retries: {method} {url} params={params}\nLast error: {last_err}")

    def get_event(self, event_ticker: str, with_nested_markets: bool = True) -> Dict[str, Any]:
        path = f"{self.base_path}/events/{event_ticker}"
        params = {"with_nested_markets": "true" if with_nested_markets else "false"}
        return self._request("GET", path, params=params)

    def list_events(
        self,
        series_ticker: str,
        status: Optional[str] = None,
        limit: int = 200,
        cursor: Optional[str] = None,
        with_nested_markets: bool = False,
    ) -> Dict[str, Any]:
        path = f"{self.base_path}/events"
        params: Dict[str, Any] = {
            "series_ticker": series_ticker,
            "limit": int(limit),
            "with_nested_markets": "true" if with_nested_markets else "false",
        }
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", path, params=params)

    def get_market_candlesticks(
        self,
        series_ticker: str,
        market_ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval_min: int = 1,
        include_latest_before_start: bool = False,
    ) -> Dict[str, Any]:
        path = f"{self.base_path}/series/{series_ticker}/markets/{market_ticker}/candlesticks"
        params: Dict[str, Any] = {
            "start_ts": int(start_ts),
            "end_ts": int(end_ts),
            "period_interval": int(period_interval_min),
        }
        if include_latest_before_start:
            params["include_latest_before_start"] = "true"
        return self._request("GET", path, params=params)


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


def event_ticker_from_date(series_ticker: str, day: date) -> str:
    yy = day.year % 100
    mm = MONTH_ABBR[day.month]
    dd = f"{day.day:02d}"
    return f"{series_ticker.upper()}-{yy:02d}{mm}{dd}"


def _event_matches_date(ev: Dict[str, Any], target_date: date, tz_name: str) -> bool:
    strike = ev.get("strike_date")
    if not strike:
        return False

    try:
        dt_utc = parse_iso8601_z(strike)
    except Exception:
        return False

    tz = try_zoneinfo(tz_name)
    if tz is None:
        return dt_utc.date() == target_date

    return dt_utc.astimezone(tz).date() == target_date


def resolve_event_for_date(
    client: KalshiClient,
    series_ticker: str,
    target_date: date,
    tz_name: str,
) -> Dict[str, Any]:
    candidate_ticker = event_ticker_from_date(series_ticker, target_date)
    try:
        ev_resp = client.get_event(candidate_ticker, with_nested_markets=True)
        ev = ev_resp.get("event") or {}
        if _event_matches_date(ev, target_date, tz_name):
            return ev_resp
    except Exception:
        pass

    statuses_to_try = ["settled", "closed", "open", None]
    for status in statuses_to_try:
        cursor: Optional[str] = None
        while True:
            resp = client.list_events(
                series_ticker=series_ticker,
                status=status,
                limit=200,
                cursor=cursor,
                with_nested_markets=False,
            )
            for ev in resp.get("events", []) or []:
                if _event_matches_date(ev, target_date, tz_name):
                    return client.get_event(ev.get("event_ticker") or ev.get("ticker"), with_nested_markets=True)

            cursor = resp.get("cursor")
            if not cursor:
                break

    raise RuntimeError(f"No event found for series={series_ticker} on date={target_date}")


def extract_markets_from_get_event_response(ev_resp: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    ev = ev_resp.get("event") or {}

    markets = ev.get("markets")
    if isinstance(markets, list) and markets:
        return ev, markets

    top_markets = ev_resp.get("markets")
    if isinstance(top_markets, list):
        return ev, top_markets

    return ev, []


def choose_market_end_ts(market: Dict[str, Any]) -> Tuple[int, str]:
    for field in ("settlement_ts", "close_time", "latest_expiration_time", "expiration_time"):
        ep = iso8601_to_epoch_seconds(market.get(field))
        if ep is not None:
            return ep, field
    return epoch_seconds_now(), "now"


def choose_market_start_ts(market: Dict[str, Any]) -> Tuple[int, str]:
    for field in ("open_time", "created_time"):
        ep = iso8601_to_epoch_seconds(market.get(field))
        if ep is not None:
            return ep, field
    return epoch_seconds_now(), "now"


def fetch_all_candles(
    client: KalshiClient,
    series_ticker: str,
    market_ticker: str,
    start_ts: int,
    end_ts: int,
    period_interval_min: int = 1,
    chunk_minutes: int = 1440,
    include_latest_before_start: bool = False,
) -> List[Dict[str, Any]]:
    step_s = int(period_interval_min) * 60
    if step_s <= 0:
        raise ValueError("period_interval_min must be positive")

    chunk_s = max(step_s, int(chunk_minutes) * 60)

    if start_ts > end_ts:
        return []

    all_by_end_ts: Dict[int, Dict[str, Any]] = {}

    t0 = int(start_ts)
    while t0 <= end_ts:
        t1 = min(end_ts, t0 + chunk_s - 1)

        resp = client.get_market_candlesticks(
            series_ticker=series_ticker,
            market_ticker=market_ticker,
            start_ts=t0,
            end_ts=t1,
            period_interval_min=period_interval_min,
            include_latest_before_start=include_latest_before_start,
        )

        for c in resp.get("candlesticks", []) or []:
            try:
                end_period_ts = int(c.get("end_period_ts"))
            except Exception:
                continue
            all_by_end_ts[end_period_ts] = c

        t0 = t1 + 1

    return [all_by_end_ts[k] for k in sorted(all_by_end_ts.keys())]


def yes_price_from_candle(candle: Dict[str, Any]) -> Optional[str]:
    price = candle.get("price") or {}

    close_dollars = dollars_from_dollars_or_cents(price.get("close_dollars"), price.get("close"))
    if close_dollars is not None:
        return close_dollars

    prev_dollars = dollars_from_dollars_or_cents(price.get("previous_dollars"), price.get("previous"))
    if prev_dollars is not None:
        return prev_dollars

    return None


def build_minute_series(
    candles: List[Dict[str, Any]],
    start_ts: int,
    end_ts: int,
    carry_forward: bool = True,
    period_interval_min: int = 1,
) -> List[Dict[str, Any]]:
    step = int(period_interval_min) * 60
    if step <= 0:
        raise ValueError("period_interval_min must be positive")

    start_boundary = ceil_to_multiple(int(start_ts), step)
    end_boundary = floor_to_multiple(int(end_ts), step)

    if end_boundary < start_boundary:
        return []

    by_ts: Dict[int, Optional[str]] = {}
    for candle in candles:
        try:
            t = int(candle.get("end_period_ts"))
        except Exception:
            continue
        by_ts[t] = yes_price_from_candle(candle)

    out: List[Dict[str, Any]] = []
    last_known: Optional[str] = None

    t = start_boundary
    while t <= end_boundary:
        if t in by_ts:
            price = by_ts[t]
            if price is None and carry_forward:
                price = last_known
        else:
            price = last_known if carry_forward else None

        if price is not None:
            last_known = price

        end_period_utc = datetime.fromtimestamp(t, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        out.append({"end_period_ts": t, "end_period_utc": end_period_utc, "yes_price_dollars": price})
        t += step

    return out


def build_day_json(
    client: KalshiClient,
    series_ticker: str,
    target_date: date,
    tz_name: str,
    chunk_minutes: int = 1440,
    carry_forward: bool = True,
) -> Dict[str, Any]:
    ev_resp = resolve_event_for_date(client, series_ticker, target_date, tz_name)
    ev, markets = extract_markets_from_get_event_response(ev_resp)

    event_ticker = (ev.get("event_ticker") or ev.get("ticker") or "").upper()

    out: Dict[str, Any] = {
        "location": "Miami, FL",
        "series_ticker": series_ticker,
        "date": target_date.isoformat(),
        "event_ticker": event_ticker,
        "strike_date": ev.get("strike_date"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "period_interval_minutes": 1,
        "buckets": [],
    }

    for m in markets:
        market_ticker = (m.get("ticker") or "").upper()
        if not market_ticker:
            continue

        start_ts, start_source = choose_market_start_ts(m)
        end_ts, end_source = choose_market_end_ts(m)

        candles = fetch_all_candles(
            client=client,
            series_ticker=series_ticker,
            market_ticker=market_ticker,
            start_ts=start_ts,
            end_ts=end_ts,
            period_interval_min=1,
            chunk_minutes=chunk_minutes,
            include_latest_before_start=False,
        )

        yes_series = build_minute_series(
            candles=candles,
            start_ts=start_ts,
            end_ts=end_ts,
            carry_forward=carry_forward,
            period_interval_min=1,
        )

        out["buckets"].append(
            {
                "market_ticker": market_ticker,
                "title": m.get("title"),
                "subtitle": m.get("subtitle"),
                "floor_strike": m.get("floor_strike"),
                "cap_strike": m.get("cap_strike"),
                "strike_type": m.get("strike_type"),
                "status": m.get("status"),
                "result": m.get("result"),
                "open_time": m.get("open_time"),
                "close_time": m.get("close_time"),
                "settlement_ts": m.get("settlement_ts"),
                "end_ts_source_used": end_source,
                "start_ts_source_used": start_source,
                "yes_prices_1m": yes_series,
            }
        )

    out["buckets"].sort(key=lambda x: (x.get("title") or "", x.get("market_ticker") or ""))
    return out


def daterange_inclusive(d0: date, d1: date) -> Iterable[date]:
    if d1 < d0:
        raise ValueError("date-to must be >= date-from")
    cur = d0
    while cur <= d1:
        yield cur
        cur = cur + timedelta(days=1)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Export minute-by-minute YES price history for every KXHIGHMIA bucket (market) for a given day. "
            "Writes one JSON per day."
        )
    )

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="Single date in YYYY-MM-DD (Miami local date)")
    group.add_argument("--date-from", help="Start date in YYYY-MM-DD (inclusive)")

    ap.add_argument("--date-to", default="", help="End date in YYYY-MM-DD (inclusive). Required if using --date-from.")

    ap.add_argument("--out-dir", default=".", help="Directory to write JSON files into")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON (indent=2).")

    ap.add_argument("--series", default="KXHIGHMIA", help="Series ticker (default: KXHIGHMIA)")
    ap.add_argument("--tz", default="America/New_York", help="Timezone for matching strike_date to your requested date")

    ap.add_argument(
        "--host",
        default="https://api.elections.kalshi.com",
        help=(
            "API host. Default is the public market-data host. "
            "If using authenticated requests, you may want https://api.kalshi.com (prod) or https://demo-api.kalshi.co (demo)."
        ),
    )
    ap.add_argument("--base-path", default="/trade-api/v2", help="API base path (default: /trade-api/v2)")

    ap.add_argument(
        "--chunk-minutes",
        type=int,
        default=1440,
        help="Fetch candlesticks in chunks of this many minutes (default: 1440 = 1 day)",
    )
    ap.add_argument(
        "--no-carry-forward",
        action="store_true",
        help="If set, minutes with no data are null instead of carrying forward the last known price.",
    )

    ap.add_argument("--api-key-id", default="", help="(Optional) API key id for signing")
    ap.add_argument("--private-key-path", default="", help="(Optional) Path to your private key .key/.pem file")

    args = ap.parse_args(argv)

    series_ticker = args.series.upper()

    auth: Optional[KalshiAuth] = None
    if args.api_key_id or args.private_key_path:
        if not (args.api_key_id and args.private_key_path):
            ap.error("If using authentication, you must supply BOTH --api-key-id and --private-key-path")
        with open(args.private_key_path, "rb") as f:
            key_bytes = f.read()
        auth = KalshiAuth(api_key_id=args.api_key_id, private_key_pem_bytes=key_bytes)

    client = KalshiClient(
        host=args.host,
        base_path=args.base_path,
        auth=auth,
        timeout_s=30,
        max_retries=6,
    )

    if args.date:
        d0 = date.fromisoformat(args.date)
        dates = [d0]
    else:
        if not args.date_to:
            ap.error("--date-to is required when using --date-from")
        d0 = date.fromisoformat(args.date_from)
        d1 = date.fromisoformat(args.date_to)
        dates = list(daterange_inclusive(d0, d1))

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    carry_forward = not args.no_carry_forward

    combined_days: List[Dict[str, Any]] = []
    for day in dates:
        day_obj = build_day_json(
            client=client,
            series_ticker=series_ticker,
            target_date=day,
            tz_name=args.tz,
            chunk_minutes=args.chunk_minutes,
            carry_forward=carry_forward,
        )
        combined_days.append(day_obj)

        filename = f"{series_ticker}_{day.isoformat()}.json"
        path = out_dir.rstrip("/\\") + os.sep + filename

        json_text = json.dumps(
            day_obj,
            indent=2 if args.pretty else None,
            separators=None if args.pretty else (",", ":"),
            ensure_ascii=False,
        )

        with open(path, "w", encoding="utf-8") as f:
            f.write(json_text)

        print(f"Wrote {path}", file=sys.stderr)

    if combined_days:
        first_day = dates[0].isoformat()
        last_day = dates[-1].isoformat()
        combined_obj = {
            "location": "Miami, FL",
            "series_ticker": series_ticker,
            "date_from": first_day,
            "date_to": last_day,
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "days": combined_days,
        }
        combined_name = f"{series_ticker}_{first_day}_to_{last_day}.json"
        combined_path = out_dir.rstrip("/\\") + os.sep + combined_name
        combined_text = json.dumps(
            combined_obj,
            indent=2 if args.pretty else None,
            separators=None if args.pretty else (",", ":"),
            ensure_ascii=False,
        )
        with open(combined_path, "w", encoding="utf-8") as f:
            f.write(combined_text)
        print(f"Wrote {combined_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
