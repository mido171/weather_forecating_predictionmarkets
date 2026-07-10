#!/usr/bin/env python3
"""
Thin Kalshi Trade API v2 client for public market-data endpoints.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests

DEFAULT_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_TIMEOUT = (10, 60)
DEFAULT_MAX_RETRIES = 6
DEFAULT_USER_AGENT = "weather-forecasting-predictionmarkets (kalshi minute downloader)"


class KalshiApiError(RuntimeError):
    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class KalshiClient:
    base_url: str = DEFAULT_BASE_URL
    timeout: tuple = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    sleep_on_rate_limit: bool = True
    user_agent: str = DEFAULT_USER_AGENT

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

    def _full_url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_url}{path}"

    def _sleep_backoff(self, attempt: int) -> None:
        base = 0.5 * (2 ** attempt)
        jitter = random.random() * 0.25
        time.sleep(min(30.0, base + jitter))

    def _request_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = self._full_url(path)
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 429:
                    if self.sleep_on_rate_limit:
                        retry_after = resp.headers.get("Retry-After")
                        if retry_after:
                            try:
                                time.sleep(float(retry_after))
                            except Exception:
                                self._sleep_backoff(attempt)
                        else:
                            self._sleep_backoff(attempt)
                        continue
                    raise KalshiApiError(f"HTTP 429 rate limit for {url}", status_code=resp.status_code)

                if resp.status_code in (500, 502, 503, 504):
                    self._sleep_backoff(attempt)
                    continue

                if not (200 <= resp.status_code < 300):
                    text = resp.text
                    raise KalshiApiError(
                        f"HTTP {resp.status_code} for {url} params={params} response={text[:500]}",
                        status_code=resp.status_code,
                    )

                try:
                    return resp.json()
                except json.JSONDecodeError as exc:
                    raise KalshiApiError(f"Invalid JSON response from {url}: {exc}")

            except KalshiApiError as exc:
                last_err = exc
                if isinstance(exc, KalshiApiError) and exc.status_code == 404:
                    raise
                if attempt >= self.max_retries:
                    raise
                self._sleep_backoff(attempt)
            except requests.RequestException as exc:
                last_err = exc
                if attempt >= self.max_retries:
                    raise KalshiApiError(f"Request failed for {url}: {exc}")
                self._sleep_backoff(attempt)

        raise KalshiApiError(f"Request failed after retries for {url}: {last_err}")

    def get_cutoff(self) -> Dict[str, Any]:
        return self._request_json("/historical/cutoff")

    def get_event(self, event_ticker: str) -> Dict[str, Any]:
        return self._request_json(f"/events/{event_ticker}", params={"with_nested_markets": "true"})

    def get_historical_markets(self, event_ticker: str) -> List[Dict[str, Any]]:
        markets: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        while True:
            params: Dict[str, Any] = {"event_ticker": event_ticker}
            if cursor:
                params["cursor"] = cursor
            resp = self._request_json("/historical/markets", params=params)
            batch = resp.get("markets") or []
            if isinstance(batch, list):
                markets.extend(batch)
            cursor = resp.get("cursor")
            if not cursor:
                break
        return markets

    def get_batch_candlesticks(
        self,
        market_tickers: Iterable[str],
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
    ) -> Dict[str, Any]:
        tickers = ",".join([t for t in market_tickers if t])
        params = {
            "market_tickers": tickers,
            "start_ts": int(start_ts),
            "end_ts": int(end_ts),
            "period_interval": int(period_interval),
        }
        return self._request_json("/markets/candlesticks", params=params)

    def get_historical_candlesticks(
        self,
        market_ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
    ) -> Dict[str, Any]:
        params = {
            "start_ts": int(start_ts),
            "end_ts": int(end_ts),
            "period_interval": int(period_interval),
        }
        return self._request_json(f"/historical/markets/{market_ticker}/candlesticks", params=params)


_default_client: Optional[KalshiClient] = None


def _get_default_client() -> KalshiClient:
    global _default_client
    if _default_client is None:
        _default_client = KalshiClient()
    return _default_client


def get_cutoff(client: Optional[KalshiClient] = None) -> Dict[str, Any]:
    return (client or _get_default_client()).get_cutoff()


def get_event(event_ticker: str, client: Optional[KalshiClient] = None) -> Dict[str, Any]:
    return (client or _get_default_client()).get_event(event_ticker)


def get_historical_markets(event_ticker: str, client: Optional[KalshiClient] = None) -> List[Dict[str, Any]]:
    return (client or _get_default_client()).get_historical_markets(event_ticker)


def get_batch_candlesticks(
    market_tickers: Iterable[str],
    start_ts: int,
    end_ts: int,
    period_interval: int = 1,
    client: Optional[KalshiClient] = None,
) -> Dict[str, Any]:
    return (client or _get_default_client()).get_batch_candlesticks(
        market_tickers=market_tickers,
        start_ts=start_ts,
        end_ts=end_ts,
        period_interval=period_interval,
    )


def get_historical_candlesticks(
    market_ticker: str,
    start_ts: int,
    end_ts: int,
    period_interval: int = 1,
    client: Optional[KalshiClient] = None,
) -> Dict[str, Any]:
    return (client or _get_default_client()).get_historical_candlesticks(
        market_ticker=market_ticker,
        start_ts=start_ts,
        end_ts=end_ts,
        period_interval=period_interval,
    )
