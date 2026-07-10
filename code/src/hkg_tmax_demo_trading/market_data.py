"""Polymarket Gamma/CLOB read-only adapters for HKG daily Tmax markets."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx
from hkg_tmax_probability.bucket_rules import BUCKET_KEYS

from .domain import hkg_event_slug_for_date, hkg_event_title_for_date, hkg_event_url_for_date

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
USER_AGENT = "HKG-Tmax-Demo-Backtester/0.1"
CLOB_PRICE_TIMEOUT_SECONDS = 3.0


class MarketDataUnavailable(RuntimeError):
    """Raised when public market metadata cannot be fetched or parsed."""


@dataclass(frozen=True)
class MarketBucket:
    bucket_key: str
    label: str
    market_id: str | None
    market_slug: str | None
    question: str | None
    active: bool
    closed: bool
    accepting_orders: bool
    volume: float | None
    liquidity: float | None
    yes_token: str | None
    no_token: str | None
    yes_fallback_price: Decimal | None
    no_fallback_price: Decimal | None
    market_probability: float | None


def _fetch_json(url: str, timeout: float = 20.0) -> Any:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return value


def _as_decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    decimal = _as_decimal(value)
    return None if decimal is None else float(decimal)


def normalize_bucket_label(label: str, question: str = "") -> str | None:
    text = f"{label} {question}".lower()
    text = text.replace("deg.", "").replace("degrees", "c").replace("degree", "c")
    if "24" in text and ("below" in text or "or less" in text or "or below" in text):
        return "24_or_below"
    if "34" in text and ("higher" in text or "above" in text or "or more" in text):
        return "34_or_higher"
    match = re.search(r"\b(2[5-9]|3[0-3])\s*(?:deg|c)?\b", text)
    if match:
        return match.group(1)
    return None


def event_shell(target_date: date) -> dict[str, Any]:
    return {
        "url": hkg_event_url_for_date(target_date),
        "slug": hkg_event_slug_for_date(target_date),
        "title": hkg_event_title_for_date(target_date),
        "gamma_event_id": None,
        "active": False,
        "closed": False,
        "archived": False,
        "volume": None,
    }


def fetch_event_for_date(target_date: date) -> dict[str, Any]:
    slug = hkg_event_slug_for_date(target_date)
    url = f"{GAMMA_BASE}/events/slug/{slug}"
    try:
        payload = _fetch_json(url)
    except Exception as exc:  # noqa: BLE001 - translated at service boundary.
        raise MarketDataUnavailable(f"Gamma event fetch failed for {slug}: {exc}") from exc
    if isinstance(payload, list):
        if not payload:
            raise MarketDataUnavailable(f"Gamma returned no event for {slug}")
        payload = payload[0]
    title = str(payload.get("title") or payload.get("slug") or "")
    slug_text = str(payload.get("slug") or slug)
    if "hong kong" not in title.lower() and "hong-kong" not in slug_text.lower():
        raise MarketDataUnavailable(f"Event is not an HKG highest-temperature market: {title}")
    return payload


def parse_market_buckets(event: dict[str, Any]) -> dict[str, MarketBucket]:
    buckets: dict[str, MarketBucket] = {}
    for market in event.get("markets", []) or []:
        label = str(market.get("groupItemTitle") or market.get("question") or "")
        question = str(market.get("question") or "")
        bucket_key = normalize_bucket_label(label, question)
        if bucket_key is None or bucket_key not in BUCKET_KEYS:
            continue
        outcomes = _parse_jsonish(market.get("outcomes") or [])
        token_ids = _parse_jsonish(market.get("clobTokenIds") or [])
        prices = _parse_jsonish(market.get("outcomePrices") or [])
        yes_token = None
        no_token = None
        yes_fallback = None
        no_fallback = None
        for index, outcome in enumerate(outcomes if isinstance(outcomes, list) else []):
            name = str(outcome).strip().lower()
            token = str(token_ids[index]) if isinstance(token_ids, list) and index < len(token_ids) else None
            fallback = _as_decimal(prices[index]) if isinstance(prices, list) and index < len(prices) else None
            if name == "yes":
                yes_token = token
                yes_fallback = fallback
            elif name == "no":
                no_token = token
                no_fallback = fallback
        buckets[bucket_key] = MarketBucket(
            bucket_key=bucket_key,
            label=label or bucket_key,
            market_id=None if market.get("id") is None else str(market.get("id")),
            market_slug=None if market.get("slug") is None else str(market.get("slug")),
            question=question or None,
            active=bool(market.get("active")),
            closed=bool(market.get("closed")),
            accepting_orders=bool(market.get("acceptingOrders")),
            volume=_as_float(market.get("volume")),
            liquidity=_as_float(market.get("liquidity")),
            yes_token=yes_token,
            no_token=no_token,
            yes_fallback_price=yes_fallback,
            no_fallback_price=no_fallback,
            market_probability=None if yes_fallback is None else float(yes_fallback),
        )
    return buckets


def clob_buy_price(
    token_id: str | None,
    timeout: float = CLOB_PRICE_TIMEOUT_SECONDS,
) -> tuple[Decimal | None, str]:
    if not token_id:
        return None, "missing_token"
    url = f"{CLOB_BASE}/price"
    try:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        with httpx.Client(timeout=timeout, headers=headers) as client:
            response = client.get(url, params={"token_id": token_id, "side": "SELL"})
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:  # noqa: BLE001 - returned as source metadata.
        return None, f"clob_error:{type(exc).__name__}"
    price = _as_decimal(payload.get("price"))
    if price is None:
        return None, "clob_missing_price"
    return price, "clob_ask"


def event_view_from_gamma(target_date: date, event: dict[str, Any] | None) -> dict[str, Any]:
    shell = event_shell(target_date)
    if not event:
        return shell
    return {
        "url": shell["url"],
        "slug": hkg_event_slug_for_date(target_date),
        "title": str(event.get("title") or shell["title"]),
        "gamma_event_id": None if event.get("id") is None else str(event.get("id")),
        "active": bool(event.get("active")),
        "closed": bool(event.get("closed")),
        "archived": bool(event.get("archived")),
        "volume": _as_float(event.get("volume")),
    }
