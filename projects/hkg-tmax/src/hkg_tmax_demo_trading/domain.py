"""Domain math for fictitious HKG Polymarket demo trades."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from zoneinfo import ZoneInfo

from hkg_tmax_probability.bucket_rules import BUCKET_KEYS

UTC = ZoneInfo("UTC")
HKT = ZoneInfo("Asia/Hong_Kong")

DEFAULT_START_BALANCE_USD = Decimal("1000.00")
DEFAULT_WINDOW_START = date(2026, 7, 1)
DEFAULT_WINDOW_END = date(2026, 7, 10)

MONTH_SLUGS = {
    1: "january",
    2: "february",
    3: "march",
    4: "april",
    5: "may",
    6: "june",
    7: "july",
    8: "august",
    9: "september",
    10: "october",
    11: "november",
    12: "december",
}

QUANT_6 = Decimal("0.000001")
QUANT_8 = Decimal("0.00000001")
QUANT_10 = Decimal("0.0000000001")
QUANT_CENTS = Decimal("0.0001")


class DemoTradingError(ValueError):
    """Expected domain error suitable for a 4xx API response."""


@dataclass(frozen=True)
class TradeComputation:
    stake_usd: Decimal
    shares: Decimal
    entry_price_cents: Decimal
    model_probability_bucket: Decimal
    model_win_probability: Decimal
    edge_pp: Decimal
    ev_usd: Decimal


def decimal_value(value: object, field_name: str) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise DemoTradingError(f"{field_name} must be numeric") from exc


def quantize(value: Decimal, quantum: Decimal = QUANT_6) -> Decimal:
    return value.quantize(quantum, rounding=ROUND_HALF_UP)


def normalize_side(side: str) -> str:
    normalized = str(side).strip().lower()
    if normalized not in {"yes", "no"}:
        raise DemoTradingError("side must be yes or no")
    return normalized


def validate_bucket_key(bucket_key: str) -> str:
    normalized = str(bucket_key).strip()
    if normalized not in BUCKET_KEYS:
        raise DemoTradingError(f"Unsupported HKG bucket: {bucket_key}")
    return normalized


def validate_price_cents(value: object) -> Decimal:
    price = decimal_value(value, "entry price").quantize(QUANT_CENTS, rounding=ROUND_HALF_UP)
    if price <= 0 or price > 100:
        raise DemoTradingError("entry price must be > 0 and <= 100 cents")
    return price


def validate_stake_usd(value: object) -> Decimal:
    stake = decimal_value(value, "stake").quantize(QUANT_6, rounding=ROUND_HALF_UP)
    if stake <= 0:
        raise DemoTradingError("stake must be greater than zero")
    return stake


def probability_decimal(value: object, field_name: str) -> Decimal:
    probability = decimal_value(value, field_name).quantize(QUANT_10, rounding=ROUND_HALF_UP)
    if probability < 0 or probability > 1:
        raise DemoTradingError(f"{field_name} must be between 0 and 1")
    return probability


def win_probability_for_side(bucket_probability: Decimal, side: str) -> Decimal:
    normalized_side = normalize_side(side)
    return bucket_probability if normalized_side == "yes" else Decimal("1") - bucket_probability


def edge_pp(model_win_probability: Decimal, entry_price_cents: Decimal) -> Decimal:
    return (model_win_probability * Decimal("100") - entry_price_cents).quantize(
        QUANT_CENTS, rounding=ROUND_HALF_UP
    )


def shares_for_stake(stake_usd: Decimal, entry_price_cents: Decimal) -> Decimal:
    price_usd = entry_price_cents / Decimal("100")
    return (stake_usd / price_usd).quantize(QUANT_8, rounding=ROUND_HALF_UP)


def expected_value_usd(
    stake_usd: Decimal,
    shares: Decimal,
    model_win_probability: Decimal,
) -> Decimal:
    expected_payout = shares * model_win_probability
    return (expected_payout - stake_usd).quantize(QUANT_6, rounding=ROUND_HALF_UP)


def compute_trade(
    *,
    stake_usd: object,
    entry_price_cents: object,
    model_probability_bucket: object,
    side: str,
) -> TradeComputation:
    stake = validate_stake_usd(stake_usd)
    price = validate_price_cents(entry_price_cents)
    bucket_probability = probability_decimal(model_probability_bucket, "model_probability_bucket")
    win_probability = win_probability_for_side(bucket_probability, side)
    shares = shares_for_stake(stake, price)
    return TradeComputation(
        stake_usd=stake,
        shares=shares,
        entry_price_cents=price,
        model_probability_bucket=bucket_probability,
        model_win_probability=win_probability,
        edge_pp=edge_pp(win_probability, price),
        ev_usd=expected_value_usd(stake, shares, win_probability),
    )


def realized_pnl_usd(*, stake_usd: Decimal, shares: Decimal, did_win: bool) -> Decimal:
    payout = shares if did_win else Decimal("0")
    return (payout - stake_usd).quantize(QUANT_6, rounding=ROUND_HALF_UP)


def did_trade_win(side: str, bucket_key: str, settlement_bucket_key: str) -> bool:
    normalized_side = normalize_side(side)
    bucket = validate_bucket_key(bucket_key)
    settlement = validate_bucket_key(settlement_bucket_key)
    bucket_won = bucket == settlement
    return bucket_won if normalized_side == "yes" else not bucket_won


def hkg_event_slug_for_date(target_date: date) -> str:
    month = MONTH_SLUGS[target_date.month]
    return f"highest-temperature-in-hong-kong-on-{month}-{target_date.day}-{target_date.year}"


def hkg_event_url_for_date(target_date: date) -> str:
    return f"https://polymarket.com/event/{hkg_event_slug_for_date(target_date)}"


def hkg_event_title_for_date(target_date: date) -> str:
    month = MONTH_SLUGS[target_date.month].title()
    return f"Highest temperature in Hong Kong on {month} {target_date.day}?"


def h24n_cutoff_utc(target_date: date) -> datetime:
    cutoff_hkt = datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        15,
        0,
        tzinfo=HKT,
    ) - timedelta(days=1)
    return cutoff_hkt.astimezone(UTC)


def today_hkt() -> date:
    return datetime.now(HKT).date()


def date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise DemoTradingError("end date must be on or after start date")
    if (end - start).days > 45:
        raise DemoTradingError("date range may not exceed 45 days")
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def edge_classification(edge: Decimal | float | None) -> str:
    if edge is None:
        return "unavailable"
    value = Decimal(str(edge))
    if value <= 0:
        return "bad"
    if value < 5:
        return "normal"
    if value < 8:
        return "good"
    if value < 12:
        return "very good"
    return "ELITE"


def decimal_to_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
