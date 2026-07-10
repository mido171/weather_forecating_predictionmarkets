from __future__ import annotations

from decimal import Decimal

import pytest
from hkg_tmax_demo_trading.domain import (
    DemoTradingError,
    compute_trade,
    did_trade_win,
    h24n_cutoff_utc,
    realized_pnl_usd,
    validate_price_cents,
)
from hkg_tmax_probability.bucket_rules import bucket_key


def test_hko_one_decimal_bucket_boundaries() -> None:
    assert bucket_key("24.9") == "24_or_below"
    assert bucket_key("25.0") == "25"
    assert bucket_key("31.9") == "31"
    assert bucket_key("32.0") == "32"
    assert bucket_key("33.9") == "33"
    assert bucket_key("34.0") == "34_or_higher"


def test_yes_trade_ev_and_edge_math() -> None:
    trade = compute_trade(
        stake_usd="20",
        entry_price_cents="40",
        model_probability_bucket="0.55",
        side="yes",
    )

    assert trade.shares == Decimal("50.00000000")
    assert trade.model_win_probability == Decimal("0.5500000000")
    assert trade.edge_pp == Decimal("15.0000")
    assert trade.ev_usd == Decimal("7.500000")


def test_no_trade_uses_complement_probability() -> None:
    trade = compute_trade(
        stake_usd="10",
        entry_price_cents="25",
        model_probability_bucket="0.30",
        side="no",
    )

    assert trade.shares == Decimal("40.00000000")
    assert trade.model_win_probability == Decimal("0.7000000000")
    assert trade.edge_pp == Decimal("45.0000")
    assert trade.ev_usd == Decimal("18.000000")


def test_settlement_pnl_for_yes_and_no() -> None:
    assert did_trade_win("yes", "32", "32") is True
    assert did_trade_win("yes", "32", "31") is False
    assert did_trade_win("no", "32", "31") is True
    assert did_trade_win("no", "32", "32") is False

    assert realized_pnl_usd(stake_usd=Decimal("20"), shares=Decimal("50"), did_win=True) == Decimal("30.000000")
    assert realized_pnl_usd(stake_usd=Decimal("20"), shares=Decimal("50"), did_win=False) == Decimal("-20.000000")


@pytest.mark.parametrize("price", ["0", "-1", "100.01"])
def test_entry_price_validation_rejects_invalid_manual_prices(price: str) -> None:
    with pytest.raises(DemoTradingError):
        validate_price_cents(price)


def test_h24n_cutoff_is_1500_hkt_previous_day() -> None:
    cutoff = h24n_cutoff_utc(__import__("datetime").date(2026, 7, 6))
    assert cutoff.isoformat() == "2026-07-05T07:00:00+00:00"
