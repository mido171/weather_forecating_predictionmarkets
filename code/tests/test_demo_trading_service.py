from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import hkg_tmax_demo_trading.service as service_module
import pytest
from hkg_tmax_demo_trading.domain import DemoTradingError
from hkg_tmax_demo_trading.market_data import MarketBucket
from hkg_tmax_demo_trading.service import DemoTradingService, ResourceUnavailable


class FakeConnection:
    def commit(self) -> None:
        return None


class FakeStore:
    def __init__(self) -> None:
        self.inserted: dict[str, Any] | None = None
        self.fallback_snapshot: dict[str, Any] | None = None

    @contextmanager
    def connect(self) -> Any:
        yield FakeConnection()

    def get_or_create_active_account(self, _connection: Any) -> dict[str, Any]:
        return {
            "id": 1,
            "starting_balance_usd": Decimal("1000"),
            "started_at_utc": datetime(2026, 7, 6, tzinfo=UTC),
        }

    def settle_open_trades(self, _connection: Any) -> list[dict[str, Any]]:
        return []

    def list_trades(self, _connection: Any, _account_id: int) -> list[dict[str, Any]]:
        return []

    def insert_trade(self, _connection: Any, **kwargs: Any) -> dict[str, Any]:
        self.inserted = kwargs
        return {
            "id": 10,
            "bucket_key": kwargs["bucket_key_value"],
            "side": kwargs["side"],
            "entry_price_cents": kwargs["entry_price_cents"],
            "price_source": kwargs["price_source"],
            "status": "open",
        }

    def trade_by_id(self, _connection: Any, _trade_id: int) -> dict[str, Any] | None:
        return None

    def open_trade_for_contract(
        self,
        _connection: Any,
        *,
        account_id: int,
        target_date: date,
        bucket_key_value: str,
        side: str,
    ) -> dict[str, Any] | None:
        return None

    def latest_successful_snapshot(
        self,
        _connection: Any,
        _target_date: date,
        *,
        as_of_profile: str | None = None,
    ) -> dict[str, Any] | None:
        return self.fallback_snapshot


class MarketOnlyService(DemoTradingService):
    def refresh_market(self, target_date: date, as_of_profile: str | None = None) -> dict[str, Any]:
        return {
            "snapshotId": 55,
            "createdAtUtc": "2026-07-06T10:00:00+00:00",
            "as_of_profile": as_of_profile or "t_minus_1_2359_hkt",
            "status": "ok",
            "event": {"slug": "highest-temperature-in-hong-kong-on-july-7-2026"},
            "forecast": {"forecastMaxC": 31.0, "source": "fake"},
            "model": {
                "method": "B4_hierarchical_residual_pmf",
                "cutoff_profile": as_of_profile or "t_minus_1_2359_hkt",
                "train_rows": 100,
            },
            "profile": {
                "id": as_of_profile or "t_minus_1_2359_hkt",
                "label": "17:59 Stockholm",
                "stockholmEntry": "17:59",
                "hktCutoff": "23:59",
                "validationStatus": "validated_apples_to_apples",
                "tradeable": True,
                "forecastAnchorAvailable": True,
            },
            "marketRows": [
                {
                    "bucket": "28",
                    "label": "28°C",
                    "modelProbability": 0.75,
                    "marketBuyYesC": 55.0,
                    "marketBuyYesSource": "clob_ask",
                    "marketBuyNoC": 70.0,
                    "marketBuyNoSource": "clob_ask",
                }
            ],
        }

    def get_account(self, since: date | None) -> dict[str, Any]:
        return {"cashUsd": 975.0, "openTrades": []}


class CachedUnavailableSnapshotStore:
    def __init__(self) -> None:
        self.requested_profile: str | None = None

    @contextmanager
    def connect(self) -> Any:
        yield FakeConnection()

    def latest_snapshot(
        self,
        _connection: Any,
        target_date: date,
        *,
        as_of_profile: str | None = None,
    ) -> dict[str, Any]:
        self.requested_profile = as_of_profile
        return {
            "id": 12,
            "target_date": target_date,
            "created_at_utc": datetime(2026, 7, 7, 9, 0, tzinfo=UTC),
            "status": "unavailable",
            "status_reason": "stale forecast miss",
            "as_of_profile": as_of_profile,
            "snapshot_json": {
                "snapshotId": 12,
                "targetDate": target_date.isoformat(),
                "status": "unavailable",
                "statusReason": "stale forecast miss",
            },
        }


class CachedWrongSourceSnapshotStore:
    def __init__(self) -> None:
        self.requested_profile: str | None = None

    @contextmanager
    def connect(self) -> Any:
        yield FakeConnection()

    def latest_snapshot(
        self,
        _connection: Any,
        target_date: date,
        *,
        as_of_profile: str | None = None,
    ) -> dict[str, Any]:
        self.requested_profile = as_of_profile
        return {
            "id": 13,
            "target_date": target_date,
            "created_at_utc": datetime(2026, 7, 7, 9, 0, tzinfo=UTC),
            "status": "ok",
            "status_reason": None,
            "as_of_profile": as_of_profile,
            "snapshot_json": {
                "snapshotId": 13,
                "targetDate": target_date.isoformat(),
                "as_of_profile": as_of_profile,
                "status": "ok",
                "forecast": {
                    "source": "HKO fnd 9-day forecast live cutoff fetch",
                    "raw": {"live_cutoff_fetch": True},
                },
            },
        }


class RefreshingGetMarketService(DemoTradingService):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.refreshed_profile: str | None = None

    def refresh_market(self, target_date: date, as_of_profile: str | None = None) -> dict[str, Any]:
        self.refreshed_profile = as_of_profile
        return {
            "status": "ok",
            "targetDate": target_date.isoformat(),
            "as_of_profile": as_of_profile,
        }


def test_get_market_refreshes_cached_unavailable_snapshot(repo_root: Path) -> None:
    store = CachedUnavailableSnapshotStore()
    service = RefreshingGetMarketService(store=store, repo_root=repo_root, database_url="postgresql://example")

    result = service.get_market(date(2026, 7, 8), as_of_profile="t_minus_1_1900_hkt")

    assert result["status"] == "ok"
    assert store.requested_profile == "t_minus_1_1900_hkt"
    assert service.refreshed_profile == "t_minus_1_1900_hkt"


def test_get_market_refreshes_cached_validated_snapshot_from_wrong_live_source(repo_root: Path) -> None:
    store = CachedWrongSourceSnapshotStore()
    service = RefreshingGetMarketService(store=store, repo_root=repo_root, database_url="postgresql://example")

    result = service.get_market(date(2026, 7, 8), as_of_profile="t_minus_1_1800_hkt")

    assert result["status"] == "ok"
    assert store.requested_profile == "t_minus_1_1800_hkt"
    assert service.refreshed_profile == "t_minus_1_1800_hkt"


def test_create_trade_uses_market_quote_without_manual_override(repo_root: Path) -> None:
    store = FakeStore()
    service = MarketOnlyService(store=store, repo_root=repo_root, database_url="postgresql://example")

    result = service.create_trade(
        target_date=date(2026, 7, 7),
        bucket_key="28",
        side="yes",
        stake_usd=25,
    )

    assert result["trade"]["price_source"] == "clob_ask"
    assert store.inserted is not None
    assert store.inserted["entry_price_cents"] == Decimal("55.0000")
    assert store.inserted["price_source"] == "clob_ask"
    entry = store.inserted["metadata"]["entry"]
    assert entry["manual_price_cents"] is None
    assert entry["price_source"] == "clob_ask"
    assert entry["as_of_profile"] == "t_minus_1_2359_hkt"
    assert entry["profile"]["tradeable"] is True


def test_create_trade_explicit_manual_price_wins_for_replay(repo_root: Path) -> None:
    store = FakeStore()
    service = MarketOnlyService(store=store, repo_root=repo_root, database_url="postgresql://example")

    result = service.create_trade(
        target_date=date(2026, 7, 7),
        bucket_key="28",
        side="yes",
        stake_usd=25,
        manual_price_cents=58,
    )

    assert result["trade"]["price_source"] == "manual_override"
    assert store.inserted is not None
    assert store.inserted["entry_price_cents"] == Decimal("58.0000")
    assert store.inserted["price_source"] == "manual_override"
    entry = store.inserted["metadata"]["entry"]
    assert entry["manual_price_cents"] == 58
    assert entry["price_source"] == "manual_override"


class UnavailableMarketService(DemoTradingService):
    def refresh_market(self, target_date: date, as_of_profile: str | None = None) -> dict[str, Any]:
        return {
            "snapshotId": 99,
            "status": "unavailable",
            "statusReason": "No live forecast candidate found",
            "event": {"slug": "highest-temperature-in-hong-kong-on-july-6-2026"},
            "marketRows": [],
        }

    def get_account(self, since: date | None) -> dict[str, Any]:
        return {"cashUsd": 950.0, "openTrades": []}


def test_create_manual_trade_can_use_prior_successful_snapshot_after_live_rolloff(repo_root: Path) -> None:
    store = FakeStore()
    store.fallback_snapshot = {
        "id": 44,
        "target_date": date(2026, 7, 6),
        "created_at_utc": datetime(2026, 7, 6, 1, 5, tzinfo=UTC),
        "snapshot_json": {
            "snapshotId": 44,
            "targetDate": "2026-07-06",
            "status": "ok",
            "event": {"slug": "highest-temperature-in-hong-kong-on-july-6-2026"},
            "marketRows": [
                {
                    "bucket": "32",
                    "label": "32C",
                    "modelProbability": 0.22826367229339364,
                    "marketBuyNoC": 99.0,
                    "marketBuyNoSource": "clob_ask",
                }
            ],
        },
    }
    service = UnavailableMarketService(store=store, repo_root=repo_root, database_url="postgresql://example")

    result = service.create_trade(
        target_date=date(2026, 7, 6),
        bucket_key="32",
        side="no",
        stake_usd=50,
        manual_price_cents=58,
    )

    assert result["trade"]["bucket_key"] == "32"
    assert result["trade"]["side"] == "no"
    assert store.inserted is not None
    assert store.inserted["snapshot_id"] == 44
    assert store.inserted["entry_price_cents"] == Decimal("58.0000")
    assert store.inserted["model_probability_bucket"] == Decimal("0.2282636723")
    assert store.inserted["model_win_probability"] == Decimal("0.7717363277")
    assert store.inserted["edge_pp"] == Decimal("19.1736")


class SnapshotFallbackStore:
    def __init__(self) -> None:
        self.inserted: dict[str, Any] | None = None

    @contextmanager
    def connect(self) -> Any:
        yield FakeConnection()

    def latest_successful_snapshot(
        self,
        _connection: Any,
        target_date: date,
        *,
        as_of_profile: str | None = None,
    ) -> dict[str, Any]:
        return {
            "id": 14,
            "target_date": target_date,
            "created_at_utc": datetime(2026, 7, 6, 1, 5, tzinfo=UTC),
            "snapshot_json": {
                "as_of_profile": "live_now_artifact",
                "forecast": {
                    "source": "HKO fnd 9-day forecast",
                    "update_time_hkt": "2026-07-06T01:05:00+08:00",
                    "target_date": "2026-07-06",
                    "forecast_min_c": 27.0,
                    "forecast_max_c": 31.0,
                    "raw": {},
                },
                "model": {
                    "method": "B4_hierarchical_residual_pmf",
                    "bucket_keys": [
                        "24_or_below",
                        "25",
                        "26",
                        "27",
                        "28",
                        "29",
                        "30",
                        "31",
                        "32",
                        "33",
                        "34_or_higher",
                    ],
                    "probabilities": {
                        "24_or_below": 0.0,
                        "25": 0.0,
                        "26": 0.0,
                        "27": 0.0,
                        "28": 0.0,
                        "29": 0.0,
                        "30": 0.25,
                        "31": 0.25,
                        "32": 0.25,
                        "33": 0.25,
                        "34_or_higher": 0.0,
                    },
                    "train_rows": 9644,
                    "train_start": "2000-01-02",
                    "train_end": "2026-05-31",
                },
            },
        }

    def insert_snapshot(self, _connection: Any, snapshot: dict[str, Any]) -> dict[str, Any]:
        self.inserted = snapshot
        return {
            "id": 77,
            "target_date": snapshot["target_date"],
            "created_at_utc": datetime(2026, 7, 6, 10, 0, tzinfo=UTC),
            "snapshot_json": snapshot,
        }


class WrongSourceSnapshotFallbackStore(SnapshotFallbackStore):
    def latest_successful_snapshot(
        self,
        _connection: Any,
        target_date: date,
        *,
        as_of_profile: str | None = None,
    ) -> dict[str, Any]:
        return {
            "id": 15,
            "target_date": target_date,
            "as_of_profile": as_of_profile,
            "created_at_utc": datetime(2026, 7, 7, 8, 35, tzinfo=UTC),
            "snapshot_json": {
                "as_of_profile": as_of_profile,
                "forecast": {
                    "source": "HKO fnd 9-day forecast live cutoff fetch",
                    "raw": {"live_cutoff_fetch": True},
                },
                "model": {
                    "method": "B4_hierarchical_residual_pmf",
                    "probabilities": {"31": 0.2},
                    "bucket_keys": ["31"],
                },
            },
        }


def test_refresh_market_reuses_successful_probability_snapshot_after_forecast_rolloff(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
) -> None:
    store = SnapshotFallbackStore()
    service = DemoTradingService(store=store, repo_root=repo_root, database_url="postgresql://example")
    monkeypatch.setattr(service_module, "fetch_event_for_date", lambda _target_date: {})
    monkeypatch.setattr(
        service_module,
        "event_view_from_gamma",
        lambda target_date, _payload: {
            "slug": f"highest-temperature-in-hong-kong-on-july-{target_date.day}-2026",
            "url": "https://polymarket.com/event/highest-temperature-in-hong-kong-on-july-6-2026",
            "title": "Highest temperature in Hong Kong on July 6?",
        },
    )
    monkeypatch.setattr(service_module, "parse_market_buckets", lambda _payload: {})

    def unavailable_probability(**_: Any) -> Any:
        raise service_module.ForecastUnavailable("No HKO flw/fnd forecast candidate found")

    monkeypatch.setattr(service_module, "build_probability_snapshot", unavailable_probability)

    view = service.refresh_market(date(2026, 7, 6))

    assert view["status"] == "ok"
    assert "Reused latest successful probability snapshot" in view["status_reason"]
    assert store.inserted is not None
    assert store.inserted["status"] == "ok"
    row32 = next(row for row in view["marketRows"] if row["bucket"] == "32")
    assert row32["modelProbability"] == 0.25
    assert row32["modelYesPct"] == 25.0


def test_refresh_market_rejects_wrong_source_successful_probability_snapshot_after_forecast_rolloff(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
) -> None:
    store = WrongSourceSnapshotFallbackStore()
    service = DemoTradingService(store=store, repo_root=repo_root, database_url="postgresql://example")
    monkeypatch.setattr(service_module, "fetch_event_for_date", lambda _target_date: {})
    monkeypatch.setattr(
        service_module,
        "event_view_from_gamma",
        lambda target_date, _payload: {
            "slug": f"highest-temperature-in-hong-kong-on-july-{target_date.day}-2026",
            "url": "https://polymarket.com/event/highest-temperature-in-hong-kong-on-july-8-2026",
            "title": "Highest temperature in Hong Kong on July 8?",
        },
    )
    monkeypatch.setattr(service_module, "parse_market_buckets", lambda _payload: {})

    def unavailable_probability(**_: Any) -> Any:
        raise service_module.ForecastUnavailable("No live HKO local forecast update is eligible")

    monkeypatch.setattr(service_module, "build_probability_snapshot", unavailable_probability)

    view = service.refresh_market(date(2026, 7, 8), as_of_profile="t_minus_1_1800_hkt")

    assert view["status"] == "unavailable"
    assert view["forecast"] is None
    assert "Reused latest successful probability snapshot" not in view["status_reason"]
    assert store.inserted is not None
    assert store.inserted["status"] == "unavailable"


def test_build_market_rows_fetches_clob_quotes_without_losing_executable_sources(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
) -> None:
    calls: list[str] = []

    def fake_clob_buy_price(token_id: str | None) -> tuple[Decimal | None, str]:
        assert token_id is not None
        calls.append(token_id)
        return (Decimal("0.42"), "clob_ask") if token_id == "yes-token" else (Decimal("0.61"), "clob_ask")

    monkeypatch.setattr(service_module, "clob_buy_price", fake_clob_buy_price)
    service = DemoTradingService(store=FakeStore(), repo_root=repo_root, database_url="postgresql://example")
    rows = service._build_market_rows(
        {
            "32": MarketBucket(
                bucket_key="32",
                label="32",
                market_id="m1",
                market_slug="m1-slug",
                question="32?",
                active=True,
                closed=False,
                accepting_orders=True,
                volume=100.0,
                liquidity=50.0,
                yes_token="yes-token",
                no_token="no-token",
                yes_fallback_price=None,
                no_fallback_price=None,
                market_probability=None,
            )
        },
        {"probabilities": {"32": 0.70}},
    )

    row32 = next(row for row in rows if row["bucket"] == "32")
    assert sorted(calls) == ["no-token", "yes-token"]
    assert row32["marketBuyYesC"] == 42.0
    assert row32["marketBuyYesSource"] == "clob_ask"
    assert row32["marketBuyYesExecutable"] is True
    assert row32["marketBuyNoC"] == 61.0
    assert row32["marketBuyNoSource"] == "clob_ask"
    assert row32["marketBuyNoExecutable"] is True


class ManualSettleStore:
    def __init__(self, *, current_price_cents: Decimal | None = Decimal("98.2000")) -> None:
        self.current_price_cents = current_price_cents
        self.recorded_contract_price: Decimal | None = None
        self.recorded_loss_contract_price: Decimal | None = None
        self.recorded_loss_fraction: Decimal | None = None
        self.settled_trade: dict[str, Any] | None = None
        self.trade = {
            "id": 10,
            "account_session_id": 1,
            "snapshot_id": 55,
            "target_date": date(2026, 7, 6),
            "event_slug": "highest-temperature-in-hong-kong-on-july-6-2026",
            "bucket_key": "32",
            "side": "no",
            "stake_usd": Decimal("50.000000"),
            "shares": Decimal("86.20689655"),
            "entry_price_cents": Decimal("58.0000"),
            "price_source": "manual_override",
            "model_probability_bucket": Decimal("0.2282636723"),
            "model_win_probability": Decimal("0.7717363277"),
            "edge_pp": Decimal("19.1736"),
            "ev_usd": Decimal("16.528994"),
            "status": "open",
            "result": None,
            "settlement_tmax_c": None,
            "settlement_bucket_key": None,
            "settled_at_utc": None,
            "realized_pnl_usd": Decimal("0"),
            "marked_value_usd": Decimal("0"),
            "unrealized_pnl_usd": Decimal("0"),
            "opened_at_utc": datetime(2026, 7, 6, 9, 0, tzinfo=UTC),
            "metadata_json": {"no_real_order": True},
        }

    @contextmanager
    def connect(self) -> Any:
        yield FakeConnection()

    def get_or_create_active_account(self, _connection: Any) -> dict[str, Any]:
        return {
            "id": 1,
            "starting_balance_usd": Decimal("1000"),
            "started_at_utc": datetime(2026, 7, 6, tzinfo=UTC),
        }

    def settle_open_trades(self, _connection: Any) -> list[dict[str, Any]]:
        return []

    def trade_by_id(self, _connection: Any, trade_id: int) -> dict[str, Any] | None:
        if trade_id != self.trade["id"]:
            return None
        return self.settled_trade or self.trade

    def latest_snapshot(self, _connection: Any, target_date: date) -> dict[str, Any] | None:
        if self.current_price_cents is None:
            return None
        return {
            "id": 77,
            "target_date": target_date,
            "created_at_utc": datetime(2026, 7, 6, 10, 0, tzinfo=UTC),
            "snapshot_json": {
                "marketRows": [
                    {
                        "bucket": "32",
                        "marketBuyNoC": self.current_price_cents,
                    }
                ]
            },
        }

    def settle_trade_as_win(
        self,
        _connection: Any,
        *,
        trade: dict[str, Any],
        contract_price_cents: Decimal,
    ) -> dict[str, Any]:
        self.recorded_contract_price = contract_price_cents
        self.settled_trade = {
            **trade,
            "status": "settled",
            "result": "win",
            "settled_at_utc": datetime(2026, 7, 6, 10, 0, tzinfo=UTC),
            "realized_pnl_usd": Decimal("36.206897"),
            "marked_value_usd": Decimal("0"),
            "unrealized_pnl_usd": Decimal("0"),
            "metadata_json": {
                **trade["metadata_json"],
                "settlement": {
                    "result": "win",
                    "did_win": True,
                    "source": "manual_contract_price_threshold",
                    "contract_price_cents": contract_price_cents,
                },
            },
        }
        return self.settled_trade

    def settle_trade_as_loss(
        self,
        _connection: Any,
        *,
        trade: dict[str, Any],
        contract_price_cents: Decimal,
        loss_fraction: Decimal,
    ) -> dict[str, Any]:
        self.recorded_loss_contract_price = contract_price_cents
        self.recorded_loss_fraction = loss_fraction
        self.settled_trade = {
            **trade,
            "status": "settled",
            "result": "loss",
            "settled_at_utc": datetime(2026, 7, 6, 10, 0, tzinfo=UTC),
            "realized_pnl_usd": Decimal("-50.000000"),
            "marked_value_usd": Decimal("0"),
            "unrealized_pnl_usd": Decimal("0"),
            "metadata_json": {
                **trade["metadata_json"],
                "settlement": {
                    "result": "loss",
                    "did_win": False,
                    "source": "manual_position_loss_threshold",
                    "contract_price_cents": contract_price_cents,
                    "loss_fraction_of_max_loss": loss_fraction,
                },
            },
        }
        return self.settled_trade

    def list_trades(self, _connection: Any, _account_id: int) -> list[dict[str, Any]]:
        return [self.settled_trade or self.trade]

    def update_open_mark(
        self,
        _connection: Any,
        _trade_id: int,
        _marked_value_usd: Decimal,
        _unrealized_pnl_usd: Decimal,
    ) -> None:
        return None


def test_settle_trade_as_win_books_result_when_contract_price_is_at_least_98(repo_root: Path) -> None:
    store = ManualSettleStore(current_price_cents=Decimal("98.2000"))
    service = DemoTradingService(store=store, repo_root=repo_root, database_url="postgresql://example")

    result = service.settle_trade_as_win(10)

    assert store.recorded_contract_price == Decimal("98.2000")
    assert result["trade"]["status"] == "settled"
    assert result["trade"]["result"] == "win"
    assert result["trade"]["realized_pnl_usd"] == 36.206897
    assert result["trade"]["metadata_json"]["settlement"]["result"] == "win"
    assert result["account"]["realizedPnlUsd"] == 36.206897


def test_settle_trade_as_win_rejects_contract_price_below_98(repo_root: Path) -> None:
    store = ManualSettleStore(current_price_cents=Decimal("97.9900"))
    service = DemoTradingService(store=store, repo_root=repo_root, database_url="postgresql://example")

    with pytest.raises(DemoTradingError, match="at least 98c"):
        service.settle_trade_as_win(10)

    assert store.recorded_contract_price is None


def test_settle_trade_as_win_requires_current_contract_price(repo_root: Path) -> None:
    store = ManualSettleStore(current_price_cents=None)
    service = DemoTradingService(store=store, repo_root=repo_root, database_url="postgresql://example")

    with pytest.raises(ResourceUnavailable, match="Current contract price is unavailable"):
        service.settle_trade_as_win(10)


def test_settle_trade_as_loss_books_result_when_position_has_lost_at_least_97_percent(
    repo_root: Path,
) -> None:
    store = ManualSettleStore(current_price_cents=Decimal("1.7400"))
    service = DemoTradingService(store=store, repo_root=repo_root, database_url="postgresql://example")

    result = service.settle_trade_as_loss(10)

    assert store.recorded_loss_contract_price == Decimal("1.7400")
    assert store.recorded_loss_fraction == Decimal("0.970000")
    assert result["trade"]["status"] == "settled"
    assert result["trade"]["result"] == "loss"
    assert result["trade"]["realized_pnl_usd"] == -50.0
    assert result["trade"]["metadata_json"]["settlement"]["result"] == "loss"
    assert result["account"]["realizedPnlUsd"] == -50.0


def test_settle_trade_as_loss_rejects_position_below_97_percent_loss(repo_root: Path) -> None:
    store = ManualSettleStore(current_price_cents=Decimal("1.7500"))
    service = DemoTradingService(store=store, repo_root=repo_root, database_url="postgresql://example")

    with pytest.raises(DemoTradingError, match="at least 97%"):
        service.settle_trade_as_loss(10)

    assert store.recorded_loss_contract_price is None
    assert store.recorded_loss_fraction is None


def test_settle_trade_as_loss_requires_current_contract_price(repo_root: Path) -> None:
    store = ManualSettleStore(current_price_cents=None)
    service = DemoTradingService(store=store, repo_root=repo_root, database_url="postgresql://example")

    with pytest.raises(ResourceUnavailable, match="Current contract price is unavailable"):
        service.settle_trade_as_loss(10)


def test_open_account_trade_exposes_manual_settle_eligibility(repo_root: Path) -> None:
    store = ManualSettleStore(current_price_cents=Decimal("98.0000"))
    service = DemoTradingService(store=store, repo_root=repo_root, database_url="postgresql://example")

    account = service.get_account(None)

    open_trade = account["openTrades"][0]
    assert open_trade["current_price_cents"] == 98.0
    assert open_trade["manual_settle_eligible"] is True
    assert open_trade["manual_settle_result"] == "win"
    assert open_trade["manual_win_settle_eligible"] is True
    assert open_trade["manual_loss_settle_eligible"] is False
    assert open_trade["manual_settle_threshold_cents"] == 98.0
    assert open_trade["manual_win_settle_threshold_cents"] == 98.0
    assert open_trade["manual_loss_settle_threshold_fraction"] == 0.97


def test_open_account_trade_exposes_manual_loss_settle_eligibility(repo_root: Path) -> None:
    store = ManualSettleStore(current_price_cents=Decimal("0.2000"))
    service = DemoTradingService(store=store, repo_root=repo_root, database_url="postgresql://example")

    account = service.get_account(None)

    open_trade = account["openTrades"][0]
    assert open_trade["current_price_cents"] == 0.2
    assert open_trade["manual_settle_eligible"] is True
    assert open_trade["manual_settle_result"] == "loss"
    assert open_trade["manual_win_settle_eligible"] is False
    assert open_trade["manual_loss_settle_eligible"] is True
    assert open_trade["manual_loss_fraction_of_max_loss"] > 0.97
