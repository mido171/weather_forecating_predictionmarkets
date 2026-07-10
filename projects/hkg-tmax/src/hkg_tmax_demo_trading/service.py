"""Application service for the HKG Polymarket demo backtester."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from hkg_tmax_probability.bucket_rules import BUCKET_KEYS

from .domain import (
    DemoTradingError,
    compute_trade,
    date_range,
    decimal_to_float,
    edge_classification,
    hkg_event_slug_for_date,
    hkg_event_title_for_date,
    hkg_event_url_for_date,
    normalize_side,
    quantize,
    today_hkt,
    validate_bucket_key,
    validate_price_cents,
    validate_stake_usd,
)
from .market_data import (
    MarketDataUnavailable,
    clob_buy_price,
    event_view_from_gamma,
    fetch_event_for_date,
    parse_market_buckets,
)
from .probability import (
    LOCAL_FORECAST_SOURCE,
    ForecastUnavailable,
    build_probability_snapshot,
    normalize_as_of_profile,
    profile_metadata,
)
from .store import PgDemoTradingStore, json_safe, row_to_json


class ResourceUnavailable(DemoTradingError):
    """Expected missing-source error for market/trade creation."""


MANUAL_WIN_SETTLE_THRESHOLD_CENTS = Decimal("98.0000")
MANUAL_LOSS_SETTLE_THRESHOLD_FRACTION = Decimal("0.970000")
MIN_STRATEGY_EDGE_PP = Decimal("15.0000")
MIN_STRATEGY_WIN_PROBABILITY = Decimal("0.7000")
CLOB_PRICE_WORKERS = 8


def _cents_from_probability_price(value: Decimal | None) -> float | None:
    if value is None:
        return None
    return float((value * Decimal("100")).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def _as_decimal_or_none(value: Any) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def _marked_value_at_price(*, shares: Any, contract_price_cents: Decimal) -> Decimal:
    return (Decimal(str(shares)) * contract_price_cents / Decimal("100")).quantize(
        Decimal("0.000001"), rounding=ROUND_HALF_UP
    )


def _loss_fraction_of_max_loss(*, stake_usd: Any, marked_value_usd: Any) -> Decimal:
    stake = Decimal(str(stake_usd))
    if stake <= 0:
        return Decimal("0.000000")
    loss = stake - Decimal(str(marked_value_usd))
    if loss <= 0:
        return Decimal("0.000000")
    fraction = loss / stake
    if fraction > 1:
        fraction = Decimal("1")
    return fraction.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


def _forecast_payload(value: Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    return {
        "source": value.source,
        "update_time_hkt": value.update_time_hkt,
        "target_date": value.target_date,
        "forecast_min_c": value.forecast_min_c,
        "forecast_max_c": value.forecast_max_c,
        "as_of_profile": value.as_of_profile,
        "profile": profile_metadata(value.as_of_profile),
        "raw": value.raw,
    }


def _display_profile_id(as_of_profile: str | None) -> str:
    try:
        return normalize_as_of_profile(as_of_profile) or "live_now"
    except ForecastUnavailable:
        return str(as_of_profile or "unknown")


def _uses_invalid_validated_live_source(payload: dict[str, Any], as_of_profile: str | None = None) -> bool:
    profile = _display_profile_id(payload.get("as_of_profile") or as_of_profile)
    if profile == "live_now":
        return False
    forecast = payload.get("forecast")
    if not isinstance(forecast, dict):
        return True
    raw = forecast.get("raw")
    return (
        isinstance(raw, dict)
        and raw.get("live_cutoff_fetch") is True
        and forecast.get("source") != f"{LOCAL_FORECAST_SOURCE} live cutoff fetch"
    )


def _snapshot_needs_refresh(snapshot: dict[str, Any]) -> bool:
    if snapshot.get("status") != "ok":
        return True
    payload = snapshot.get("snapshot_json")
    if not isinstance(payload, dict):
        return True
    return _uses_invalid_validated_live_source(payload, snapshot.get("as_of_profile"))


def _fallback_snapshot_payload(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    payload = snapshot.get("snapshot_json")
    if not isinstance(payload, dict):
        return None
    if _uses_invalid_validated_live_source(payload, snapshot.get("as_of_profile")):
        return None
    return payload


class DemoTradingService:
    def __init__(self, *, store: PgDemoTradingStore, repo_root: Path, database_url: str) -> None:
        self.store = store
        self.repo_root = repo_root
        self.database_url = database_url

    def apply_schema(self) -> None:
        self.store.apply_schema()

    def list_markets(self, start: date, end: date) -> dict[str, Any]:
        days = date_range(start, end)
        with self.store.connect() as connection:
            account = self.store.get_or_create_active_account(connection)
            latest = self.store.latest_snapshots_for_range(connection, start, end)
            trades = self.store.list_trades(connection, account["id"])
            open_counts: dict[date, int] = {}
            for trade in trades:
                if trade["status"] == "open":
                    open_counts[trade["target_date"]] = open_counts.get(trade["target_date"], 0) + 1
            items = []
            for target_date in days:
                settlement = self.store.find_settlement(connection, target_date)
                snapshot = latest.get(target_date)
                items.append(
                    {
                        "targetDate": target_date.isoformat(),
                        "slug": hkg_event_slug_for_date(target_date),
                        "title": hkg_event_title_for_date(target_date),
                        "url": hkg_event_url_for_date(target_date),
                        "isPast": target_date < today_hkt(),
                        "isToday": target_date == today_hkt(),
                        "openTradeCount": open_counts.get(target_date, 0),
                        "settlement": None
                        if settlement is None
                        else {
                            "targetTmaxC": decimal_to_float(settlement["target_tmax_c"]),
                            "bucketKey": settlement["settlement_bucket_key"],
                            "source": settlement["target_source_id"],
                        },
                        "snapshot": None
                        if snapshot is None
                        else {
                            "id": snapshot["id"],
                            "status": snapshot["status"],
                            "statusReason": snapshot["status_reason"],
                            "createdAtUtc": snapshot["created_at_utc"].isoformat(),
                            "asOfProfile": snapshot["as_of_profile"],
                        },
                    }
                )
            connection.commit()
        return {"start": start.isoformat(), "end": end.isoformat(), "markets": items}

    def get_market(self, target_date: date, as_of_profile: str | None = None) -> dict[str, Any]:
        normalized_profile = normalize_as_of_profile(as_of_profile) if as_of_profile else None
        with self.store.connect() as connection:
            snapshot = self.store.latest_snapshot(connection, target_date, as_of_profile=normalized_profile)
            connection.commit()
        if snapshot is None or _snapshot_needs_refresh(snapshot):
            return self.refresh_market(target_date, normalized_profile)
        return self._snapshot_view(snapshot)

    def refresh_market(self, target_date: date, as_of_profile: str | None = None) -> dict[str, Any]:
        event_error = None
        event_payload: dict[str, Any] | None = None
        try:
            event_payload = fetch_event_for_date(target_date)
        except MarketDataUnavailable as exc:
            event_error = str(exc)
        event = event_view_from_gamma(target_date, event_payload)
        markets = parse_market_buckets(event_payload) if event_payload else {}

        probability = None
        probability_error = None
        with self.store.connect() as connection:
            try:
                probability = build_probability_snapshot(
                    connection=connection,
                    repo_root=self.repo_root,
                    database_url=self.database_url,
                    target_date=target_date,
                    as_of_profile=as_of_profile,
                )
            except ForecastUnavailable as exc:
                probability_error = str(exc)
            except Exception as exc:  # noqa: BLE001 - surface as snapshot status.
                probability_error = f"{type(exc).__name__}: {exc}"

            fallback_snapshot_payload = None
            normalized_profile = None
            try:
                normalized_profile = normalize_as_of_profile(as_of_profile) if as_of_profile else None
            except ForecastUnavailable:
                normalized_profile = as_of_profile
            if probability is None:
                fallback_snapshot = self.store.latest_successful_snapshot(
                    connection,
                    target_date,
                    as_of_profile=normalized_profile,
                )
                fallback_snapshot_payload = _fallback_snapshot_payload(fallback_snapshot)

            snapshot = self._build_snapshot_payload(
                target_date=target_date,
                event=event,
                markets=markets,
                probability=probability,
                event_error=event_error,
                probability_error=probability_error,
                as_of_profile=as_of_profile,
                fallback_snapshot_payload=fallback_snapshot_payload,
            )
            row = self.store.insert_snapshot(connection, snapshot)
            connection.commit()
        return self._snapshot_view(row)

    def create_trade(
        self,
        *,
        target_date: date,
        bucket_key: str,
        side: str,
        stake_usd: Any,
        manual_price_cents: Any | None = None,
        as_of_profile: str | None = None,
    ) -> dict[str, Any]:
        bucket = validate_bucket_key(bucket_key)
        normalized_side = normalize_side(side)
        stake = validate_stake_usd(stake_usd)
        normalized_profile = normalize_as_of_profile(as_of_profile) if as_of_profile else None

        with self.store.connect() as connection:
            account = self.store.get_or_create_active_account(connection)
            self.store.settle_open_trades(connection)
            existing_open = self.store.open_trade_for_contract(
                connection,
                account_id=account["id"],
                target_date=target_date,
                bucket_key_value=bucket,
                side=normalized_side,
            )
            connection.commit()
        if existing_open is not None:
            raise DemoTradingError(
                "open demo trade already exists for "
                f"{normalized_side.upper()} {bucket} on {target_date.isoformat()} "
                f"(trade #{existing_open['id']})"
            )

        has_manual_price = manual_price_cents is not None
        market_view = self.refresh_market(target_date, normalized_profile)
        if market_view["status"] != "ok" and has_manual_price:
            with self.store.connect() as connection:
                fallback_snapshot = self.store.latest_successful_snapshot(
                    connection,
                    target_date,
                    as_of_profile=normalized_profile,
                )
                connection.commit()
            if _fallback_snapshot_payload(fallback_snapshot) is not None:
                market_view = self._snapshot_view(fallback_snapshot)
        if market_view["status"] != "ok":
            raise ResourceUnavailable(
                market_view.get("statusReason")
                or market_view.get("status_reason")
                or "Market edge snapshot is unavailable"
            )

        bucket_row = next((row for row in market_view["marketRows"] if row["bucket"] == bucket), None)
        if bucket_row is None:
            raise ResourceUnavailable(f"Bucket {bucket} is missing from market snapshot")

        price_value = bucket_row.get(f"marketBuy{normalized_side.title()}C")
        if has_manual_price:
            entry_price = validate_price_cents(manual_price_cents)
            price_source = "manual_override"
        elif price_value is not None:
            entry_price = validate_price_cents(price_value)
            price_source = str(bucket_row.get(f"marketBuy{normalized_side.title()}Source") or "snapshot_price")
        else:
            raise DemoTradingError("No executable market price is available for this side")

        model_probability_bucket = bucket_row.get("modelProbability")
        if model_probability_bucket is None:
            raise ResourceUnavailable("Model probability is unavailable for this bucket")
        computed = compute_trade(
            stake_usd=stake,
            entry_price_cents=entry_price,
            model_probability_bucket=model_probability_bucket,
            side=normalized_side,
        )
        profile = market_view.get("profile") or {}
        if not has_manual_price:
            if not profile.get("tradeable"):
                raise DemoTradingError(
                    "selected probability profile is not validated for apples-to-apples trading"
                )
            if price_source != "clob_ask":
                raise DemoTradingError("demo trade entry requires an executable CLOB ask price")
            if computed.model_win_probability < MIN_STRATEGY_WIN_PROBABILITY:
                raise DemoTradingError("model win probability must be at least 70.0%")
            if computed.edge_pp < MIN_STRATEGY_EDGE_PP:
                raise DemoTradingError("edge must be at least 15.0 percentage points")

        with self.store.connect() as connection:
            account = self.store.get_or_create_active_account(connection)
            self.store.settle_open_trades(connection)
            existing_open = self.store.open_trade_for_contract(
                connection,
                account_id=account["id"],
                target_date=target_date,
                bucket_key_value=bucket,
                side=normalized_side,
            )
            if existing_open is not None:
                raise DemoTradingError(
                    "open demo trade already exists for "
                    f"{normalized_side.upper()} {bucket} on {target_date.isoformat()} "
                    f"(trade #{existing_open['id']})"
                )
            trades_before = self.store.list_trades(connection, account["id"])
            summary_before = self._summarize_account(connection, account, trades_before, update_marks=False)
            cash = Decimal(str(summary_before["cashUsd"]))
            if computed.stake_usd > cash:
                raise DemoTradingError("stake exceeds available demo cash")
            trade = self.store.insert_trade(
                connection,
                account_id=account["id"],
                snapshot_id=int(market_view["snapshotId"]),
                target_date=target_date,
                event_slug=market_view["event"]["slug"],
                bucket_key_value=bucket,
                side=normalized_side,
                stake_usd=computed.stake_usd,
                shares=computed.shares,
                entry_price_cents=computed.entry_price_cents,
                price_source=price_source,
                model_probability_bucket=computed.model_probability_bucket,
                model_win_probability=computed.model_win_probability,
                edge_pp=computed.edge_pp,
                ev_usd=computed.ev_usd,
                metadata={
                    "bucket_snapshot": bucket_row,
                    "entry": {
                        "manual_price_cents": manual_price_cents,
                        "price_source": price_source,
                        "as_of_profile": market_view.get("as_of_profile"),
                        "profile": profile,
                        "snapshot_created_at_utc": market_view.get("createdAtUtc"),
                        "forecast": market_view.get("forecast"),
                        "model": {
                            "method": (market_view.get("model") or {}).get("method"),
                            "cutoff_profile": (market_view.get("model") or {}).get("cutoff_profile"),
                            "train_rows": (market_view.get("model") or {}).get("train_rows"),
                            "train_start": (market_view.get("model") or {}).get("train_start"),
                            "train_end": (market_view.get("model") or {}).get("train_end"),
                            "profile": (market_view.get("model") or {}).get("profile"),
                        },
                    },
                    "no_real_order": True,
                },
            )
            self.store.settle_open_trades(connection)
            refreshed = self.store.trade_by_id(connection, trade["id"]) or trade
            connection.commit()
        return {"trade": row_to_json(refreshed), "account": self.get_account(None)}

    def settle(self) -> dict[str, Any]:
        with self.store.connect() as connection:
            settled = self.store.settle_open_trades(connection)
            connection.commit()
        return {"settled": [row_to_json(row) for row in settled], "account": self.get_account(None)}

    def settle_trade_as_win(self, trade_id: int) -> dict[str, Any]:
        try:
            normalized_trade_id = int(trade_id)
        except (TypeError, ValueError) as exc:
            raise DemoTradingError("trade id must be an integer") from exc
        if normalized_trade_id <= 0:
            raise DemoTradingError("trade id must be positive")

        with self.store.connect() as connection:
            account = self.store.get_or_create_active_account(connection)
            self.store.settle_open_trades(connection)
            trade = self.store.trade_by_id(connection, normalized_trade_id)
            if trade is None:
                raise DemoTradingError(f"Trade {normalized_trade_id} was not found")
            if trade["account_session_id"] != account["id"]:
                raise DemoTradingError("Trade does not belong to the active demo account")
            if trade["status"] != "open":
                raise DemoTradingError("Only open trades can be settled manually")

            contract_price = self._snapshot_price_for_trade(connection, trade)
            if contract_price is None:
                raise ResourceUnavailable("Current contract price is unavailable for this trade")
            if contract_price < MANUAL_WIN_SETTLE_THRESHOLD_CENTS:
                raise DemoTradingError("Current contract price must be at least 98c to settle as a win")

            settled = self.store.settle_trade_as_win(
                connection,
                trade=trade,
                contract_price_cents=contract_price,
            )
            connection.commit()
        return {"trade": row_to_json(settled), "account": self.get_account(None)}

    def settle_trade_as_loss(self, trade_id: int) -> dict[str, Any]:
        try:
            normalized_trade_id = int(trade_id)
        except (TypeError, ValueError) as exc:
            raise DemoTradingError("trade id must be an integer") from exc
        if normalized_trade_id <= 0:
            raise DemoTradingError("trade id must be positive")

        with self.store.connect() as connection:
            account = self.store.get_or_create_active_account(connection)
            self.store.settle_open_trades(connection)
            trade = self.store.trade_by_id(connection, normalized_trade_id)
            if trade is None:
                raise DemoTradingError(f"Trade {normalized_trade_id} was not found")
            if trade["account_session_id"] != account["id"]:
                raise DemoTradingError("Trade does not belong to the active demo account")
            if trade["status"] != "open":
                raise DemoTradingError("Only open trades can be settled manually")

            contract_price = self._snapshot_price_for_trade(connection, trade)
            if contract_price is None:
                raise ResourceUnavailable("Current contract price is unavailable for this trade")
            marked_value = _marked_value_at_price(
                shares=trade["shares"],
                contract_price_cents=contract_price,
            )
            loss_fraction = _loss_fraction_of_max_loss(
                stake_usd=trade["stake_usd"],
                marked_value_usd=marked_value,
            )
            if loss_fraction < MANUAL_LOSS_SETTLE_THRESHOLD_FRACTION:
                raise DemoTradingError(
                    "Current position loss must be at least 97% of maximum loss to settle as a loss"
                )

            settled = self.store.settle_trade_as_loss(
                connection,
                trade=trade,
                contract_price_cents=contract_price,
                loss_fraction=loss_fraction,
            )
            connection.commit()
        return {"trade": row_to_json(settled), "account": self.get_account(None)}

    def reset_account(self) -> dict[str, Any]:
        with self.store.connect() as connection:
            account = self.store.reset_account(connection)
            connection.commit()
        return {"account": self.get_account(None), "session": row_to_json(account)}

    def get_account(self, since: date | None) -> dict[str, Any]:
        with self.store.connect() as connection:
            account = self.store.get_or_create_active_account(connection)
            self.store.settle_open_trades(connection)
            trades = self.store.list_trades(connection, account["id"])
            summary = self._summarize_account(connection, account, trades, update_marks=True, since=since)
            connection.commit()
        return summary

    def _build_snapshot_payload(
        self,
        *,
        target_date: date,
        event: dict[str, Any],
        markets: dict[str, Any],
        probability: Any | None,
        event_error: str | None,
        probability_error: str | None,
        as_of_profile: str | None,
        fallback_snapshot_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        reused_probability = probability is None and bool(fallback_snapshot_payload)
        model = (
            fallback_snapshot_payload.get("model")
            if reused_probability and fallback_snapshot_payload is not None
            else None if probability is None else probability.model
        )
        forecast = (
            fallback_snapshot_payload.get("forecast")
            if reused_probability and fallback_snapshot_payload is not None
            else None if probability is None else probability.forecast
        )
        forecast_payload = _forecast_payload(forecast)
        market_rows = self._build_market_rows(markets, model)
        status = "ok" if model is not None else "unavailable"
        status_reason = None
        if probability_error:
            status_reason = (
                f"Reused latest successful probability snapshot: {probability_error}"
                if reused_probability
                else probability_error
            )
        if event_error and status == "ok":
            status_reason = f"Market metadata partial: {event_error}"

        edges = self._flatten_edges(market_rows)
        snapshot_profile = (
            fallback_snapshot_payload.get("as_of_profile", "reused_successful_snapshot")
            if reused_probability and fallback_snapshot_payload is not None
            else probability.forecast.as_of_profile
            if probability is not None
            else _display_profile_id(as_of_profile)
        )
        profile = profile_metadata(snapshot_profile)
        if probability is None and not reused_probability:
            profile = {**profile, "forecastAnchorAvailable": False, "tradeable": False}
        else:
            profile = {**profile, "forecastAnchorAvailable": True}
        return {
            "target_date": target_date,
            "as_of_profile": snapshot_profile,
            "profile": profile,
            "status": status,
            "status_reason": status_reason,
            "event": event,
            "forecast": forecast_payload,
            "model": model,
            "marketRows": market_rows,
            "edges": edges,
            "bestEdge": edges[0] if edges else None,
            "rules": {
                "rounding": "31.9C stays bucket 31; 32.0C starts bucket 32; 34.0C+ is 34_or_higher",
                "price_source": "Tradeable entries require CLOB ask side=SELL; Gamma outcome prices are display-only fallback metadata",
                "strategy_gate": "Demo entries require edge >= 15.0 pp and model win probability >= 70%.",
                "scope": "local_demo_only_no_real_orders",
            },
            "errors": {"event": event_error, "probability": probability_error},
        }

    def _fetch_clob_prices(self, markets: dict[str, Any]) -> dict[str, tuple[Decimal | None, str]]:
        token_ids = {
            token
            for market in markets.values()
            for token in (getattr(market, "yes_token", None), getattr(market, "no_token", None))
            if token
        }
        if not token_ids:
            return {}
        results: dict[str, tuple[Decimal | None, str]] = {}
        with ThreadPoolExecutor(max_workers=min(CLOB_PRICE_WORKERS, len(token_ids))) as executor:
            futures = {executor.submit(clob_buy_price, token): token for token in token_ids}
            for future in as_completed(futures):
                token = futures[future]
                try:
                    results[token] = future.result()
                except Exception as exc:  # noqa: BLE001 - quote failures are row metadata, not snapshot blockers.
                    results[token] = (None, f"clob_error:{type(exc).__name__}")
        return results

    def _build_market_rows(self, markets: dict[str, Any], model: dict[str, Any] | None) -> list[dict[str, Any]]:
        probabilities = {} if model is None else model.get("probabilities", {})
        clob_prices = self._fetch_clob_prices(markets)
        rows = []
        for bucket in BUCKET_KEYS:
            probability = None if bucket not in probabilities else float(probabilities[bucket])
            fair_yes = None if probability is None else probability * 100.0
            fair_no = None if probability is None else (1.0 - probability) * 100.0
            market = markets.get(bucket)
            yes_price = None
            yes_source = "missing_market"
            no_price = None
            no_source = "missing_market"
            base = {
                "bucket": bucket,
                "label": bucket,
                "status": "missing_market" if market is None else "ok",
                "marketId": None,
                "marketSlug": None,
                "question": None,
                "active": False,
                "closed": False,
                "acceptingOrders": False,
                "volume": None,
                "liquidity": None,
                "modelProbability": probability,
                "modelYesPct": fair_yes,
                "modelNoPct": fair_no,
            }
            if market is not None:
                yes_decimal, yes_source = (
                    clob_prices.get(market.yes_token, (None, "clob_not_fetched"))
                    if market.yes_token
                    else (None, "missing_token")
                )
                no_decimal, no_source = (
                    clob_prices.get(market.no_token, (None, "clob_not_fetched"))
                    if market.no_token
                    else (None, "missing_token")
                )
                if yes_decimal is None and market.yes_fallback_price is not None:
                    yes_decimal = market.yes_fallback_price
                    yes_source = "gamma_outcome_price_fallback"
                if no_decimal is None and market.no_fallback_price is not None:
                    no_decimal = market.no_fallback_price
                    no_source = "gamma_outcome_price_fallback"
                yes_price = _cents_from_probability_price(yes_decimal)
                no_price = _cents_from_probability_price(no_decimal)
                base.update(
                    {
                        "label": market.label,
                        "marketId": market.market_id,
                        "marketSlug": market.market_slug,
                        "question": market.question,
                        "active": market.active,
                        "closed": market.closed,
                        "acceptingOrders": market.accepting_orders,
                        "volume": market.volume,
                        "liquidity": market.liquidity,
                    }
                )
            yes_edge = None if fair_yes is None or yes_price is None else fair_yes - yes_price
            no_edge = None if fair_no is None or no_price is None else fair_no - no_price
            base.update(
                {
                    "marketBuyYesC": yes_price,
                    "marketBuyYesSource": yes_source,
                    "marketBuyYesExecutable": yes_price is not None and yes_source == "clob_ask",
                    "marketBuyNoC": no_price,
                    "marketBuyNoSource": no_source,
                    "marketBuyNoExecutable": no_price is not None and no_source == "clob_ask",
                    "yesEdgePp": yes_edge,
                    "noEdgePp": no_edge,
                    "yesClass": edge_classification(yes_edge),
                    "noClass": edge_classification(no_edge),
                    "yesEvPerShareC": yes_edge,
                    "noEvPerShareC": no_edge,
                }
            )
            rows.append(base)
        return rows

    def _flatten_edges(self, market_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        for row in market_rows:
            for side in ("yes", "no"):
                price = row.get(f"marketBuy{side.title()}C")
                edge = row.get(f"{side}EdgePp")
                fair = row.get(f"model{side.title()}Pct")
                if price is None or edge is None or fair is None:
                    continue
                edges.append(
                    {
                        "side": side,
                        "bucket": row["bucket"],
                        "label": row["label"],
                        "marketPriceC": price,
                        "modelFairC": fair,
                        "edgePp": edge,
                        "classification": edge_classification(edge),
                        "priceSource": row.get(f"marketBuy{side.title()}Source"),
                        "executable": bool(row.get(f"marketBuy{side.title()}Executable")),
                    }
                )
        return sorted(edges, key=lambda item: item["edgePp"], reverse=True)

    def _snapshot_view(self, row: dict[str, Any]) -> dict[str, Any]:
        payload = dict(row["snapshot_json"])
        payload["snapshotId"] = row["id"]
        payload["createdAtUtc"] = row["created_at_utc"].isoformat()
        payload["targetDate"] = row["target_date"].isoformat()
        return json_safe(payload)

    def _snapshot_price_for_trade(self, connection: Any, trade: dict[str, Any]) -> Decimal | None:
        latest = self.store.latest_snapshot(connection, trade["target_date"])
        if latest is None:
            return None
        rows = latest["snapshot_json"].get("marketRows") or []
        bucket = trade["bucket_key"]
        side = trade["side"]
        row = next((item for item in rows if item.get("bucket") == bucket), None)
        if not row:
            return None
        key = f"marketBuy{side.title()}C"
        value = row.get(key)
        if value is None:
            return None
        return validate_price_cents(value)

    def _mark_price_for_trade(self, connection: Any, trade: dict[str, Any]) -> Decimal:
        snapshot_price = self._snapshot_price_for_trade(connection, trade)
        if snapshot_price is not None:
            return snapshot_price
        return Decimal(str(trade["entry_price_cents"]))

    def _summarize_account(
        self,
        connection: Any,
        account: dict[str, Any],
        trades: list[dict[str, Any]],
        *,
        update_marks: bool,
        since: date | None = None,
    ) -> dict[str, Any]:
        starting = Decimal(str(account["starting_balance_usd"]))
        cash = starting
        marked_value = Decimal("0")
        realized = Decimal("0")
        unrealized = Decimal("0")
        open_exposure = Decimal("0")
        trade_views = []
        for trade in trades:
            stake = Decimal(str(trade["stake_usd"]))
            shares = Decimal(str(trade["shares"]))
            cash -= stake
            if trade["status"] == "settled":
                pnl = Decimal(str(trade["realized_pnl_usd"]))
                realized += pnl
                cash += stake + pnl
                trade_view = row_to_json(trade)
            elif trade["status"] == "open":
                open_exposure += stake
                current_price = self._snapshot_price_for_trade(connection, trade)
                mark_price = current_price or Decimal(str(trade["entry_price_cents"]))
                mark_value = (shares * mark_price / Decimal("100")).quantize(
                    Decimal("0.000001"), rounding=ROUND_HALF_UP
                )
                mark_pnl = (mark_value - stake).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
                loss_fraction = _loss_fraction_of_max_loss(
                    stake_usd=stake,
                    marked_value_usd=mark_value,
                )
                manual_win_eligible = (
                    current_price is not None and current_price >= MANUAL_WIN_SETTLE_THRESHOLD_CENTS
                )
                manual_loss_eligible = (
                    current_price is not None
                    and loss_fraction >= MANUAL_LOSS_SETTLE_THRESHOLD_FRACTION
                )
                marked_value += mark_value
                unrealized += mark_pnl
                if update_marks:
                    self.store.update_open_mark(connection, trade["id"], mark_value, mark_pnl)
                trade_view = row_to_json(
                    {
                        **trade,
                        "current_price_cents": current_price,
                        "manual_settle_eligible": manual_win_eligible or manual_loss_eligible,
                        "manual_settle_result": "win"
                        if manual_win_eligible
                        else "loss"
                        if manual_loss_eligible
                        else None,
                        "manual_win_settle_eligible": manual_win_eligible,
                        "manual_loss_settle_eligible": manual_loss_eligible,
                        "manual_settle_threshold_cents": MANUAL_WIN_SETTLE_THRESHOLD_CENTS,
                        "manual_win_settle_threshold_cents": MANUAL_WIN_SETTLE_THRESHOLD_CENTS,
                        "manual_loss_settle_threshold_fraction": MANUAL_LOSS_SETTLE_THRESHOLD_FRACTION,
                        "manual_loss_fraction_of_max_loss": loss_fraction,
                        "marked_value_usd": mark_value,
                        "unrealized_pnl_usd": mark_pnl,
                    }
                )
            else:
                trade_view = row_to_json(trade)
            trade_views.append(trade_view)

        equity = cash + marked_value
        total_pnl = equity - starting
        since_pnl = total_pnl
        if since is not None:
            since_pnl = Decimal("0")
            for trade in trades:
                opened_at = trade["opened_at_utc"]
                opened_date = opened_at.date() if isinstance(opened_at, datetime) else date.fromisoformat(str(opened_at)[:10])
                if opened_date < since:
                    continue
                if trade["status"] == "settled":
                    since_pnl += Decimal(str(trade["realized_pnl_usd"]))
                elif trade["status"] == "open":
                    since_pnl += Decimal(str(trade.get("unrealized_pnl_usd") or "0"))

        balance_curve = self._balance_curve(starting, trades, total_pnl)
        return {
            "sessionId": account["id"],
            "startedAtUtc": account["started_at_utc"].isoformat(),
            "startingBalanceUsd": decimal_to_float(starting),
            "cashUsd": decimal_to_float(quantize(cash)),
            "openExposureUsd": decimal_to_float(quantize(open_exposure)),
            "markedValueUsd": decimal_to_float(quantize(marked_value)),
            "equityUsd": decimal_to_float(quantize(equity)),
            "realizedPnlUsd": decimal_to_float(quantize(realized)),
            "unrealizedPnlUsd": decimal_to_float(quantize(unrealized)),
            "totalPnlUsd": decimal_to_float(quantize(total_pnl)),
            "changeSinceUsd": decimal_to_float(quantize(since_pnl)),
            "since": None if since is None else since.isoformat(),
            "openTrades": [trade for trade in trade_views if trade["status"] == "open"],
            "trades": trade_views,
            "balanceCurve": balance_curve,
        }

    def _balance_curve(
        self,
        starting_balance: Decimal,
        trades: list[dict[str, Any]],
        current_total_pnl: Decimal,
    ) -> list[dict[str, Any]]:
        points = [{"date": "start", "equityUsd": decimal_to_float(starting_balance), "pnlUsd": 0.0}]
        cumulative = Decimal("0")
        settled = [trade for trade in trades if trade["status"] == "settled"]
        settled.sort(key=lambda row: row.get("settled_at_utc") or row.get("opened_at_utc"))
        for trade in settled:
            cumulative += Decimal(str(trade["realized_pnl_usd"]))
            settled_at = trade["settled_at_utc"]
            label = settled_at.date().isoformat() if isinstance(settled_at, datetime) else str(settled_at)[:10]
            points.append(
                {
                    "date": label,
                    "equityUsd": decimal_to_float(quantize(starting_balance + cumulative)),
                    "pnlUsd": decimal_to_float(quantize(cumulative)),
                }
            )
        points.append(
            {
                "date": datetime.utcnow().date().isoformat(),
                "equityUsd": decimal_to_float(quantize(starting_balance + current_total_pnl)),
                "pnlUsd": decimal_to_float(quantize(current_total_pnl)),
            }
        )
        return points
