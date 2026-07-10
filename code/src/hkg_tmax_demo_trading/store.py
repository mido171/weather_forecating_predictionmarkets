"""PostgreSQL persistence for the local demo trading ledger."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import psycopg
from hkg_tmax_probability.bucket_rules import bucket_key
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .domain import (
    DEFAULT_START_BALANCE_USD,
    DemoTradingError,
    did_trade_win,
    quantize,
    realized_pnl_usd,
)

MIGRATION_PATH = (
    Path(__file__).resolve().parents[3]
    / "migrations"
    / "postgres"
    / "20260706_0009_demo_trading_backtester.sql"
)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


class PgDemoTradingStore:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    @contextmanager
    def connect(self) -> Iterator[Any]:
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            yield connection

    def apply_schema(self) -> None:
        sql = MIGRATION_PATH.read_text(encoding="utf-8")
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql)
            connection.commit()

    def get_or_create_active_account(self, connection: Any) -> dict[str, Any]:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM demo_trading.account_session
                WHERE active
                ORDER BY started_at_utc DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            cursor.execute(
                """
                INSERT INTO demo_trading.account_session (starting_balance_usd, metadata_json)
                VALUES (%s, %s)
                RETURNING *
                """,
                (DEFAULT_START_BALANCE_USD, Jsonb({"created_by": "hkg_tmax_demo_trading"})),
            )
            return dict(cursor.fetchone())

    def reset_account(self, connection: Any) -> dict[str, Any]:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE demo_trading.account_session
                SET active = false,
                    ended_at_utc = now(),
                    reset_reason = 'user_reset'
                WHERE active
                """
            )
            cursor.execute(
                """
                INSERT INTO demo_trading.account_session (starting_balance_usd, metadata_json)
                VALUES (%s, %s)
                RETURNING *
                """,
                (DEFAULT_START_BALANCE_USD, Jsonb({"created_by": "hkg_tmax_demo_trading", "reset": True})),
            )
            return dict(cursor.fetchone())

    def insert_snapshot(self, connection: Any, snapshot: dict[str, Any]) -> dict[str, Any]:
        forecast = snapshot.get("forecast") or {}
        model = snapshot.get("model") or {}
        event = snapshot.get("event") or {}
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO demo_trading.market_snapshot (
                    target_date,
                    event_slug,
                    event_url,
                    event_title,
                    as_of_profile,
                    status,
                    status_reason,
                    forecast_source,
                    forecast_update_time_hkt,
                    forecast_min_c,
                    forecast_max_c,
                    model_method,
                    train_rows,
                    train_start,
                    train_end,
                    snapshot_json
                )
                VALUES (
                    %(target_date)s,
                    %(event_slug)s,
                    %(event_url)s,
                    %(event_title)s,
                    %(as_of_profile)s,
                    %(status)s,
                    %(status_reason)s,
                    %(forecast_source)s,
                    %(forecast_update_time_hkt)s,
                    %(forecast_min_c)s,
                    %(forecast_max_c)s,
                    %(model_method)s,
                    %(train_rows)s,
                    %(train_start)s,
                    %(train_end)s,
                    %(snapshot_json)s
                )
                RETURNING *
                """,
                {
                    "target_date": snapshot["target_date"],
                    "event_slug": event["slug"],
                    "event_url": event["url"],
                    "event_title": event["title"],
                    "as_of_profile": snapshot.get("as_of_profile") or "unknown",
                    "status": snapshot["status"],
                    "status_reason": snapshot.get("status_reason"),
                    "forecast_source": forecast.get("source"),
                    "forecast_update_time_hkt": forecast.get("update_time_hkt"),
                    "forecast_min_c": forecast.get("forecast_min_c"),
                    "forecast_max_c": forecast.get("forecast_max_c"),
                    "model_method": model.get("method"),
                    "train_rows": model.get("train_rows"),
                    "train_start": model.get("train_start"),
                    "train_end": model.get("train_end"),
                    "snapshot_json": Jsonb(json_safe(snapshot)),
                },
            )
            return dict(cursor.fetchone())

    def latest_snapshot(
        self,
        connection: Any,
        target_date: date,
        *,
        as_of_profile: str | None = None,
    ) -> dict[str, Any] | None:
        with connection.cursor() as cursor:
            if as_of_profile:
                cursor.execute(
                    """
                    SELECT *
                    FROM demo_trading.market_snapshot
                    WHERE target_date = %s
                      AND as_of_profile = %s
                    ORDER BY created_at_utc DESC
                    LIMIT 1
                    """,
                    (target_date, as_of_profile),
                )
            else:
                cursor.execute(
                    """
                    SELECT *
                    FROM demo_trading.market_snapshot
                    WHERE target_date = %s
                    ORDER BY created_at_utc DESC
                    LIMIT 1
                    """,
                    (target_date,),
                )
            row = cursor.fetchone()
        return None if row is None else dict(row)

    def latest_successful_snapshot(
        self,
        connection: Any,
        target_date: date,
        *,
        as_of_profile: str | None = None,
    ) -> dict[str, Any] | None:
        with connection.cursor() as cursor:
            if as_of_profile:
                cursor.execute(
                    """
                    SELECT *
                    FROM demo_trading.market_snapshot
                    WHERE target_date = %s
                      AND as_of_profile = %s
                      AND status = 'ok'
                    ORDER BY created_at_utc DESC
                    LIMIT 1
                    """,
                    (target_date, as_of_profile),
                )
            else:
                cursor.execute(
                    """
                    SELECT *
                    FROM demo_trading.market_snapshot
                    WHERE target_date = %s
                      AND status = 'ok'
                    ORDER BY created_at_utc DESC
                    LIMIT 1
                    """,
                    (target_date,),
                )
            row = cursor.fetchone()
        return None if row is None else dict(row)

    def latest_snapshots_for_range(
        self,
        connection: Any,
        start: date,
        end: date,
    ) -> dict[date, dict[str, Any]]:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT DISTINCT ON (target_date) *
                FROM demo_trading.market_snapshot
                WHERE target_date BETWEEN %s AND %s
                ORDER BY target_date, created_at_utc DESC
                """,
                (start, end),
            )
            rows = cursor.fetchall()
        return {row["target_date"]: dict(row) for row in rows}

    def find_settlement(self, connection: Any, target_date: date) -> dict[str, Any] | None:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT local_date, target_tmax_c, target_station, target_source_id, quality_status, source_rank
                FROM (
                    SELECT local_date, target_tmax_c, target_station, target_source_id, quality_status, 1 AS source_rank
                    FROM sealed_confirmation.hko_daily_tmax
                    WHERE local_date = %s AND quality_status = 'VALID'
                    UNION ALL
                    SELECT local_date, target_tmax_c, target_station, target_source_id, quality_status, 2 AS source_rank
                    FROM label_core.hko_daily_tmax
                    WHERE local_date = %s AND quality_status = 'VALID'
                ) candidates
                ORDER BY source_rank
                LIMIT 1
                """,
                (target_date, target_date),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        result = dict(row)
        result["settlement_bucket_key"] = bucket_key(result["target_tmax_c"])
        return result

    def insert_trade(
        self,
        connection: Any,
        *,
        account_id: int,
        snapshot_id: int,
        target_date: date,
        event_slug: str,
        bucket_key_value: str,
        side: str,
        stake_usd: Decimal,
        shares: Decimal,
        entry_price_cents: Decimal,
        price_source: str,
        model_probability_bucket: Decimal,
        model_win_probability: Decimal,
        edge_pp: Decimal,
        ev_usd: Decimal,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO demo_trading.trade (
                    account_session_id,
                    snapshot_id,
                    target_date,
                    event_slug,
                    bucket_key,
                    side,
                    stake_usd,
                    shares,
                    entry_price_cents,
                    price_source,
                    model_probability_bucket,
                    model_win_probability,
                    edge_pp,
                    ev_usd,
                    metadata_json
                )
                VALUES (
                    %(account_session_id)s,
                    %(snapshot_id)s,
                    %(target_date)s,
                    %(event_slug)s,
                    %(bucket_key)s,
                    %(side)s,
                    %(stake_usd)s,
                    %(shares)s,
                    %(entry_price_cents)s,
                    %(price_source)s,
                    %(model_probability_bucket)s,
                    %(model_win_probability)s,
                    %(edge_pp)s,
                    %(ev_usd)s,
                    %(metadata_json)s
                )
                RETURNING *
                """,
                {
                    "account_session_id": account_id,
                    "snapshot_id": snapshot_id,
                    "target_date": target_date,
                    "event_slug": event_slug,
                    "bucket_key": bucket_key_value,
                    "side": side,
                    "stake_usd": stake_usd,
                    "shares": shares,
                    "entry_price_cents": entry_price_cents,
                    "price_source": price_source,
                    "model_probability_bucket": model_probability_bucket,
                    "model_win_probability": model_win_probability,
                    "edge_pp": edge_pp,
                    "ev_usd": ev_usd,
                    "metadata_json": Jsonb(json_safe(metadata)),
                },
            )
            return dict(cursor.fetchone())

    def open_trade_for_contract(
        self,
        connection: Any,
        *,
        account_id: int,
        target_date: date,
        bucket_key_value: str,
        side: str,
    ) -> dict[str, Any] | None:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM demo_trading.trade
                WHERE account_session_id = %s
                  AND target_date = %s
                  AND bucket_key = %s
                  AND side = %s
                  AND status = 'open'
                ORDER BY opened_at_utc DESC, id DESC
                LIMIT 1
                """,
                (account_id, target_date, bucket_key_value, side),
            )
            row = cursor.fetchone()
        return None if row is None else dict(row)

    def list_trades(self, connection: Any, account_id: int) -> list[dict[str, Any]]:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM demo_trading.trade
                WHERE account_session_id = %s
                ORDER BY opened_at_utc DESC, id DESC
                """,
                (account_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def open_trades(self, connection: Any) -> list[dict[str, Any]]:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM demo_trading.trade
                WHERE status = 'open'
                ORDER BY target_date, id
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def update_open_mark(
        self,
        connection: Any,
        trade_id: int,
        marked_value_usd: Decimal,
        unrealized_pnl_usd: Decimal,
    ) -> None:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE demo_trading.trade
                SET marked_value_usd = %s,
                    unrealized_pnl_usd = %s
                WHERE id = %s AND status = 'open'
                """,
                (marked_value_usd, unrealized_pnl_usd, trade_id),
            )

    def settle_open_trades(self, connection: Any) -> list[dict[str, Any]]:
        settled: list[dict[str, Any]] = []
        for trade in self.open_trades(connection):
            settlement = self.find_settlement(connection, trade["target_date"])
            if settlement is None:
                continue
            settlement_bucket = settlement["settlement_bucket_key"]
            did_win = did_trade_win(trade["side"], trade["bucket_key"], settlement_bucket)
            result = "win" if did_win else "loss"
            pnl = realized_pnl_usd(
                stake_usd=Decimal(str(trade["stake_usd"])),
                shares=Decimal(str(trade["shares"])),
                did_win=did_win,
            )
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE demo_trading.trade
                    SET status = 'settled',
                        result = %(result)s,
                        settlement_tmax_c = %(settlement_tmax_c)s,
                        settlement_bucket_key = %(settlement_bucket_key)s,
                        settled_at_utc = now(),
                        realized_pnl_usd = %(realized_pnl_usd)s,
                        marked_value_usd = 0,
                        unrealized_pnl_usd = 0,
                        metadata_json = metadata_json || %(settlement_metadata)s
                    WHERE id = %(trade_id)s
                    RETURNING *
                    """,
                    {
                        "settlement_tmax_c": settlement["target_tmax_c"],
                        "settlement_bucket_key": settlement_bucket,
                        "result": result,
                        "realized_pnl_usd": pnl,
                        "settlement_metadata": Jsonb(
                            json_safe(
                                {
                                    "settlement": {
                                        "result": result,
                                        "target_tmax_c": settlement["target_tmax_c"],
                                        "bucket_key": settlement_bucket,
                                        "did_win": did_win,
                                        "source": settlement["target_source_id"],
                                    }
                                }
                            )
                        ),
                        "trade_id": trade["id"],
                    },
                )
                settled.append(dict(cursor.fetchone()))
        return settled

    def settle_trade_as_win(
        self,
        connection: Any,
        *,
        trade: dict[str, Any],
        contract_price_cents: Decimal,
    ) -> dict[str, Any]:
        pnl = realized_pnl_usd(
            stake_usd=Decimal(str(trade["stake_usd"])),
            shares=Decimal(str(trade["shares"])),
            did_win=True,
        )
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE demo_trading.trade
                SET status = 'settled',
                    result = 'win',
                    settlement_tmax_c = NULL,
                    settlement_bucket_key = NULL,
                    settled_at_utc = now(),
                    realized_pnl_usd = %(realized_pnl_usd)s,
                    marked_value_usd = 0,
                    unrealized_pnl_usd = 0,
                    metadata_json = metadata_json || %(settlement_metadata)s
                WHERE id = %(trade_id)s
                  AND status = 'open'
                RETURNING *
                """,
                {
                    "realized_pnl_usd": pnl,
                    "settlement_metadata": Jsonb(
                        json_safe(
                            {
                                "settlement": {
                                    "result": "win",
                                    "did_win": True,
                                    "source": "manual_contract_price_threshold",
                                    "contract_price_cents": contract_price_cents,
                                    "threshold_cents": Decimal("98"),
                                }
                            }
                        )
                    ),
                    "trade_id": trade["id"],
                },
            )
            row = cursor.fetchone()
        if row is None:
            raise DemoTradingError("Only open trades can be settled manually")
        return dict(row)

    def settle_trade_as_loss(
        self,
        connection: Any,
        *,
        trade: dict[str, Any],
        contract_price_cents: Decimal,
        loss_fraction: Decimal,
    ) -> dict[str, Any]:
        pnl = realized_pnl_usd(
            stake_usd=Decimal(str(trade["stake_usd"])),
            shares=Decimal(str(trade["shares"])),
            did_win=False,
        )
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE demo_trading.trade
                SET status = 'settled',
                    result = 'loss',
                    settlement_tmax_c = NULL,
                    settlement_bucket_key = NULL,
                    settled_at_utc = now(),
                    realized_pnl_usd = %(realized_pnl_usd)s,
                    marked_value_usd = 0,
                    unrealized_pnl_usd = 0,
                    metadata_json = metadata_json || %(settlement_metadata)s
                WHERE id = %(trade_id)s
                  AND status = 'open'
                RETURNING *
                """,
                {
                    "realized_pnl_usd": pnl,
                    "settlement_metadata": Jsonb(
                        json_safe(
                            {
                                "settlement": {
                                    "result": "loss",
                                    "did_win": False,
                                    "source": "manual_position_loss_threshold",
                                    "contract_price_cents": contract_price_cents,
                                    "loss_fraction_of_max_loss": loss_fraction,
                                    "threshold_loss_fraction": Decimal("0.97"),
                                }
                            }
                        )
                    ),
                    "trade_id": trade["id"],
                },
            )
            row = cursor.fetchone()
        if row is None:
            raise DemoTradingError("Only open trades can be settled manually")
        return dict(row)

    def trade_by_id(self, connection: Any, trade_id: int) -> dict[str, Any] | None:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM demo_trading.trade WHERE id = %s", (trade_id,))
            row = cursor.fetchone()
        return None if row is None else dict(row)


def row_to_json(row: dict[str, Any]) -> dict[str, Any]:
    return json_safe(dict(row))


def decimal_sum(values: list[Decimal]) -> Decimal:
    total = Decimal("0")
    for value in values:
        total += value
    return quantize(total)
