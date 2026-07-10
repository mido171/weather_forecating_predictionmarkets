from __future__ import annotations

from pathlib import Path


def test_demo_trading_migration_contract(repo_root: Path) -> None:
    sql = (
        repo_root
        / "migrations"
        / "postgres"
        / "20260706_0009_demo_trading_backtester.sql"
    ).read_text(encoding="utf-8")

    assert "CREATE SCHEMA IF NOT EXISTS demo_trading" in sql
    assert "CREATE TABLE IF NOT EXISTS demo_trading.account_session" in sql
    assert "CREATE TABLE IF NOT EXISTS demo_trading.market_snapshot" in sql
    assert "CREATE TABLE IF NOT EXISTS demo_trading.trade" in sql
    assert "starting_balance_usd numeric(18, 6) NOT NULL DEFAULT 1000.000000" in sql
    assert "snapshot_json jsonb NOT NULL" in sql
    assert "metadata_json jsonb NOT NULL DEFAULT '{}'::jsonb" in sql
    assert "result text" in sql
    assert "account_session_single_active_idx" in sql
    assert "market_snapshot_json_gin_idx" in sql
    assert "trade_metadata_gin_idx" in sql
    assert "CHECK (side IN ('yes', 'no'))" in sql
    assert "CHECK (entry_price_cents > 0 AND entry_price_cents <= 100)" in sql
    assert "CHECK (result IS NULL OR result IN ('win', 'loss'))" in sql
    assert "No real orders or credentials" in sql
