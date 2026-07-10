BEGIN;

CREATE SCHEMA IF NOT EXISTS demo_trading;

CREATE TABLE IF NOT EXISTS demo_trading.account_session (
    id bigserial PRIMARY KEY,
    started_at_utc timestamp with time zone NOT NULL DEFAULT now(),
    ended_at_utc timestamp with time zone,
    starting_balance_usd numeric(18, 6) NOT NULL DEFAULT 1000.000000,
    active boolean NOT NULL DEFAULT true,
    reset_reason text,
    metadata_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    CHECK (starting_balance_usd > 0),
    CHECK ((active = true AND ended_at_utc IS NULL) OR active = false)
);

CREATE UNIQUE INDEX IF NOT EXISTS account_session_single_active_idx
    ON demo_trading.account_session ((active))
    WHERE active;

CREATE TABLE IF NOT EXISTS demo_trading.market_snapshot (
    id bigserial PRIMARY KEY,
    target_date date NOT NULL,
    event_slug text NOT NULL,
    event_url text NOT NULL,
    event_title text NOT NULL,
    as_of_profile text NOT NULL,
    status text NOT NULL,
    status_reason text,
    forecast_source text,
    forecast_update_time_hkt timestamp with time zone,
    forecast_min_c numeric(8, 3),
    forecast_max_c numeric(8, 3),
    model_method text,
    train_rows integer,
    train_start date,
    train_end date,
    snapshot_json jsonb NOT NULL,
    created_at_utc timestamp with time zone NOT NULL DEFAULT now(),
    CHECK (status IN ('ok', 'unavailable', 'error')),
    CHECK (forecast_min_c IS NULL OR forecast_min_c BETWEEN -20 AND 60),
    CHECK (forecast_max_c IS NULL OR forecast_max_c BETWEEN -20 AND 60),
    CHECK (train_rows IS NULL OR train_rows >= 0)
);

CREATE INDEX IF NOT EXISTS market_snapshot_target_date_created_idx
    ON demo_trading.market_snapshot (target_date, created_at_utc DESC);

CREATE INDEX IF NOT EXISTS market_snapshot_status_idx
    ON demo_trading.market_snapshot (status);

CREATE INDEX IF NOT EXISTS market_snapshot_json_gin_idx
    ON demo_trading.market_snapshot USING gin (snapshot_json);

CREATE TABLE IF NOT EXISTS demo_trading.trade (
    id bigserial PRIMARY KEY,
    account_session_id bigint NOT NULL REFERENCES demo_trading.account_session(id),
    snapshot_id bigint NOT NULL REFERENCES demo_trading.market_snapshot(id),
    target_date date NOT NULL,
    event_slug text NOT NULL,
    bucket_key text NOT NULL,
    side text NOT NULL,
    stake_usd numeric(18, 6) NOT NULL,
    shares numeric(18, 8) NOT NULL,
    entry_price_cents numeric(10, 4) NOT NULL,
    price_source text NOT NULL,
    model_probability_bucket numeric(12, 10) NOT NULL,
    model_win_probability numeric(12, 10) NOT NULL,
    edge_pp numeric(10, 4) NOT NULL,
    ev_usd numeric(18, 6) NOT NULL,
    status text NOT NULL DEFAULT 'open',
    result text,
    settlement_tmax_c numeric(8, 3),
    settlement_bucket_key text,
    settled_at_utc timestamp with time zone,
    realized_pnl_usd numeric(18, 6) NOT NULL DEFAULT 0,
    marked_value_usd numeric(18, 6) NOT NULL DEFAULT 0,
    unrealized_pnl_usd numeric(18, 6) NOT NULL DEFAULT 0,
    opened_at_utc timestamp with time zone NOT NULL DEFAULT now(),
    metadata_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    CHECK (bucket_key IN ('24_or_below','25','26','27','28','29','30','31','32','33','34_or_higher')),
    CHECK (side IN ('yes', 'no')),
    CHECK (stake_usd > 0),
    CHECK (shares > 0),
    CHECK (entry_price_cents > 0 AND entry_price_cents <= 100),
    CHECK (model_probability_bucket >= 0 AND model_probability_bucket <= 1),
    CHECK (model_win_probability >= 0 AND model_win_probability <= 1),
    CHECK (status IN ('open', 'settled', 'cancelled')),
    CONSTRAINT trade_result_check CHECK (result IS NULL OR result IN ('win', 'loss')),
    CHECK (settlement_bucket_key IS NULL OR settlement_bucket_key IN ('24_or_below','25','26','27','28','29','30','31','32','33','34_or_higher'))
);

ALTER TABLE demo_trading.trade
    ADD COLUMN IF NOT EXISTS result text;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'trade_result_check'
          AND conrelid = 'demo_trading.trade'::regclass
    ) THEN
        ALTER TABLE demo_trading.trade
            ADD CONSTRAINT trade_result_check
            CHECK (result IS NULL OR result IN ('win', 'loss'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS trade_account_opened_idx
    ON demo_trading.trade (account_session_id, opened_at_utc DESC);

CREATE INDEX IF NOT EXISTS trade_status_target_date_idx
    ON demo_trading.trade (status, target_date);

CREATE UNIQUE INDEX IF NOT EXISTS trade_one_open_position_per_contract_idx
    ON demo_trading.trade (account_session_id, target_date, bucket_key, side)
    WHERE status = 'open';

CREATE INDEX IF NOT EXISTS trade_metadata_gin_idx
    ON demo_trading.trade USING gin (metadata_json);

COMMENT ON SCHEMA demo_trading IS
    'Local-only fictitious Polymarket demo trading ledger for HKG Tmax backtesting. No real orders or credentials.';

COMMENT ON TABLE demo_trading.market_snapshot IS
    'Frozen HKG market, forecast, probability, price, and edge metadata captured for demo trade auditability.';

COMMENT ON TABLE demo_trading.trade IS
    'Fictitious demo account trades with exact entry metadata, model edge, settlement, and PnL fields.';

COMMIT;
