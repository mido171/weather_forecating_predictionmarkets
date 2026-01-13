# WX-303 — Extend DB schema + models module for Kalshi market data + backtest results (Flyway + JPA)

## Objective
Extend the shared `models` module (Flyway migrations + JPA entities) to persist:
1) Kalshi market metadata for weather series (events + markets)
2) Historical Kalshi quote/price series (candlesticks)
3) Backtest runs + executed simulated trades
4) Ingestion checkpoints for idempotent restartability

## Why this matters
Epic #3 must be:
- idempotent (restartable)
- auditable (raw data + timestamps)
- fast (backtest should read from DB, not call external APIs mid-run)

## Requirements

## A) Tables (minimum)
### 1) `kalshi_event`
Stores per-event metadata (one per trading day / event_ticker).
- `event_ticker` (PK)
- `series_ticker` (FK → existing `kalshi_series.series_ticker`)
- `strike_date_utc` (TIMESTAMP NULL) — if available
- `open_time_utc` (TIMESTAMP NULL)
- `close_time_utc` (TIMESTAMP NULL)
- `created_at_utc` (TIMESTAMP)
- `updated_at_utc` (TIMESTAMP)
- `raw_json` (JSON/TEXT optional)

Indexes:
- `(series_ticker, close_time_utc)`
- `(series_ticker, event_ticker)` unique implied by PK

### 2) `kalshi_market`
Stores per-market/bin metadata.
- `market_ticker` (PK)
- `event_ticker` (FK → kalshi_event.event_ticker)
- `series_ticker` (redundant FK for easier queries; keep consistent via app logic)
- `title` (VARCHAR)
- `subtitle` (VARCHAR NULL)
- `strike_type` (VARCHAR) — e.g. between/lt/gt (exact values from API)
- `floor_strike` (DECIMAL NULL)
- `cap_strike` (DECIMAL NULL)
- `functional_strike` (VARCHAR/DECIMAL NULL)
- `open_time_utc` / `close_time_utc` / `expiration_time_utc` (TIMESTAMP NULL)
- `status` (VARCHAR) — open/closed/settled
- `result` (VARCHAR NULL) — yes/no (when settled)
- `settlement_ts_utc` (TIMESTAMP NULL)
- `last_price` (DECIMAL NULL)
- `volume` (BIGINT NULL)
- `open_interest` (BIGINT NULL)
- `updated_at_utc` (TIMESTAMP)
- `raw_json` (JSON/TEXT optional)

Indexes:
- `(series_ticker, event_ticker)`
- `(event_ticker, strike_type, floor_strike, cap_strike)`

### 3) `kalshi_candlestick_1m`
Stores **1-minute** candlesticks (top-of-book proxy).
Natural key:
- `(market_ticker, end_period_ts_utc)` with `period_interval_minutes = 1`

Columns (store both cents and dollars if API returns both; otherwise normalize):
- `market_ticker` (FK)
- `period_interval_minutes` (SMALLINT, default 1)
- `end_period_ts_utc` (TIMESTAMP)
- `yes_bid_open` / `yes_bid_high` / `yes_bid_low` / `yes_bid_close` (DECIMAL NULL)
- `yes_ask_open` / `yes_ask_high` / `yes_ask_low` / `yes_ask_close` (DECIMAL NULL)
- `price_open` / `price_high` / `price_low` / `price_close` (DECIMAL NULL)
- `volume` (BIGINT NULL)
- `open_interest` (BIGINT NULL)
- `retrieved_at_utc` (TIMESTAMP)
- `raw_json` (JSON/TEXT optional)

Indexes:
- PK/unique: `(market_ticker, end_period_ts_utc, period_interval_minutes)`
- `(end_period_ts_utc)` for time-sliced queries

**Note:** If we later add 60m or 1440m, create separate tables or include period_interval in PK and index heavily.

### 4) `kalshi_backfill_checkpoint`
Generic checkpoint table to support idempotent backfills.
- `job_name` (PK part) — e.g. `KALSHI_EVENT_CANDLES_1M`
- `series_ticker` (PK part)
- `event_ticker` (PK part)
- `window_start_ts_utc` (TIMESTAMP)
- `window_end_ts_utc` (TIMESTAMP)
- `last_success_ts_utc` (TIMESTAMP NULL)
- `status` (ENUM: RUNNING, SUCCESS, FAILED)
- `notes` (TEXT)

### 5) `backtest_run`
- `run_id` (PK; UUID)
- `created_at_utc`
- `git_sha` (VARCHAR NULL)
- `random_seed` (BIGINT)
- `config_yaml` (TEXT) — resolved config snapshot
- `date_start_local` / `date_end_local` (DATE)
- `series_tickers` (TEXT/JSON)
- `asof_policy_id` (FK to asof_policy)
- `trade_window_policy` (VARCHAR)
- `status` (ENUM: RUNNING, SUCCESS, FAILED)
- `metrics_json` (JSON/TEXT)
- `artifact_path` (VARCHAR) — filesystem path for run artifacts

### 6) `backtest_trade`
- `run_id` (FK)
- `trade_id` (PK; UUID)
- `station_id` (FK to station_registry)
- `series_ticker`
- `event_ticker`
- `market_ticker`
- `decision_ts_utc`
- `side` (ENUM: BUY_YES, BUY_NO)
- `qty` (INT)
- `entry_price` (DECIMAL) — dollars
- `entry_fee` (DECIMAL) — dollars
- `model_prob_yes` (DECIMAL)
- `model_ev_per_contract` (DECIMAL)
- `settled_result` (VARCHAR) — yes/no (copied from market)
- `pnl` (DECIMAL)
- `notes` (TEXT)

Indexes:
- `(run_id, event_ticker)`
- `(market_ticker, decision_ts_utc)`

## B) Flyway + JPA deliverables
- Flyway migration file(s) in models module:
  - versioned, repeatable if needed
- JPA entities with:
  - correct PKs and indexes
  - JSON raw payload stored as TEXT/JSON
- README update: how to run migrations locally

## Acceptance Criteria
- [ ] `mvn clean install` passes with new Flyway migrations and entities.
- [ ] Database can be migrated from empty to latest with Flyway.
- [ ] Each table has correct unique constraints to prevent duplicates on retry.
- [ ] A short schema doc is added under `models/docs/kalshi-backtest-schema.md`.
