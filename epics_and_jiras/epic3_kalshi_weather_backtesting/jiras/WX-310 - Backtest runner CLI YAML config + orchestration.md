# WX-310 — Backtest runner CLI (YAML config) + orchestration (data checks, optional auto-backfill)

## Objective
Create a single CLI entrypoint that:
- reads a YAML config
- validates config
- ensures required Kalshi data exists (catalog + candlesticks)
- runs the backtest engine for the configured series/date range
- writes artifacts + DB rows

## Requirements

### 1) CLI interface
Example:
- `python -m weather_backtest.cli run --config backtest/configs/backtest.yaml`

Must support:
- `--dry-run` (validate config + show planned work)
- `--no-fetch` (fail if data missing; never call Kalshi API)
- `--auto-fetch` (default): run missing backfills before simulation
- `--max-parallel-events N` (but obey global rate limits)

### 2) YAML config (minimum fields)
- `series_tickers: [KXHIGHMIA, ...]`
- `date_start_local: YYYY-MM-DD`
- `date_end_local: YYYY-MM-DD`
- `asof_policy`:
  - `local_time: "23:00"`
  - `days_before: 1`
- `trade_window_policy: "market_close" | "station_day_end"`
- `cadence_minutes: 1`
- `execution_model: "taker"`
- `risk`:
  - `start_balance_usd`
  - `min_ev_per_contract`
  - `fixed_qty`
  - caps...
- `output_dir`

### 3) Orchestration logic
For each series:
1) ensure catalog exists (WX-304)
2) for each event in date range:
   - ensure candlesticks exist for required window (WX-305)
3) run simulation (WX-308 + WX-309)

### 4) Concurrency & throttling
- If fetching is enabled, enforce:
  - global read rate limit
  - max concurrent requests
- Prefer fetching by event (event candlesticks) to reduce call count.

### 5) Run ID + artifacts
- Create UUID run_id at start
- Write artifacts to `backtests/<run_id>/`
- Store `artifact_path` in `backtest_run`

## Acceptance Criteria
- [ ] `--dry-run` prints a readable plan and exits 0.
- [ ] `--no-fetch` fails clearly if any required candles are missing.
- [ ] A successful run produces:
  - run_manifest.json
  - trades.csv
  - metrics.json
  - report.md
- [ ] Config validation catches missing/invalid fields with clear errors.
