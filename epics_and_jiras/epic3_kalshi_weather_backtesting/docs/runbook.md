# Runbook (Epic #3)

## 0) Prerequisites
- Epic #1 data ingestion is running and populated (stations, CLI, MOS)
- Epic #2 training has produced persisted model artifacts (μ̂ and σ̂ models)

## 1) Environment variables
- `KALSHI_ENV`: `prod` | `demo`
- `KALSHI_API_KEY_ID`: optional for authenticated requests
- `KALSHI_PRIVATE_KEY_PATH`: optional for authenticated requests
- `MYSQL_URL`, `MYSQL_USER`, `MYSQL_PASSWORD`
- `BACKTEST_OUTPUT_DIR`: default `backtests/`

## 2) Configure what to backtest
Edit `backtest/configs/backtest.yaml`:
- series tickers (e.g., `KXHIGHMIA`)
- stationId mapping (MIA → KMIA, etc) or read from DB station registry
- date range
- as-of policy
- trade window policy
- risk constraints:
  - max contracts per event
  - max exposure per station
  - EV threshold

## 3) Data backfill (recommended)
Run:
- market catalog sync
- candlestick backfill

Then run the backtest without hitting the API.

## 4) Backtest run
Run the backtest runner:
- produces `run_manifest.json`, `trades.csv`, `metrics.json`, `report.md`

## 5) Common failure modes
- Missing candlesticks:
  - backfill for the missing window
- Event date mapping failures:
  - inspect `kalshi_event_date_mapping_errors` table/logs
- Rate limiting:
  - throttle, add API key, reduce concurrency
- Data leakage:
  - ensure decision time is respected (asOfUtc) and no future MOS runs are used
