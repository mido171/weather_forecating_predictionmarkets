# WX-306 — Fee model implementation (Kalshi quadratic fees) + optional fee-change ingestion

## Objective
Implement the fee calculation used by the backtest engine so that trade EV and realized P&L incorporate Kalshi fees.

Optionally ingest series-specific fee metadata/changes to support future-proofing.

## References
- Kalshi fee schedule (published): https://kalshi.com/fee-schedule
- Kalshi API: series includes `fee_type` / `fee_multiplier`
- Kalshi API: `GET /exchange/series_fee_changes?series_ticker=...&show_historical=true`

## Requirements

### 1) Fee calculation utility (required)
Implement a pure function:
- `fee_usd = calc_fee_usd(fee_policy, side, qty, price_usd, is_maker=False)`

Default policy (taker):
- `fee = ceil( 0.07 × qty × P × (1 − P) )`

If maker:
- `fee = ceil( 0.0175 × qty × P × (1 − P) )`

Where:
- qty = number of contracts
- P is the executed price in dollars (0..1)
- ceil rounds up to the nearest cent (strictly conservative).

### 2) Persist fee assumptions per backtest run (required)
In `backtest_run.metrics_json` and/or `run_manifest.json`, store:
- fee coefficients used
- rounding mode used (ceiling cents)
- whether maker simulation is enabled (default false)

### 3) Optional: ingest fee changes (recommended)
If time permits, implement a small ingest job:
- calls `GET /exchange/series_fee_changes` for each series ticker
- stores a change timeline in a new table:
  - `kalshi_series_fee_change(series_ticker, scheduled_ts_utc, fee_type, fee_multiplier, retrieved_at_utc, raw_json)`
- backtest can then pick the effective fee policy at decision time.

If fee_multiplier semantics are unclear from docs, store it but do not apply until verified.

## Acceptance Criteria
- [ ] Unit tests cover:
  - P=0.5, qty=100 (max fee case)
  - P close to 0 or 1 (near-zero fee)
  - maker vs taker coefficients
  - strict ceiling behavior
- [ ] Backtest run manifest always includes fee settings.
- [ ] (If optional ingestion is implemented) fee changes are persisted idempotently with a unique key.
