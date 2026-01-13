# WX-311 — Backtest reporting: metrics.json + report.md + trade/position exports

## Objective
Implement the reporting layer that transforms raw simulated trades + portfolio state into:
- `metrics.json` (machine-readable)
- `report.md` (human-readable)
- `trades.csv` and `positions_daily.csv`

Per `docs/backtest-metrics.md`.

## Requirements

### 1) Data inputs
- trades list (from engine)
- daily portfolio snapshots (cash, exposure, pnl)
- config snapshot
- run_id + metadata

### 2) Required outputs
Write to `backtests/<run_id>/`:
- `metrics.json` with required fields
- `report.md` with required sections
- `trades.csv`
- `positions_daily.csv`
- optional plots:
  - equity curve
  - distribution of EV at entry
  - histogram of P&L per trade

### 3) Metrics calculations (minimum)
- start/end balance
- total pnl
- total fees
- ROI%
- win rate
- avg profit, avg loss
- profit factor
- max drawdown (absolute and %)

### 4) Diagnostics section (must exist)
Report must include:
- # events attempted
- # events skipped
- skip reasons counts:
  - missing probabilities
  - missing quotes
  - event date mapping failure
- % candles missing inside trade windows

## Acceptance Criteria
- [ ] A sample run generates all required artifacts.
- [ ] Profit factor and win rate match hand-calculated values for a small synthetic test case.
- [ ] Report includes a diagnostics section with skip counts.
