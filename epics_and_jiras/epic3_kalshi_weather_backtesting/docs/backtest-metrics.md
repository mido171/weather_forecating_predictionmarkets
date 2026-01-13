# Backtest Metrics & Reports (Epic #3)

This doc defines the minimum reporting outputs required for every backtest run.

## 1) Mandatory artifacts per run
Directory structure:
- `backtests/<run_id>/`
  - `run_manifest.json`
  - `config_resolved.yaml`
  - `trades.csv`
  - `positions_daily.csv`
  - `metrics.json`
  - `report.md`
  - `plots/` (optional)

## 2) Minimum metrics.json fields
### Portfolio / P&L
- `start_balance_usd`
- `end_balance_usd`
- `total_pnl_usd`
- `total_fees_usd`
- `roi_pct`
- `max_drawdown_usd`
- `max_drawdown_pct`

### Trade-level
- `num_trades`
- `win_rate_pct`
- `avg_profit_usd`
- `avg_loss_usd`
- `profit_factor` = gross_profit / gross_loss
- `avg_r_multiple` (if we define a risk per trade)
- `exposure_time_avg_minutes`

### Calibration / forecast diagnostics (optional but recommended)
- Brier score per event bucket set (if probabilities are available)
- Reliability diagram bins (stored as arrays for plotting)

## 3) report.md sections (human-readable)
- Run summary (station(s), date range, as-of policy, trade window policy)
- Key metrics table
- Equity curve summary
- Largest winners/losers (top 10)
- Diagnostics:
  - missing price candles
  - missing probability inputs
  - % of events skipped and why

## 4) Determinism requirements
- The report must record:
  - Git commit SHA (if available)
  - random seed
  - data snapshot timestamps (DB watermark)
