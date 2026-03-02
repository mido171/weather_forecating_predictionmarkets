# 04 - Run Record (2026-03-01, Entry Gate 15:30Z, Cap 400)

This is the canonical run record for the audited MOS backtest variant.

Status:

- historical single-station baseline (`blend_00`, KNYC-only),
- retained for comparability and sanity-framework lineage,
- not the current strict co-joined reference (see run record `09`).

## 1) Script and Configuration

Script:

- `backtesting/mos_blend00_entry1530z_cap400_audit.py`

Effective parameters:

- predictions:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\03_blends\blend_00\dev_predictions.parquet`
  - `D:\Ahmed\data\kalshi\Experiments\MOS\03_blends\blend_00\test_predictions.parquet`
- market root:
  - `D:\Ahmed\data\kalshi\kalshi_history`
- filters:
  - `ev_min = 0.10`
  - `win_min = 0.60`
- bankroll:
  - `start_balance = 2700.0`
  - `risk_fraction = 0.055`
  - `stake_cap_usd = 400.0`
- entry rule:
  - first quoted row with `timestamp >= T-1 15:30:00Z`

## 2) Output Files

Core backtest outputs:

- `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\trades_all_available_blend00_ev0p1_win60_risk5p5pct_entry1530z_cap400.csv`
- `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_all_available_blend00_ev0p1_win60_risk5p5pct_entry1530z_cap400.json`
- `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\sanity_all_available_blend00_ev0p1_win60_risk5p5pct_entry1530z_cap400.json`
- `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\deep_sanity_all_available_blend00_ev0p1_win60_risk5p5pct_entry1530z_cap400.json`
- `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\market_file_integrity_knyc_selected_index.json`

Presentation table:

- `D:\Ahmed\data\kalshi\plots\june_dec_2025_trade_table_sideaware_entry1530z_cap400.csv`

## 3) Headline Metrics (Summary JSON)

- prediction date range: `2022-01-01 .. 2025-12-31`
- market overlap days: `431`
- days with no market file: `1027`
- no-candidate days: `238`
- trades: `193`
- wins/losses: `134 / 59`
- win rate: `0.6943`
- profit factor: `1.5126`
- start balance: `$2,700.00`
- final balance: `$12,614.79`
- total PnL: `$9,914.79`
- average EV at entry: `0.2134`
- median EV at entry: `0.1844`
- max drawdown: `22.94%`
- side split: `NO=184`, `YES=9`

## 4) Sanity Outcome

Primary sanity result:

- `checked_trades = 193`
- `passes_all_checks = true`
- all failure counters are zero.

Secondary deep sanity result:

- `issues_count = 0`
- summary/trade arithmetic and table consistency all pass.

Market index integrity:

- `selected_days = 492`
- missing timestamp files = `0`
- no-bucket files = `0`
- parse-failed files = `0`

## 5) Important Edge Cases Observed

1. zero-price trades:
   - count: `3`
   - these arise from side complements and are currently retained.
2. one high-impact low-price winner:
   - max profit trade occurred with entry price `0.04` and produced a large positive PnL due to high share count.
3. entry lag variability:
   - median lag after cutoff: `10` minutes,
   - maximum lag: `1236` minutes.

These are not parser errors; they are execution-data realism behaviors and should be addressed by separate policy constraints if desired.

## 6) June-Dec 2025 Table

Table rows:

- `89` (exactly matches June-Dec subset in trades CSV)

Columns:

1. Entry time (Stockholm)
2. Bucket
3. Side
4. Market win % (side)
5. Model win %
6. EV
7. Amount invested ($)
8. Profit made ($)
9. Result

## 7) Prior Parsing Error Status

Previously observed parsing defects were fixed in the audited script:

- bucket range hyphen misread as negative sign,
- non-normalized bucket label output issues.

Current artifacts contain normalized ASCII bucket labels (`XFY to YF`, `XF or below`, `XF or above`) and pass parser sanity checks.
