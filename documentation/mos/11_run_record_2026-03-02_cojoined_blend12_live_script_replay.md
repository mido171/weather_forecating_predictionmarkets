# 11 - Run Record (2026-03-02, Co-Joined blend_12 Replay Using Live Inference Script)

This record documents the replay backtest where co-joined KNYC/KMIA forecasts are generated per day from:

- `tools/live/mos_quantile_live_inference.py`

instead of reading precomputed prediction parquet files.

## 1) Locked Rule Pack

- date range: `2024-10-01` to `2025-12-31`
- stations: `KNYC`, `KMIA`
- runtime policy used by live script: `T-1 12:00:00Z`
- gate: `T-1 12:00:00Z`
- open delay: `entry >= market_open + 30 minutes` (per station)
- effective cutoff:
  - `effective_cutoff = max(gate_cutoff, market_open + 30m)`
- filters:
  - `EV >= 0.30`
  - `model_win_prob >= 0.75`
  - `min_market_price >= 0.10`
- bankroll:
  - `start_balance = 2700`
  - `risk_fraction = 0.06`
  - `stake_cap_usd = 500`
- execution invariant:
  - one trade max per day globally across both stations

## 2) Entry Gate Rule (Explicit)

Per target date `T` and station:

- `gate_cutoff_utc = T-1 12:00:00Z`
- `effective_cutoff_utc = max(gate_cutoff_utc, market_open_utc + 30m)`
- eligible quote rows are those with `timestamp >= effective_cutoff_utc`

Global co-joined entry:

- walk union of KNYC/KMIA eligible timestamps in ascending order,
- enter at first timestamp with any eligible candidate.

Tie-break at shared timestamp:

1. higher `model_win_prob`
2. higher `EV`
3. lower `market_price`
4. station alphabetical

## 3) Stake Sizing Rule (Explicit)

- `stake = min(balance_before * 0.06, 500)`

Settlement:

- hold to expiry binary payout (`YES`/`NO`) using settled `y_tmax`.

## 4) Replay Command

```powershell
python backtesting/mos_blend12_knyc_kmia_cojoined_audit.py --prediction-source live-script --start-date 2024-10-01 --end-date 2025-12-31 --entry-hour-z 12 --entry-minute-z 0 --min-entry-minutes-after-open 30 --ev-min 0.30 --win-min 0.75 --min-market-price 0.10 --start-balance 2700 --risk-fraction 0.06 --stake-cap-usd 500 --out-dir "D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest" --out-prefix "cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script" --table-out "D:\Ahmed\data\kalshi\plots\cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script_stockholm.csv"
```

## 5) Output Artifacts

- trades:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\trades_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script.csv`
- summary:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script.json`
- sanity:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\sanity_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script.json`
- day debug:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\day_debug_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script.json`
- side-aware table (with balance):
  - `D:\Ahmed\data\kalshi\plots\all_trades_sideaware_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script_with_balance.csv`
- stockholm display table:
  - `D:\Ahmed\data\kalshi\plots\cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script_stockholm.csv`

## 6) Summary Snapshot

From summary JSON:

- `total_days = 457`
- `days_with_any_prediction = 456`
- `trades = 387`
- `wins = 257`
- `losses = 130`
- `win_rate = 0.6641`
- `profit_factor = 1.8019`
- `final_balance = 52254.17`
- `max_drawdown = 0.3101`

## 7) Data Availability Note

Replay uses strict live-script runtime gating (`T-1 12:00Z`, no fallback).

One day failed live inference due missing required GFS runtime in source payload:

- target date: `2025-05-30`
- effect: treated as no prediction for that day
- failure details are captured in:
  - `summary.prediction_source_meta.live_loader_stats.failed_live_inference_days`

## 8) Claim Anchors For This Run

Any performance claim for this run must include:

1. entry gate rule:
   - `entry_timestamp >= max(T-1 12:00:00Z, market_open_utc + 30m)`
2. stake rule:
   - `stake = min(balance_before * 0.06, 500)`
3. summary JSON path:
   - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script.json`
4. sanity JSON path:
   - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\sanity_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script.json`
