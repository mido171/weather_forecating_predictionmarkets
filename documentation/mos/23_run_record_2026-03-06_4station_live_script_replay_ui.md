# 23 - Run Record (2026-03-06, 4-Station `KNYC+KMIA+KMDW+KLAX` Live-Script Replay + UI)

## 1) Scope

This record documents the corrected fixed-risk 4-station cojoined replay where daily forecasts are sourced from:

- `tools/live/mos_quantile_live_inference.py`

for all four stations:

1. `KNYC`
2. `KMIA`
3. `KMDW`
4. `KLAX`

This replaces the earlier parquet-backed KLAX UI option with a true live-script replay dataset.

It also captures the KLAX bucket-parse fix for Kalshi labels using `<N°` / `>N°` style endpoints.

## 2) Locked Rule Pack

- date range: `2024-10-01` to `2025-12-31`
- stations: `KNYC`, `KMIA`, `KMDW`, `KLAX`
- runtime policy used by live script: `T-1 12:00:00Z`
- gate: `T-1 12:00:00Z`
- open delay: `entry >= market_open + 30 minutes` (per station)
- effective cutoff:
  - `effective_cutoff = max(gate_cutoff, market_open + 30m)`
- filters:
  - `EV >= 0.25`
  - `model_win_prob >= 0.85`
  - `min_market_price >= 0.25`
- bankroll:
  - `start_balance = 2700`
  - `risk_fraction = 0.075`
  - `stake_cap_usd = 700`
- execution invariant:
  - one trade max per day globally across all four stations

## 3) Entry Gate Rule (Explicit)

Per target date `T` and station:

- `gate_cutoff_utc = T-1 12:00:00Z`
- `effective_cutoff_utc = max(gate_cutoff_utc, market_open_utc + 30m)`
- eligible quote rows are those with `timestamp >= effective_cutoff_utc`

Global cojoined entry:

- walk union of all eligible station timestamps in ascending order,
- enter at first timestamp with any eligible candidate.

Tie-break at shared timestamp:

1. higher `model_win_prob`
2. higher `EV`
3. lower `market_price`
4. station alphabetical

## 4) Stake Sizing Rule (Explicit)

- `stake = min(balance_before * 0.075, 700)`

Settlement:

- hold to expiry binary payout (`YES`/`NO`) using settled `y_tmax`.

## 5) Replay Safety Note

The cojoined replay loader in:

- `backtesting/mos_blend12_knyc_kmia_cojoined_audit.py`

was hardened so cached live inference reports are accepted only if they contain inference blocks for all requested stations.

If a cached report is stale or partial:

1. the day is rerun, or
2. the replay fails closed for that day.

This prevents silently reusing older two-station replay roots for a four-station backtest.

## 6) KLAX Bucket Parse Fix

KLAX minute history files use market headers like:

1. `Will the **high temp in LA** be <57° on Apr 26, 2025?`
2. `Will the **high temp in LA** be >64° on Apr 26, 2025?`

The original parser handled:

1. `X-Y`
2. `X or below`
3. `X or above`

but did not correctly handle symbolic `<` / `>` prefixes when the question text also contained a calendar date.

Bad prior behavior example:

1. `>64° on Apr 26, 2025` was misparsed as `26F to 64F`
2. `>75° on Jun 23, 2025` was misparsed as `23F to 75F`

Corrected behavior:

1. `<57°` -> `56F or below`
2. `>64°` -> `65F or above`
3. `>75°` -> `76F or above`

This fix was applied in:

1. `backtesting/mos_blend12_knyc_kmia_cojoined_audit.py`
2. `tools/live/mos_quantile_live_inference.py`

## 7) Replay Command

```powershell
python backtesting/mos_blend12_knyc_kmia_cojoined_audit.py `
  --mode cojoined `
  --stations KNYC,KMIA,KMDW,KLAX `
  --prediction-source live-script `
  --start-date 2024-10-01 `
  --end-date 2025-12-31 `
  --entry-hour-z 12 `
  --entry-minute-z 0 `
  --min-entry-minutes-after-open 30 `
  --ev-min 0.25 `
  --win-min 0.85 `
  --min-market-price 0.25 `
  --start-balance 2700 `
  --risk-fraction 0.075 `
  --stake-cap-usd 700 `
  --live-inference-root "D:\Ahmed\data\live\mos_quantile_live_inference\backtest_replay_knyc_kmia_kmdw_klax_2024_2025" `
  --out-dir "D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest" `
  --out-prefix "cojoined_blend12_knyc_kmia_kmdw_klax_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_live_script_2024_2025" `
  --table-out "D:\Ahmed\data\kalshi\plots\cojoined_blend12_knyc_kmia_kmdw_klax_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_live_script_2024_2025_stockholm.csv"
```

## 8) Replay Root

Per-day live inference replay cache:

- `D:\Ahmed\data\live\mos_quantile_live_inference\backtest_replay_knyc_kmia_kmdw_klax_2024_2025`

Observed coverage:

1. `457` target-day folders (`2024-10-01` through `2025-12-31`)
2. one runtime-gate failure day:
   - `2025-05-30`

## 9) Output Artifacts

1. trades:
   - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\trades_cojoined_blend12_knyc_kmia_kmdw_klax_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_live_script_2024_2025.csv`
2. summary:
   - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_cojoined_blend12_knyc_kmia_kmdw_klax_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_live_script_2024_2025.json`
3. sanity:
   - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\sanity_cojoined_blend12_knyc_kmia_kmdw_klax_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_live_script_2024_2025.json`
4. day debug:
   - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\day_debug_cojoined_blend12_knyc_kmia_kmdw_klax_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_live_script_2024_2025.json`
5. side-aware table (with balance):
   - `D:\Ahmed\data\kalshi\plots\all_trades_sideaware_cojoined_blend12_knyc_kmia_kmdw_klax_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_live_script_2024_2025_with_balance.csv`
6. stockholm display table:
   - `D:\Ahmed\data\kalshi\plots\cojoined_blend12_knyc_kmia_kmdw_klax_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_live_script_2024_2025_stockholm.csv`

## 10) Summary Snapshot

From summary JSON:

1. `total_days = 457`
2. `days_with_any_prediction = 456`
3. `days_with_any_market_file = 434`
4. `days_with_any_station_context = 433`
5. `days_without_trade_candidate = 59`
6. `trades = 398`
7. `wins/losses = 308 / 90`
8. `win_rate = 0.7738693467`
9. `profit_factor = 2.0933979432`
10. `final_balance = 68909.31491704485`
11. `max_drawdown = 0.1541385484`
12. `station_counts = {KMDW: 135, KMIA: 100, KNYC: 94, KLAX: 69}`

Per-station prediction availability:

1. `KNYC = 456`
2. `KMIA = 456`
3. `KMDW = 454`
4. `KLAX = 456`

## 11) Availability Notes

### 10.1 Runtime-gate failure day

One live inference day failed exactly as designed, with no fallback used:

1. target date: `2025-05-30`
2. reason:
   - required `gfs_12` runtime `2025-05-29T12:00:00Z` was not available for all four stations
3. affected stations:
   - `KNYC`
   - `KMIA`
   - `KMDW`
   - `KLAX`

This day is recorded in:

- `summary.prediction_source_meta.live_loader_stats.failed_live_inference_days`

### 10.2 KMDW truth gaps

`KMDW` had two dates absent from the truth CSV during this window:

1. `2025-08-30`
2. `2025-08-31`

Effect:

1. live report existed,
2. but no KMDW prediction row was admitted into replay trade generation for those two dates.

## 12) Sanity Snapshot

From sanity JSON:

1. `passes_all_checks = true`
2. `checked_trades = 398`
3. all tracked failure counters are `0`

Critical counters:

1. `entry_before_effective_cutoff = 0`
2. `entry_not_first_eligible_timestamp_globally = 0`
3. `tie_break_policy_violation = 0`
4. `market_price_mismatch = 0`
5. `model_prob_mismatch = 0`
6. `ev_mismatch = 0`
7. `pnl_mismatch = 0`

## 13) UI Wiring

The main backtesting `2024-2025` UI tab now points to this live-script replay dataset.

Updated file:

- `ui/result_viewer/src/App.jsx`

UI option:

1. key: `2024-2025`
2. label: `2024-2025`
3. station set shown in params:
   - `KNYC + KMIA + KMDW + KLAX`

UI-served CSV:

- `D:\Ahmed\git\weather\weather_forecating_predictionmarkets\ui\result_viewer\public\data\all_trades_sideaware_cojoined_blend12_knyc_kmia_kmdw_klax_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_live_script_2024_2025_with_balance.csv`

Hash check:

1. UI CSV is byte-identical to the plot CSV copy.

## 14) Claim Anchors For This Run

Any performance claim for this run must include:

1. entry gate rule:
   - `entry_timestamp >= max(T-1 12:00:00Z, market_open_utc + 30m)`
2. stake rule:
   - `stake = min(balance_before * 0.075, 700)`
3. summary JSON path:
   - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_cojoined_blend12_knyc_kmia_kmdw_klax_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_live_script_2024_2025.json`
4. sanity JSON path:
   - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\sanity_cojoined_blend12_knyc_kmia_kmdw_klax_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_live_script_2024_2025.json`
