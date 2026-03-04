# 16 - Run Record (2026-03-04, Co-Joined blend_12 `KNYC+KMIA+KMDW`, Open+30m, Fixed-Risk 7.5%, 2024-2025 Window)

## 1) Scope of This Run

This record captures the first audited 3-station co-joined run using:

1. `KNYC`
2. `KMIA`
3. `KMDW`

Window:

- `2024-10-01` through `2025-12-31`

Prediction source used for this run:

- `parquet` (not live-script replay)

## 2) Locked Rule Pack (Explicit)

Entry gate rule:

- `entry_timestamp >= max(T-1 12:00:00Z, market_open_utc + 30m)`

Stake sizing rule:

- `stake = min(balance_before * 0.075, 700)`

Selection filters:

1. `EV >= 0.25`
2. `model_win_prob >= 0.85`
3. `side_market_price >= 0.25`

Co-joined invariant:

- max one trade/day globally across configured stations

## 3) Backtest Command (Executed)

```powershell
python backtesting/mos_blend12_knyc_kmia_cojoined_audit.py `
  --mode cojoined `
  --stations KNYC,KMIA,KMDW `
  --start-date 2024-10-01 `
  --end-date 2025-12-31 `
  --entry-hour-z 12 `
  --entry-minute-z 0 `
  --min-entry-minutes-after-open 30 `
  --ev-min 0.25 `
  --win-min 0.85 `
  --min-market-price 0.25 `
  --risk-fraction 0.075 `
  --stake-cap-usd 700 `
  --out-prefix cojoined_blend12_knyc_kmia_kmdw_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_2024_2025 `
  --table-out D:\Ahmed\data\kalshi\plots\cojoined_blend12_knyc_kmia_kmdw_stockholm_table_2024_2025.csv
```

## 4) Output Artifacts

Primary run artifacts:

1. trades:
   - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\trades_cojoined_blend12_knyc_kmia_kmdw_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_2024_2025.csv`
2. summary:
   - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_cojoined_blend12_knyc_kmia_kmdw_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_2024_2025.json`
3. sanity:
   - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\sanity_cojoined_blend12_knyc_kmia_kmdw_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_2024_2025.json`
4. day debug:
   - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\day_debug_cojoined_blend12_knyc_kmia_kmdw_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_2024_2025.json`
5. side-aware with balance:
   - `D:\Ahmed\data\kalshi\plots\all_trades_sideaware_cojoined_blend12_knyc_kmia_kmdw_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_2024_2025_with_balance.csv`
6. stockholm table:
   - `D:\Ahmed\data\kalshi\plots\cojoined_blend12_knyc_kmia_kmdw_stockholm_table_2024_2025.csv`

UI-populated copy:

- `D:\Ahmed\git\weather\weather_forecating_predictionmarkets\ui\result_viewer\public\data\all_trades_sideaware_cojoined_blend12_knyc_kmia_kmdw_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_2024_2025_with_balance.csv`

## 5) Summary Snapshot

From summary JSON:

1. `total_days`: `457`
2. `days_with_any_market_file`: `434`
3. `trades`: `374`
4. `wins/losses`: `289 / 85`
5. `win_rate`: `0.7727272727`
6. `profit_factor`: `2.0214710776`
7. `start_balance`: `2700.00`
8. `final_balance`: `61520.0559508216`
9. `total_pnl`: `58820.0559508216`
10. `max_drawdown`: `0.2492504058`
11. `station_counts`: `{KMDW: 144, KNYC: 116, KMIA: 114}`
12. `side_counts`: `{NO: 373, YES: 1}`

## 6) Sanity Snapshot

From sanity JSON:

1. `passes_all_checks = true`
2. `checked_trades = 374`
3. all failure counters are `0`, including:
   - `entry_before_gate`
   - `entry_before_effective_cutoff`
   - `entry_not_first_eligible_timestamp_globally`
   - `tie_break_policy_violation`
   - `stake_cap_breach`
   - `pnl_mismatch`

## 7) Comparison Run (KNYC+KMIA only, Same Rule Pack)

Comparison command (executed):

```powershell
python backtesting/mos_blend12_knyc_kmia_cojoined_audit.py `
  --mode cojoined `
  --stations KNYC,KMIA `
  --start-date 2024-10-01 `
  --end-date 2025-12-31 `
  --entry-hour-z 12 `
  --entry-minute-z 0 `
  --min-entry-minutes-after-open 30 `
  --ev-min 0.25 `
  --win-min 0.85 `
  --min-market-price 0.25 `
  --risk-fraction 0.075 `
  --stake-cap-usd 700 `
  --out-prefix cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_2024_2025_recheck `
  --table-out D:\Ahmed\data\kalshi\plots\cojoined_blend12_knyc_kmia_stockholm_table_2024_2025_recheck.csv
```

Comparison summary path:

- `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_2024_2025_recheck.json`

Comparison sanity path:

- `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\sanity_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_2024_2025_recheck.json`

Delta (`KNYC+KMIA+KMDW` minus `KNYC+KMIA`):

1. trades: `+61` (`374` vs `313`)
2. wins: `+41`
3. losses: `+20`
4. win rate: `-1.9605pp` (`77.27%` vs `79.23%`)
5. total pnl: `+$7,832.26`
6. final balance: `+$7,832.26`
7. profit factor: `-0.1633`
8. max drawdown: `+9.6742pp`

## 8) Prediction Source Clarification

Both comparison runs above used:

- `prediction_source_meta.prediction_source = "parquet"`

No per-day live inference script replay was used in these specific runs.

## 9) Timezone + Open+30m Spot Check (KMDW Trade)

Example row (side-aware table):

1. `target_date_local = 2025-12-17`
2. `entry_time_stockholm = 2025-12-16 17:10:00 CET`
3. `market_open_utc = 2025-12-16T15:08:00Z` (`16:08 CET`)
4. `gate_cutoff_utc = 2025-12-16T12:00:00Z` (`13:00 CET`)
5. `effective_cutoff_utc = 2025-12-16T15:38:00Z` (`16:38 CET`)

Result:

- entry at `17:10 CET` occurs `+32 minutes` after effective cutoff (`16:38 CET`), therefore this row is compliant with the open+30m gate.

## 10) Implementation Hooks Updated for This Run Family

Script:

- `backtesting/mos_blend12_knyc_kmia_cojoined_audit.py`

Key additions:

1. generic station parsing via `--stations`
2. `single|cojoined` mode switch
3. JSON mapping inputs for per-station paths/prefixes
4. legacy defaults retained for `KNYC/KMIA`
5. `KMDW` defaults added

## 11) Reproducibility Notes

1. All run identifiers are encoded in output prefixes.
2. Rule parameters are embedded in output filenames and summary JSON fields.
3. Summary/sanity/day_debug artifacts are all present and cross-consistent.
4. This run is a fixed-risk co-joined reference, distinct from live-script replay references in docs `11` and `12`.
