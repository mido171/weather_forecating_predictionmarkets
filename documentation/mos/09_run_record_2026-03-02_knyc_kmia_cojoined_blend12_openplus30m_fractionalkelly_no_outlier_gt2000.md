# 09 - Run Record (2026-03-02, KNYC + KMIA Co-Joined blend_12, Open+30m, Fractional Kelly, Outlier-Filtered Recalc)

This is the current strict reference run record for co-joined MOS execution.

It adds explicit open-delay protection and keeps the strict filtering/risk sensitivity workflow.

## 1) Locked Rule Pack

- date range: `2024-10-01` to `2025-12-31`
- stations: `KNYC`, `KMIA`
- model per station: `blend_12`
- gate: `T-1 12:00:00Z`
- open delay: `entry >= market_open + 30 minutes` (per station)
- effective cutoff:
  - `effective_cutoff = max(gate_cutoff, market_open + 30m)`
- entry policy:
  - walk union of both station quote timestamps,
  - enter at first global timestamp with any eligible candidate
- filters:
  - `EV >= 0.18`
  - `model_win_prob >= 0.67`
- global invariant:
  - at most one trade/day across both stations

Tie-break at shared timestamp:

1. higher `model_win_prob`
2. higher `EV`
3. lower `market_price`
4. station alphabetical

## 2) Input Artifacts

Predictions:

- KNYC:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\03_blends\blend_12\dev_predictions.parquet`
  - `D:\Ahmed\data\kalshi\Experiments\MOS\03_blends\blend_12\test_predictions.parquet`
- KMIA:
  - `D:\Ahmed\data\kalshi\Experiments\MOS_KMIA\03_blends\blend_12\dev_predictions.parquet`
  - `D:\Ahmed\data\kalshi\Experiments\MOS_KMIA\03_blends\blend_12\test_predictions.parquet`

Markets:

- `D:\Ahmed\data\kalshi\kalshi_history\kxhighny_2024_10_01_to_2025_12_31`
- `D:\Ahmed\data\kalshi\kalshi_history\kxhighmia_2024_10_01_to_2025_12_31`

## 3) Base Stream Command (Fixed Risk, Open+30m)

```powershell
python backtesting/mos_blend12_knyc_kmia_cojoined_audit.py --start-date 2024-10-01 --end-date 2025-12-31 --entry-hour-z 12 --entry-minute-z 0 --min-entry-minutes-after-open 30 --ev-min 0.18 --win-min 0.67 --start-balance 2700 --risk-fraction 0.065 --stake-cap-usd 500 --out-dir "D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest" --out-prefix "cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p18_win67_risk6p5_cap500_base" --table-out "D:\Ahmed\data\kalshi\plots\cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p18_win67_risk6p5_cap500_base_stockholm.csv"
```

## 4) Base Stream Outputs and Metrics

Outputs:

- trades:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\trades_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p18_win67_risk6p5_cap500_base.csv`
- summary:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p18_win67_risk6p5_cap500_base.json`
- sanity:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\sanity_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p18_win67_risk6p5_cap500_base.json`
- day-debug:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\day_debug_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p18_win67_risk6p5_cap500_base.json`
- side-aware table:
  - `D:\Ahmed\data\kalshi\plots\all_trades_sideaware_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p18_win67_risk6p5_cap500_base_with_balance.csv`
- Stockholm table:
  - `D:\Ahmed\data\kalshi\plots\cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p18_win67_risk6p5_cap500_base_stockholm.csv`

Headline metrics (base stream):

- trades: `434`
- wins/losses: `319 / 115`
- win rate: `73.50%`
- profit factor: `1.7725`
- final balance: `$45,382.25`
- total pnl: `$42,682.25`
- max drawdown: `31.07%`
- station counts:
  - `KNYC: 253`
  - `KMIA: 181`

Sanity:

- `passes_all_checks = true`
- critical counters:
  - `entry_before_effective_cutoff = 0`
  - `entry_not_first_eligible_timestamp_globally = 0`
  - `tie_break_policy_violation = 0`

## 5) Strict Post-Processed Variant

A deterministic post-processing transform is applied to the base stream:

1. switch sizing to fractional Kelly:
   - `full_kelly = (q - p) / (1 - p)` (clamped to `[0,1]`)
   - `risk_fraction_used = 0.15 * full_kelly`
   - `stake = min(balance_before * risk_fraction_used, 500)`
2. remove trades where `pnl > 2000`
3. recompute full bankroll path sequentially on remaining trades

This is a risk-sensitivity/reporting transform and must be labeled as post-processed.

## 6) Strict Outputs and Metrics (Current Reference)

Outputs:

- trades:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\trades_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p18_win67_fractionalkelly0p15_cap500_no_outlier_gt2000_recalc.csv`
- summary:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p18_win67_fractionalkelly0p15_cap500_no_outlier_gt2000_recalc.json`
- sanity:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\sanity_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p18_win67_fractionalkelly0p15_cap500_no_outlier_gt2000_recalc.json`
- removed-trades log:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\removed_trades_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p18_win67_fractionalkelly0p15_cap500_no_outlier_gt2000_recalc.csv`
- side-aware table:
  - `D:\Ahmed\data\kalshi\plots\all_trades_sideaware_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p18_win67_fractionalkelly0p15_cap500_no_outlier_gt2000_recalc_with_balance.csv`
- Stockholm table:
  - `D:\Ahmed\data\kalshi\plots\cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p18_win67_fractionalkelly0p15_cap500_no_outlier_gt2000_recalc_stockholm.csv`

Headline metrics (strict post-processed):

- removed outliers: `2`
- trades: `432`
- wins/losses: `317 / 115`
- win rate: `73.38%`
- profit factor: `1.7172`
- final balance: `$40,984.41`
- total pnl: `$38,284.41`
- max drawdown: `45.31%`
- station counts:
  - `KNYC: 252`
  - `KMIA: 180`

Sanity (strict post-processed):

- `passes_all_checks = true`
- `remaining_outlier_gt_2000 = 0`
- `entry_before_effective_cutoff = 0`
- `stake_cap_breach = 0`

## 7) Leakage-Safety Verdict

For this run family, leakage-sensitive conditions are satisfied:

1. entry eligibility uses only quotes at/after effective cutoff,
2. effective cutoff is computed from gate and observed market-open timestamp,
3. one-trade/day global invariant is enforced,
4. entry timestamp equals first eligible global timestamp,
5. sanity counters report zero critical failures.

## 8) UI Consumption Path

The current UI is expected to read this strict table:

- `D:\Ahmed\data\kalshi\plots\all_trades_sideaware_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p18_win67_fractionalkelly0p15_cap500_no_outlier_gt2000_recalc_with_balance.csv`

## 9) Reproducibility Checklist

When citing this run, always include:

1. gate (`T-1 12:00Z`)
2. open-delay (`+30m`)
3. filters (`EV >= 0.18`, `win >= 0.67`)
4. sizing mode (fixed-risk base vs fractional-Kelly post-processed)
5. outlier threshold (`pnl > 2000` removal)
6. summary JSON path
7. sanity JSON path
