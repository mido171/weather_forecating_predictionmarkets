# 03 - Sanity Audit Framework

This document defines audit gates for MOS outputs.

A run is considered valid only if its sanity artifacts pass all required checks.

## 1) Audit Layers

There are three practical layers in this repo:

1. primary script-level sanity (`sanity_*.json`),
2. optional deep/post sanity (for enriched variants),
3. market-file integrity sanity (schema/parse-level checks).

## 2) Primary Audit (Single-Station)

Primary single-station audit is produced by:

- `backtesting/mos_blend00_entry1530z_cap400_audit.py`

Key checks include:

1. market file exists for selected trade row,
2. target-date to file-date alignment,
3. entry is not before configured gate,
4. entry is first valid quote at/after gate,
5. bucket column exists and parses successfully,
6. side-aware market price reconciliation,
7. model bucket probability reconciliation,
8. EV reconciliation,
9. win/loss label reconciliation vs `y_tmax`,
10. stake-cap compliance,
11. PnL arithmetic consistency,
12. finite numeric values in critical fields.

Representative output:

- `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\sanity_all_available_blend00_ev0p1_win60_risk5p5pct_entry1530z_cap400.json`

## 3) Primary Audit (Co-Joined Multi-Station)

Primary co-joined audit is produced by:

- `backtesting/mos_blend12_knyc_kmia_cojoined_audit.py`

Core co-joined failure counters:

- `more_than_one_trade_per_day_global`
- `entry_before_gate`
- `entry_before_effective_cutoff`
- `entry_not_first_eligible_timestamp_globally`
- `tie_break_policy_violation`
- `market_file_missing`
- `bucket_not_found`
- `bucket_unparseable`
- `market_price_mismatch`
- `model_prob_mismatch`
- `ev_mismatch`
- `win_label_mismatch`
- `stake_cap_breach`
- `pnl_mismatch`

Representative outputs:

- base open+30m co-joined:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\sanity_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p18_win67_risk6p5_cap500_base.json`
- strict open+30m post-processed:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\sanity_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p18_win67_fractionalkelly0p15_cap500_no_outlier_gt2000_recalc.json`
- fixed-risk 3-station (`KNYC` + `KMIA` + `KMDW`):
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\sanity_cojoined_blend12_knyc_kmia_kmdw_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_2024_2025.json`

## 4) Day-Level Debug Audit (Co-Joined)

Co-joined runs also emit day-level debug JSON:

- `day_debug_*.json`

This is the forensic layer for:

- per-station market file presence,
- per-station market open and effective cutoff,
- first eligible timestamp for the day,
- chosen trade key.

Representative file:

- `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\day_debug_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p18_win67_risk6p5_cap500_base.json`
- `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\day_debug_cojoined_blend12_knyc_kmia_kmdw_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_2024_2025.json`

## 5) Deep/Post-Processing Audit

For post-processed variants (fractional Kelly + outlier filtered recalc), additional checks are expected:

- no remaining `pnl > threshold` rows,
- risk fraction remains within expected bounds,
- recalculated bankroll path is internally consistent,
- stake cap remains respected after recalc.

In current strict artifacts this is represented in sanity counters such as:

- `remaining_outlier_gt_2000`
- `risk_fraction_out_of_bounds`
- `stake_cap_breach`

## 6) Acceptance Criteria

A run passes only if all conditions hold:

1. `passes_all_checks = true` in sanity JSON.
2. every failure counter is `0`.
3. summary-level counts are consistent with trade table row counts.
4. one-trade/day invariant holds (co-joined).
5. effective cutoff rule is respected when open-delay is configured.

## 7) Known Edge Cases That Can Still Pass

Sanity pass does not imply microstructure realism. Examples:

- side price equals `0` (often from complement side when YES=1),
- extremely low price producing large share count and high PnL.

These are arithmetic-consistent in current logic, but may require separate execution policy filters depending on deployment standards.
