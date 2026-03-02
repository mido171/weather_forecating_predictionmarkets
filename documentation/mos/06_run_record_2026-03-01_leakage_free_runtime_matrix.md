# 06 - Run Record (2026-03-01, Leakage-Free Runtime Matrix: blend_00 vs blend_12)

This record documents the leakage-free runtime matrix that compares `blend_00` and `blend_12` under multiple entry gates.

Status:

- historical matrix benchmark for timing/freshness analysis,
- complements run records `07`, `08`, and `09`,
- current strict co-joined deployment reference remains run record `09`.

The objective is to answer:

1. Which blend/timing combination performs best?
2. Is the comparison leakage-free and live-reproducible?
3. How much of performance is associated with forecast freshness vs quote regime?

## 1) Fixed Policy Used For All Matrix Runs

- filters:
  - `ev_min = 0.20`
  - `win_min = 0.65`
- bankroll:
  - `start_balance = 2700.0`
  - `risk_fraction = 0.055`
  - `stake_cap_usd = 400.0`
- settlement:
  - hold-to-expiry binary payout
- market root:
  - `D:\Ahmed\data\kalshi\kalshi_history`

Risk rule formula used in all matrix runs:

- `stake_usd = min(balance_before * 0.055, 400.0)`
- share sizing is then computed from side price at entry quote.

Concrete best filtered setup (current reference from this matrix):

- forecast model: `blend_12`
- forecast runtime used: `T-1 12:00:00Z`
- entry gate: `T-1 12:00:00Z`
- actual entry rule: first quote with `timestamp >= gate`
- trading filters: `EV >= 0.20` and `model_win_prob >= 0.65`
- bankroll sizing: `risk_fraction = 5.5%` with `stake_cap_usd = 400`

## 2) Leakage-Free Entry Gates Tested

`blend_12` runtime availability:

- forecast runtime for target day `T` is `T-1 12:00:00Z`

Tested gates for `blend_12`:

- `T-1 12:00Z`
- `T-1 15:30Z`
- `T-1 20:00Z`
- `T 00:00Z`
- `T 04:00Z`
- `T 08:00Z`
- `T 12:00Z`
- `T 16:00Z`

`blend_00` runtime availability:

- forecast runtime for target day `T` is `T 00:00:00Z`

Tested gates for `blend_00`:

- `T 00:00Z`
- `T 04:00Z`
- `T 08:00Z`
- `T 12:00Z`
- `T 16:00Z`

Entry rule in all cases:

- first quoted market row with `timestamp >= gate`

Important:

- gate means "start scanning from this timestamp".
- actual entry can be later if first quote appears later.
- example: gate `T-1 12:00Z` with first available quote at `T-1 15:01Z` means entry is `15:01Z`, not `12:00Z`.

## 3) Core Matrix Outputs

Files:

- `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\matrix_blend00_blend12_ev0p20_win65_risk5p5_cap400_comparison.csv`
- `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\matrix_blend00_blend12_overlap_entrytimes_finalbalance.csv`
- `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\matrix_blend00_blend12_freshness_vs_performance.csv`

## 4) Leakage Sanity Result

For all 13 matrix runs:

- `entry_before_runtime_count = 0`

Meaning:

- no trade was entered before forecast runtime availability.
- matrix is runtime-aligned and leakage-free on timing.

## 5) Raw Best Combo (No Extra Outlier Filter)

Best by final balance:

- combo: `blend_12 @ T-1 12:00Z`
- trades: `93`
- win/loss: `70 / 23`
- win rate: `0.7527`
- profit factor: `4.6493`
- final balance: `$34,973.46`
- total pnl: `$32,273.46`
- max drawdown: `8.62%`

Summary file:

- `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_matrix_blend_12_ev0p20_win65_risk5p5_cap400_tminus1_1200z.json`

## 6) Outlier-Capped Recalculation (Requested Post-Run)

Post-processing rule:

- remove any trade where `pnl > 3000`
- recompute stake/pnl/balance path sequentially on remaining trades

Files:

- `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\matrix_blend00_blend12_ev0p20_win65_risk5p5_cap400_pnlcap3000removed_recalc_comparison.csv`
- `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\matrix_blend00_blend12_ev0p20_win65_risk5p5_cap400_pnlcap3000removed_recalc_overlap.csv`

Best after this filter remains:

- combo: `blend_12 @ T-1 12:00Z`
- trades: `91` (2 outlier trades removed)
- win/loss: `68 / 23`
- win rate: `0.7473`
- profit factor: `2.9371`
- final balance: `$18,953.84`
- total pnl: `$16,253.84`
- max drawdown: `24.60%`

Best filtered output files:

- `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\trades_matrix_blend_12_ev0p20_win65_risk5p5_cap400_tminus1_1200z_pnlcap3000removed_recalc.csv`
- `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_matrix_blend_12_ev0p20_win65_risk5p5_cap400_tminus1_1200z_pnlcap3000removed_recalc.json`
- `D:\Ahmed\data\kalshi\plots\all_trades_sideaware_blend12_tminus1_1200z_ev0p20_win65_risk5p5_cap400_pnlcap3000removed_recalc_with_balance.csv`

## 7) Forecast Freshness Findings

Freshness defined as:

- `entry_timestamp_utc - forecast_runtime_utc` in hours

Selected rows:

- `blend_12 @ T-1 12:00Z`:
  - freshness median: `3.0167h`
  - final balance: `$34,973.46` (raw) / `$18,953.84` (outlier-capped recalculation)
- `blend_12 @ T-1 15:30Z`:
  - freshness median: `3.9083h`
  - final balance: `$5,229.43`
- `blend_00 @ T 00:00Z`:
  - freshness median: `0.0500h`
  - final balance: `$2,995.12`

Interpretation:

- freshness is one contributor, but not sufficient to explain pnl ranking alone.
- quote regime and fill-price distribution materially affect outcomes.

## 8) Timezone Clarification (Stockholm)

UTC-to-Stockholm mapping:

- winter (CET, UTC+1): `12:00Z = 13:00`, `15:30Z = 16:30`
- summer (CEST, UTC+2): `12:00Z = 14:00`, `15:30Z = 17:30`
- winter (CET): `15:00Z = 16:00`, `15:01Z = 16:01`
- summer (CEST): `15:00Z = 17:00`, `15:01Z = 17:01`

## 9) Operational Recommendation

If selecting strictly by observed backtest pnl in this matrix:

- use `blend_12` with gate `T-1 12:00Z`, with explicit monitoring of low-price fill concentration.

If selecting a more conservative deployment path:

- use `blend_12` with gate `T-1 15:30Z` and keep outlier/price-floor sensitivity checks active.

## 10) Reproducibility Note

All matrix runs were produced using:

- `backtesting/mos_blend00_entry1530z_cap400_audit.py`

with only:

- prediction files (`blend_00` vs `blend_12`) and
- entry gate time

changed across variants.

## 11) Update (2026-03-01): Explicit "Open + 30 Minutes" Entry Rule

Requested constraint:

- keep the same best strategy setup (`blend_12`, gate `T-1 12:00Z`, `EV >= 0.20`, `win >= 0.65`, `risk=5.5%`, `cap=400`)
- but force entry to satisfy:
  - `entry_timestamp_utc >= market_open_utc + 30 minutes`

Implemented execution rule:

- first quoted market row with:
  - `timestamp >= max(gate_utc, market_open_utc + 30m)`

Run outputs:

- `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\trades_matrix_blend_12_ev0p20_win65_risk5p5_cap400_tminus1_1200z_openplus30m.csv`
- `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_matrix_blend_12_ev0p20_win65_risk5p5_cap400_tminus1_1200z_openplus30m.json`
- `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\sanity_matrix_blend_12_ev0p20_win65_risk5p5_cap400_tminus1_1200z_openplus30m.json`
- `D:\Ahmed\data\kalshi\plots\all_trades_sideaware_blend12_tminus1_1200z_openplus30m_ev0p20_win65_risk5p5_cap400_with_balance.csv`

Headline metrics (this constrained variant):

- trades: `72`
- wins/losses: `50 / 22`
- win rate: `0.6944`
- profit factor: `1.7654`
- final balance: `$5,986.30`
- total pnl: `$3,286.30`
- max drawdown: `17.91%`

Sanity result:

- `passes_all_checks = true`
- `entry_before_effective_cutoff = 0`

Interpretation:

- adding the explicit open+30m floor materially reduced trade count and final pnl versus the unconstrained `blend_12 @ T-1 12:00Z` run.
