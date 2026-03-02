# 08 - Run Record (2026-03-01, KNYC + KMIA Co-Joined blend_12, Fractional Kelly, Outlier-Filtered Recalc, No Open-Delay)

This run record documents the historical strict co-joined variant without explicit open-delay.

Current strict reference with open+30m is:

- `documentation/mos/09_run_record_2026-03-02_knyc_kmia_cojoined_blend12_openplus30m_fractionalkelly_no_outlier_gt2000.md`

This file captures the prior variant using:

- stricter trade filters (`EV >= 0.18`, `model_win_prob >= 0.67`)
- fractional Kelly staking (`kelly_fraction = 0.15`)
- stake cap `$500`
- post-run outlier filter:
  - remove any trade where `pnl > 2000`
  - recompute bankroll path sequentially on remaining trades

This is a historical strict reference retained for comparison.

## 1) Locked Rule Pack

- date range: `2024-10-01` to `2025-12-31`
- stations: `KNYC`, `KMIA`
- forecast model per station: `blend_12`
- gate: `T-1 12:00:00Z`
- filters:
  - `EV >= 0.18`
  - `model_win_prob >= 0.67`
- bankroll:
  - `start_balance = 2700`
  - `risk_mode = fractional_kelly`
  - `kelly_fraction = 0.15`
  - `stake_cap_usd = 500`
- execution invariant:
  - at most one trade per day globally across both stations

Co-joined arbitration at a shared timestamp:

1. highest `model_win_prob`
2. highest `EV`
3. lowest `market_price`
4. station alphabetical (`KMIA` before `KNYC` only if prior keys tie)

## 2) Execution Semantics

Per target day:

1. Build both station quote streams with rows `timestamp >= gate`.
2. Walk global union of timestamps in ascending order.
3. At each timestamp, evaluate all candidate (bucket, side, station) rows.
4. First timestamp with any passing candidate is entry timestamp for that day.
5. Enter exactly one candidate after tie-break policy above.

No `open+30m` delay is applied in this run (this is the key difference vs run record `09`).

## 3) Sizing Formulas (Fractional Kelly)

For each chosen trade:

- let `p = market_price` (side price in [0,1])
- let `q = model_win_prob` (side probability in [0,1])

Full Kelly:

- `full_kelly = (q - p) / (1 - p)`
- clamp to `[0, 1]`
- if `p <= 0` or `p >= 1`, treat as `0`

Risk fraction used:

- `risk_fraction_used = 0.15 * full_kelly`

Stake:

- `stake = min(balance_before * risk_fraction_used, 500)`

Binary settlement:

- `shares = stake / p` (if `p > 0`, else `0`)
- win: `pnl = shares * (1 - p)`
- loss: `pnl = -stake`

## 4) Outlier Filter and Recalculation Semantics

Outlier rule applied exactly as requested:

- remove trades where `pnl > 2000`

Important:

- this filter is applied after generating the base trade stream
- then bankroll is recalculated sequentially from `start_balance = 2700`
- with the same fractional Kelly + cap rules for the remaining trades

This means all `stake`, `shares`, `pnl`, `balance_before`, `balance_after`, and drawdown values are rederived for the filtered sequence.

## 5) Commands Used

Base co-joined run (`EV 0.18`, `win 0.67`, cap 500; pre-filter stream):

```powershell
python backtesting/mos_blend12_knyc_kmia_cojoined_audit.py --start-date 2024-10-01 --end-date 2025-12-31 --entry-hour-z 12 --entry-minute-z 0 --ev-min 0.18 --win-min 0.67 --start-balance 2700 --risk-fraction 0.065 --stake-cap-usd 500 --out-dir "D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest" --out-prefix "cojoined_blend12_knyc_kmia_tminus1_1200z_ev0p18_win67_cap500_base" --table-out "D:\Ahmed\data\kalshi\plots\cojoined_blend12_knyc_kmia_tminus1_1200z_ev0p18_win67_cap500_base_stockholm.csv"
```

Post-processing step:

1. apply fractional Kelly `0.15`
2. apply cap `500`
3. remove trades `pnl > 2000`
4. recalc full bankroll path
5. write filtered outputs and sanity JSON

## 6) Output Artifacts

Primary filtered outputs:

- trades:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\trades_cojoined_blend12_knyc_kmia_tminus1_1200z_ev0p18_win67_fractionalkelly0p15_cap500_no_outlier_gt2000_recalc.csv`
- summary:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_cojoined_blend12_knyc_kmia_tminus1_1200z_ev0p18_win67_fractionalkelly0p15_cap500_no_outlier_gt2000_recalc.json`
- side-aware full table:
  - `D:\Ahmed\data\kalshi\plots\all_trades_sideaware_cojoined_blend12_knyc_kmia_tminus1_1200z_ev0p18_win67_fractionalkelly0p15_cap500_no_outlier_gt2000_recalc_with_balance.csv`
- sanity:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\sanity_cojoined_blend12_knyc_kmia_tminus1_1200z_ev0p18_win67_fractionalkelly0p15_cap500_no_outlier_gt2000_recalc.json`
- removed trades log:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\removed_trades_cojoined_blend12_knyc_kmia_tminus1_1200z_ev0p18_win67_fractionalkelly0p15_cap500_no_outlier_gt2000_recalc.csv`

Pre-filter reference (same thresholds, no outlier removal):

- `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\trades_cojoined_blend12_knyc_kmia_tminus1_1200z_ev0p18_win67_fractionalkelly0p15_cap500.csv`
- `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_cojoined_blend12_knyc_kmia_tminus1_1200z_ev0p18_win67_fractionalkelly0p15_cap500.json`
- `D:\Ahmed\data\kalshi\plots\all_trades_sideaware_cojoined_blend12_knyc_kmia_tminus1_1200z_ev0p18_win67_fractionalkelly0p15_cap500_with_balance.csv`

## 7) Metrics (Pre-Filter vs Post-Filter)

Pre-filter (`434` trades):

- wins/losses: `327 / 107`
- win rate: `75.35%`
- profit factor: `3.05396`
- final balance: `$111,367.01`
- total pnl: `$108,667.01`
- max drawdown: `14.47%`
- station counts:
  - `KNYC: 255`
  - `KMIA: 179`

Post-filter (`428` trades, `6` outliers removed):

- wins/losses: `321 / 107`
- win rate: `75.00%`
- profit factor: `2.02304`
- final balance: `$56,297.07`
- total pnl: `$53,597.07`
- max drawdown: `28.56%`

Removed outlier count:

- `6` trades with `pnl > 2000`

## 8) Removed Trades (Outlier Log Summary)

From removed-trades artifact:

- `2024-10-27` `KNYC` pnl `3241.87`
- `2024-11-05` `KNYC` pnl `6454.10`
- `2024-11-16` `KNYC` pnl `16166.67`
- `2024-11-19` `KMIA` pnl `16166.67`
- `2024-12-16` `KNYC` pnl `9500.00`
- `2025-11-28` `KNYC` pnl `2000.000000000001` (floating-point representation > 2000)

## 9) Sanity Outcome (Filtered Recalc)

Sanity file:

- `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\sanity_cojoined_blend12_knyc_kmia_tminus1_1200z_ev0p18_win67_fractionalkelly0p15_cap500_no_outlier_gt2000_recalc.json`

Result:

- `passes_all_checks = true`
- `checked_trades = 428`
- all failure counters `0`:
  - duplicate target date
  - entry before gate
  - stake cap breach
  - risk fraction mismatch
  - stake formula mismatch
  - pnl formula mismatch
  - any remaining `pnl > 2000`

## 10) Operational Notes

1. This run is a post-processed variant of a leakage-safe co-joined trade stream.
2. Outlier filtering is a reporting/risk sensitivity operation; it changes compounding path materially.
3. This run does not enforce entry `>= market_open + 30m`.
4. Any comparison with run `09` must explicitly state this execution-rule difference.
5. For reproducibility, always cite summary JSON and side-aware CSV path together.
