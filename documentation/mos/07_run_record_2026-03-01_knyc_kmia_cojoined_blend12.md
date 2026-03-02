# 07 - Run Record (2026-03-01, KNYC + KMIA Co-Joined blend_12 Baseline)

This is the baseline co-joined fixed-risk run record (historical baseline, still valid for reference).

Superseded as "current strict reference" by:

- `documentation/mos/09_run_record_2026-03-02_knyc_kmia_cojoined_blend12_openplus30m_fractionalkelly_no_outlier_gt2000.md`

Status notes:

- this file documents fixed-risk baseline: `EV >= 0.15`, `win >= 0.65`, `risk=6.5%`, `cap=400`.
- strict historical variant without open-delay is documented in `08`.
- strict current variant with open+30m is documented in `09`.

## 1) Locked Rule Pack

- date range: `2024-10-01` to `2025-12-31`
- model: `blend_12` for both stations
- gate: `T-1 12:00:00Z`
- filters:
  - `EV >= 0.15`
  - `model_win_prob >= 0.65`
- bankroll:
  - `start_balance = 2700`
  - `risk_fraction = 0.065`
  - `stake_cap_usd = 400`
- execution invariant:
  - one trade max per day globally (across both stations)

Co-joined execution:

1. Build eligible quote streams for KNYC and KMIA from the first row with `timestamp >= gate`.
2. Walk timestamps in chronological order across the union of both streams.
3. At each timestamp, evaluate all eligible candidates from both stations.
4. Enter on the first timestamp where any candidate passes filters.
5. If multiple candidates pass at that timestamp, choose:
   - highest `model_win_prob`
   - then highest `EV`
   - then lowest `market_price`
   - then station alphabetical (`KMIA` then `KNYC` when all prior keys tie).

## 2) Commands Used

KMIA training (same split boundaries as KNYC):

```powershell
python ml/run_knyc_mos_first_plan.py --mos-csv "D:\Ahmed\data\kalshi\training_data\04_mos\archive_merged\KMIA_mos_archive_2000_2025.csv.gz" --truth-csv "D:\Ahmed\data\kalshi\training_data\02_truth\KMIA_settled_tmax.csv" --out-root "D:\Ahmed\data\kalshi\Experiments\MOS_KMIA" --dev-start 2022-01-01 --dev-end 2023-12-31 --test-start 2024-01-01 --test-end 2025-12-31 --seed 42
```

Kalshi minute downloads (dedicated dirs):

```powershell
python ingestion-service/scripts/kalshi_download_temperature_minute.py --series KXHIGHNY --start-date 2024-10-01 --end-date 2025-12-31 --out-dir "D:\Ahmed\data\kalshi\kalshi_history\kxhighny_2024_10_01_to_2025_12_31" --skip-existing
python ingestion-service/scripts/kalshi_download_temperature_minute.py --series KXHIGHMIA --start-date 2024-10-01 --end-date 2025-12-31 --out-dir "D:\Ahmed\data\kalshi\kalshi_history\kxhighmia_2024_10_01_to_2025_12_31" --skip-existing
```

Co-joined audited backtest:

```powershell
python backtesting/mos_blend12_knyc_kmia_cojoined_audit.py --start-date 2024-10-01 --end-date 2025-12-31 --entry-hour-z 12 --entry-minute-z 0 --ev-min 0.15 --win-min 0.65 --risk-fraction 0.065 --stake-cap-usd 400 --out-dir "D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest" --out-prefix "cojoined_blend12_knyc_kmia_tminus1_1200z_ev0p15_win65_risk6p5_cap400" --table-out "D:\Ahmed\data\kalshi\plots\cojoined_blend12_knyc_kmia_tminus1_1200z_ev0p15_win65_risk6p5_cap400_stockholm.csv"
```

## 3) Exportability / Validation Gate

Validation artifact:

- `D:\Ahmed\data\kalshi\Experiments\MOS_KMIA\09_reports\kmia_blend12_artifact_validation.json`

Checks passed:

- required files exist/readable
- required columns exist in dev/test
- test coverage includes `2024-10-01` to `2025-12-31`
- quantiles are non-decreasing (dev/test)
- no duplicate `target_date_local`

Reproducibility manifest:

- `D:\Ahmed\data\kalshi\Experiments\MOS_KMIA\09_reports\kmia_blend12_run_manifest_2026-03-01.json`

Note:

- one missing KMIA blend test date (`2025-05-30`) was resolved via deterministic fallback from available `nam_12` for that date only, documented in the manifest.

## 4) Leakage-Safety Rationale

This run is leakage-safe with respect to forecast timing and entry timing because:

1. Forecasts are consumed at a fixed as-of gate (`T-1 12:00Z`) for both stations.
2. Entry scan starts at `timestamp >= gate` only.
3. Trade selection is based only on quote rows available at each scanned timestamp.
4. Settlement (`win/loss`) uses realized `y_tmax` only after entry logic has already been finalized.
5. Sanity checks enforce:
   - no entry before gate,
   - one-trade/day global invariant,
   - entry timestamp equals first eligible global timestamp,
   - tie-break policy compliance.

## 5) Output Artifacts

- trades:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\trades_cojoined_blend12_knyc_kmia_tminus1_1200z_ev0p15_win65_risk6p5_cap400.csv`
- summary:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_cojoined_blend12_knyc_kmia_tminus1_1200z_ev0p15_win65_risk6p5_cap400.json`
- sanity:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\sanity_cojoined_blend12_knyc_kmia_tminus1_1200z_ev0p15_win65_risk6p5_cap400.json`
- day-level debug:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\day_debug_cojoined_blend12_knyc_kmia_tminus1_1200z_ev0p15_win65_risk6p5_cap400.json`
- side-aware full table:
  - `D:\Ahmed\data\kalshi\plots\all_trades_sideaware_cojoined_blend12_knyc_kmia_tminus1_1200z_ev0p15_win65_risk6p5_cap400_with_balance.csv`
- stockholm display table:
  - `D:\Ahmed\data\kalshi\plots\cojoined_blend12_knyc_kmia_tminus1_1200z_ev0p15_win65_risk6p5_cap400_stockholm.csv`

## 6) Headline Metrics

From summary JSON:

- trades: `434`
- wins/losses: `310 / 124`
- win rate: `0.7143`
- profit factor: `2.4146`
- start balance: `$2,700.00`
- final balance: `$71,455.40`
- total pnl: `$68,755.40`
- max drawdown: `18.26%`

Station contribution:

- `KNYC`: `248` trades
- `KMIA`: `186` trades

## 7) Sanity Audit Outcome

From sanity JSON:

- `passes_all_checks = true`
- `checked_trades = 434`
- all tracked failure counters are `0`, including:
  - one-trade/day violation
  - entry-before-gate
  - first-eligible timestamp mismatch
  - tie-break policy violation
  - price/probability/EV/pnl reconciliation failures

## 8) How This Baseline Relates To Current Strict Runs

Compared to later strict runs:

- baseline here has no open-delay requirement,
- baseline here uses fixed-risk sizing only,
- baseline here does not apply outlier-filtered recalc.

Use this file for historical comparability, not as the latest strict deployment reference.
