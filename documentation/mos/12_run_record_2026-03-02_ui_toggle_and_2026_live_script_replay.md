# 12 - Run Record (2026-03-02, UI Window Toggle + 2026 Live-Script Replay Backtest)

This run record documents a full extension of the live-script replay workflow:

1. produce a dedicated 2026 replay dataset (co-joined KNYC + KMIA),
2. keep the existing 2024-2025 dataset unchanged,
3. expose both datasets in the UI via a top toggle:
   - `2024-2025`
   - `2026`

## 1) Action Plan (Executed)

1. Refresh 2026 settlement truth for both stations (KNYC, KMIA).
2. Download 2026 Kalshi minute data for both stations.
3. Run co-joined replay backtest with `--prediction-source live-script` for 2026.
4. Copy the 2026 side-aware table into UI `public/data`.
5. Implement UI toggle and wire each option to its own CSV.

All five steps were executed in this order.

## 2) 2026 Data Preparation

### 2.1 Settlement truth refresh commands

```powershell
python tools/ncei_truth/run.py --stations KNYC --start-date 2026-01-01 --end-date 2026-03-02 --root-dir "D:\Ahmed\data\truth_tmax" --write-simple-settlement-csv-path "D:\Ahmed\data\kalshi\training_data\02_truth\KNYC_settled_tmax_2026_refresh.csv" --simple-settlement-station-id KNYC --log-level INFO

python tools/ncei_truth/run.py --stations KMIA --start-date 2026-01-01 --end-date 2026-03-02 --root-dir "D:\Ahmed\data\truth_tmax" --write-simple-settlement-csv-path "D:\Ahmed\data\kalshi\training_data\02_truth\KMIA_settled_tmax_2026_refresh.csv" --simple-settlement-station-id KMIA --log-level INFO
```

Observed latest settled dates from refresh outputs:

- KNYC: `2026-02-24`
- KMIA: `2026-02-26`

Strict common settled window used for co-joined backtest:

- `2026-01-01` to `2026-02-24`

### 2.2 Kalshi minute data download commands

```powershell
python ingestion-service/scripts/kalshi_download_temperature_minute.py --series KXHIGHNY --start-date 2026-01-01 --end-date 2026-02-24 --out-dir "D:\Ahmed\data\kalshi\kalshi_history\kxhighny_2026_to_20260224" --skip-existing

python ingestion-service/scripts/kalshi_download_temperature_minute.py --series KXHIGHMIA --start-date 2026-01-01 --end-date 2026-02-24 --out-dir "D:\Ahmed\data\kalshi\kalshi_history\kxhighmia_2026_to_20260224" --skip-existing
```

Resulting coverage:

- KNYC files: `KNYC_20260101.csv` .. `KNYC_20260224.csv` (55 days)
- KMIA files: `KMIA_20260101.csv` .. `KMIA_20260224.csv` (55 days)

## 3) 2026 Live-Script Replay Backtest

Command:

```powershell
python backtesting/mos_blend12_knyc_kmia_cojoined_audit.py --prediction-source live-script --start-date 2026-01-01 --end-date 2026-02-24 --entry-hour-z 12 --entry-minute-z 0 --min-entry-minutes-after-open 30 --ev-min 0.30 --win-min 0.75 --min-market-price 0.10 --start-balance 2700 --risk-fraction 0.06 --stake-cap-usd 500 --kalshi-root-knyc "D:\Ahmed\data\kalshi\kalshi_history\kxhighny_2026_to_20260224" --kalshi-root-kmia "D:\Ahmed\data\kalshi\kalshi_history\kxhighmia_2026_to_20260224" --truth-csv-knyc "D:\Ahmed\data\kalshi\training_data\02_truth\KNYC_settled_tmax_2026_refresh.csv" --truth-csv-kmia "D:\Ahmed\data\kalshi\training_data\02_truth\KMIA_settled_tmax_2026_refresh.csv" --live-inference-root "D:\Ahmed\data\live\mos_quantile_live_inference\backtest_replay_2026" --out-dir "D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest" --out-prefix "cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script_2026" --table-out "D:\Ahmed\data\kalshi\plots\cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script_2026_stockholm.csv"
```

### 3.1 Rule anchors for this run (explicit)

- entry gate rule:
  - `entry_timestamp >= max(T-1 12:00:00Z, market_open_utc + 30m)`
- stake sizing rule:
  - `stake = min(balance_before * 0.06, 500)`
- summary JSON:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script_2026.json`
- sanity JSON:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\sanity_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script_2026.json`

### 3.2 Output artifacts

- trades:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\trades_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script_2026.csv`
- summary:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script_2026.json`
- sanity:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\sanity_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script_2026.json`
- day debug:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\day_debug_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script_2026.json`
- side-aware with balance:
  - `D:\Ahmed\data\kalshi\plots\all_trades_sideaware_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script_2026_with_balance.csv`
- stockholm display table:
  - `D:\Ahmed\data\kalshi\plots\cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script_2026_stockholm.csv`

### 3.3 Summary snapshot

From 2026 summary JSON:

- `total_days = 55`
- `trades = 52`
- `wins = 35`
- `losses = 17`
- `win_rate = 0.6731`
- `profit_factor = 1.4876`
- `final_balance = 5226.59`
- `max_drawdown = 0.2193`

Sanity:

- `passes_all_checks = true`

## 4) UI Toggle Implementation

Updated files:

- `ui/result_viewer/src/App.jsx`
- `ui/result_viewer/src/styles.css`

Behavior:

- top toggle offers two windows:
  - `2024-2025`
  - `2026`
- `2024-2025` points to existing dataset:
  - `/data/all_trades_sideaware_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script_with_balance.csv`
- `2026` points to new dataset:
  - `/data/all_trades_sideaware_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script_2026_with_balance.csv`

UI-served 2026 file copied to:

- `ui/result_viewer/public/data/all_trades_sideaware_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script_2026_with_balance.csv`

Notes:

- Existing `csv` query param / `VITE_TRADES_CSV_FILE` override still works and takes precedence.

