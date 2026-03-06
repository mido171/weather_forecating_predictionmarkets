# 22 - Run Record (2026-03-05, KLAX Onboarding + 4-Station `KNYC+KMIA+KMDW+KLAX` Backtest/UI)

## 1) Scope

This record captures:

1. onboarding `KLAX` / `KXHIGHLAX` into the SQLite-first station flow,
2. training/export of the KLAX `blend_12` live bundle,
3. KLAX parity smoke + live inference validation,
4. first audited fixed-risk 4-station cojoined backtest for `2025`,
5. UI wiring for the new 4-station dataset.

Resolved station metadata:

1. Kalshi series: `KXHIGHLAX`
2. MOS station id: `KLAX`
3. NWS/USW code: `USW00023174`
4. timezone: `America/Los_Angeles`
5. file prefix: `KLAX`

## 2) Station Flow Command

Executed:

```powershell
python tools/station_flow/run_station_full_flow.py `
  --station-id KXHIGHLAX `
  --nws-code USW00023174 `
  --mos-station-id KLAX `
  --station-zoneid America/Los_Angeles `
  --truth-end-date 2026-03-05 `
  --mos-start-year 2002 `
  --mos-end-year 2026 `
  --kalshi-start-date 2025-01-01 `
  --kalshi-end-date 2026-03-05 `
  --target-date 2026-03-06 `
  --run-cojoined `
  --cojoined-stations KNYC,KMIA,KMDW `
  --log-level INFO
```

Observed result:

1. phases 0-4 completed successfully (`NWS`, `MOS`, `Kalshi`, train, bundle, parity, live inference),
2. phase 5 exposed a bug in `tools/station_flow/run_station_full_flow.py`:
   - cojoined extras passed as station ids (`KNYC,KMIA,KMDW`) were not mapped to known Kalshi series roots,
   - resulting error: `kalshi root not found for KNYC`.
3. the bug was fixed by adding known station->series mapping for cojoined extras.

## 3) Canonical Station Artifacts

Canonical SQLite:

1. NWS:
   - `D:\Ahmed\data\sqlite\NWS\KLAX\KLAX_nws_truth_2002_2026.sqlite`
2. MOS:
   - `D:\Ahmed\data\sqlite\MOS\KLAX\KLAX_mos_2002_2026.sqlite`

Compatibility exports:

1. truth CSV:
   - `D:\Ahmed\data\kalshi\training_data\02_truth\KLAX_settled_tmax_2002_2026.csv`
2. MOS archive CSV.GZ:
   - `D:\Ahmed\data\kalshi\training_data\04_mos\archive_merged\KLAX_mos_archive_2002_2026.csv.gz`

Kalshi minute history root:

1. `D:\Ahmed\data\kalshi\kalshi_history\kxhighlax_2025_01_01_to_2026_03_05`

## 4) Training + Bundle Export

Training root:

1. `D:\Ahmed\data\kalshi\Experiments\MOS_KLAX`

Live bundle:

1. `D:\Ahmed\data\kalshi\Experiments\MOS_KLAX\03_blends\blend_12\live_model_bundle_v2_20260305`

Training report:

1. `D:\Ahmed\data\kalshi\Experiments\MOS_KLAX\09_reports\summary.json`

## 5) Parity + Live Inference

Parity smoke report:

1. `D:\Ahmed\data\runs\station_full_flow\kxhighlax\20260305T212539Z\phase3_parity_smoke\inference_report.json`

Single-station KLAX live inference report:

1. `D:\Ahmed\data\live\mos_quantile_live_inference\20260305T212539Z_target_20260306_klax\inference_report.json`

Default multi-station live script verification after KLAX default wiring:

```powershell
python tools/live/mos_quantile_live_inference.py --target-date 2026-03-06 --stdout-json summary --log-level ERROR
```

Verified report:

1. `D:\Ahmed\data\live\mos_quantile_live_inference\20260305T215400Z_target_20260306\inference_report.json`

Observed:

1. `inference_by_station` includes `KLAX`, `KMDW`, `KMIA`, `KNYC`
2. `KLAX.runtime_utc = 2026-03-05T12:00:00Z`
3. `passes_all_guardrails = true`
4. global guardrail counters are all `0`

## 6) Audited 4-Station Backtest

### 6.1 Locked Rule Pack

Entry gate rule:

- `entry_timestamp >= max(T-1 12:00:00Z, market_open_utc + 30m)`

Stake sizing rule:

- `stake = min(balance_before * 0.075, 700)`

Selection filters:

1. `EV >= 0.25`
2. `model_win_prob >= 0.85`
3. `side_market_price >= 0.25`

Window:

- `2025-01-01` through `2025-12-31`

Stations:

1. `KNYC`
2. `KMIA`
3. `KMDW`
4. `KLAX`

### 6.2 Command

```powershell
python backtesting/mos_blend12_knyc_kmia_cojoined_audit.py `
  --mode cojoined `
  --stations KNYC,KMIA,KMDW,KLAX `
  --start-date 2025-01-01 `
  --end-date 2025-12-31 `
  --entry-hour-z 12 `
  --entry-minute-z 0 `
  --min-entry-minutes-after-open 30 `
  --ev-min 0.25 `
  --win-min 0.85 `
  --min-market-price 0.25 `
  --risk-fraction 0.075 `
  --stake-cap-usd 700 `
  --out-prefix cojoined_blend12_knyc_kmia_kmdw_klax_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_2025 `
  --table-out D:\Ahmed\data\kalshi\plots\cojoined_blend12_knyc_kmia_kmdw_klax_stockholm_table_2025.csv
```

### 6.3 Artifacts

1. trades:
   - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\trades_cojoined_blend12_knyc_kmia_kmdw_klax_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_2025.csv`
2. summary:
   - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_cojoined_blend12_knyc_kmia_kmdw_klax_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_2025.json`
3. sanity:
   - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\sanity_cojoined_blend12_knyc_kmia_kmdw_klax_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_2025.json`
4. day debug:
   - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\day_debug_cojoined_blend12_knyc_kmia_kmdw_klax_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_2025.json`
5. side-aware with balance:
   - `D:\Ahmed\data\kalshi\plots\all_trades_sideaware_cojoined_blend12_knyc_kmia_kmdw_klax_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_2025_with_balance.csv`
6. stockholm table:
   - `D:\Ahmed\data\kalshi\plots\cojoined_blend12_knyc_kmia_kmdw_klax_stockholm_table_2025.csv`

### 6.4 Summary Snapshot

From summary JSON:

1. `trades = 337`
2. `wins/losses = 260 / 77`
3. `win_rate = 0.7715133531`
4. `profit_factor = 2.4169985329`
5. `final_balance = 77925.9240418257`
6. `max_drawdown = 0.2085468750`
7. `station_counts = {KMDW: 98, KLAX: 84, KMIA: 80, KNYC: 75}`

### 6.5 Sanity Snapshot

From sanity JSON:

1. `passes_all_checks = true`
2. `checked_trades = 337`
3. all failure counters are `0`

## 7) UI Wiring

New UI dataset option added:

1. key: `2025-klax`
2. label: `2025 + KLAX`

UI-backed CSV copy:

1. `D:\Ahmed\git\weather\weather_forecating_predictionmarkets\ui\result_viewer\public\data\all_trades_sideaware_cojoined_blend12_knyc_kmia_kmdw_klax_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_2025_with_balance.csv`

UI code updates:

1. `ui/result_viewer/src/App.jsx`
2. `ui/result_viewer/src/LiveTradingPage.jsx`

## 8) Default Station Wiring Updated

Updated so KLAX is treated as a first-class default station alongside `KNYC`, `KMIA`, and `KMDW`:

1. `tools/live/mos_quantile_live_inference.py`
2. `backtesting/mos_blend12_knyc_kmia_cojoined_audit.py`
3. `tools/station_flow/run_station_full_flow.py`
4. `kalshi-api-service/src/main/java/com/predictionmarkets/weather/kalshiapi/config/LiveTradingProperties.java`
5. `kalshi-api-service/src/main/resources/application.yml`

## 9) Outcome

KLAX onboarding is now complete for:

1. canonical NWS truth,
2. canonical MOS history,
3. KLAX training/export,
4. single-station live inference parity,
5. default multi-station live script coverage,
6. audited 4-station cojoined backtesting,
7. UI dataset exposure.
