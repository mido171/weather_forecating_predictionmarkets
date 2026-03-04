# MOS Backtesting Documentation

This folder is the canonical documentation set for MOS-first Kalshi backtesting in this repo.

It is explicitly designed for:

- reproducibility (same inputs -> same outputs),
- auditability (JSON sanity checks + deterministic tie-breaks),
- execution-time correctness (no entry before gate/effective cutoff),
- leakage paranoia (runtime-aligned model use, no future quote use).

## Scope

This track covers two execution families:

1. Single-station MOS runs (historical `blend_00`, runtime matrix `blend_12`/`blend_00`).
2. Co-joined multi-station MOS runs (`KNYC` + `KMIA` + optional additional stations such as `KMDW`, one trade/day globally).

It also documents:

- data contracts and date/file mapping,
- entry-gate vs market-open delay semantics,
- fixed-risk and fractional-Kelly sizing,
- outlier-filtered recalc protocol,
- sanity/audit acceptance criteria.

## Canonical Data Source Policy (Mandatory)

For MOS/NWS data in this workflow, the primary read path is:

1. `D:\Ahmed\data\sqlite\NWS\<STATION>\...`
2. `D:\Ahmed\data\sqlite\MOS\<STATION>\...`

Policy:

1. SQLite under `D:\Ahmed\data\sqlite` is the canonical source-of-truth for MOS/NWS.
2. Compatibility CSV exports are derived from SQLite and are not canonical.
3. Any run claiming station onboarding or canonical data quality must reference SQLite artifacts first, then export artifacts.

## Current Canonical Reference

Current strict methodology reference:

- run record:
  - `documentation/mos/09_run_record_2026-03-02_knyc_kmia_cojoined_blend12_openplus30m_fractionalkelly_no_outlier_gt2000.md`
- core side-aware table:
  - `D:\Ahmed\data\kalshi\plots\all_trades_sideaware_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p18_win67_fractionalkelly0p15_cap500_no_outlier_gt2000_recalc_with_balance.csv`

Current live-script replay/UI dataset:

- run record:
  - `documentation/mos/11_run_record_2026-03-02_cojoined_blend12_live_script_replay.md`
- side-aware table (window `2024-2025`):
  - `D:\Ahmed\data\kalshi\plots\all_trades_sideaware_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script_with_balance.csv`
- windowed UI + 2026 extension run record:
  - `documentation/mos/12_run_record_2026-03-02_ui_toggle_and_2026_live_script_replay.md`
- side-aware table (window `2026`):
  - `D:\Ahmed\data\kalshi\plots\all_trades_sideaware_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script_2026_with_balance.csv`

Current fixed-risk multi-station (`KNYC+KMIA+KMDW`) UI-backed dataset:

- run record:
  - `documentation/mos/16_run_record_2026-03-04_cojoined_blend12_knyc_kmia_kmdw_openplus30m_risk7p5_cap700_2024_2025.md`
- side-aware table (window `2024-2025`):
  - `D:\Ahmed\data\kalshi\plots\all_trades_sideaware_cojoined_blend12_knyc_kmia_kmdw_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_2024_2025_with_balance.csv`
- UI engineering record (virtualized table + station scoring):
  - `documentation/mos/17_run_record_2026-03-04_ui_full_trade_table_virtualization_and_station_contribution_scoring.md`

## Mandatory Read Order

1. `documentation/mos/00_scope_and_objective.md`
2. `documentation/mos/01_data_contracts_and_file_mapping.md`
3. `documentation/mos/02_backtest_logic_and_formulas.md`
4. `documentation/mos/03_sanity_audit_framework.md`
5. `documentation/mos/04_run_record_2026-03-01_entry1530z_cap400.md`
6. `documentation/mos/06_run_record_2026-03-01_leakage_free_runtime_matrix.md`
7. `documentation/mos/07_run_record_2026-03-01_knyc_kmia_cojoined_blend12.md`
8. `documentation/mos/08_run_record_2026-03-01_knyc_kmia_cojoined_blend12_fractionalkelly_no_outlier_gt2000.md`
9. `documentation/mos/09_run_record_2026-03-02_knyc_kmia_cojoined_blend12_openplus30m_fractionalkelly_no_outlier_gt2000.md`
10. `documentation/mos/13_actionplan_station_full_flow_sqlite.md`
11. `documentation/mos/14_station_full_flow_sqlite_runbook.md`
12. `documentation/mos/15_station_full_flow_sqlite_data_contracts.md`
13. `documentation/mos/16_run_record_2026-03-04_cojoined_blend12_knyc_kmia_kmdw_openplus30m_risk7p5_cap700_2024_2025.md`
14. `documentation/mos/17_run_record_2026-03-04_ui_full_trade_table_virtualization_and_station_contribution_scoring.md`
15. `documentation/mos/18_engineering_spec_2026-03-04_generic_cojoined_backtester_cli_and_station_mapping.md`
16. `documentation/mos/19_engineering_spec_2026-03-04_live_inference_station_generic_and_shared_bundle_module.md`
17. `documentation/mos/05_troubleshooting_and_common_failure_modes.md`

## Document Map

- `00_scope_and_objective.md`
  - scope boundaries, supported strategy families, and hard invariants.
- `01_data_contracts_and_file_mapping.md`
  - input/output schemas, naming contracts, date mapping, and price normalization.
- `02_backtest_logic_and_formulas.md`
  - full execution algorithm, sizing formulas, EV logic, and post-processing semantics.
- `03_sanity_audit_framework.md`
  - required sanity counters and pass criteria for single-station and co-joined runs.
- `04_run_record_2026-03-01_entry1530z_cap400.md`
  - audited single-station baseline run record.
- `05_troubleshooting_and_common_failure_modes.md`
  - operational debugging guide for timing, parsing, selection, and arithmetic issues.
- `06_run_record_2026-03-01_leakage_free_runtime_matrix.md`
  - blend_00 vs blend_12 runtime matrix benchmark and leakage timing checks.
- `07_run_record_2026-03-01_knyc_kmia_cojoined_blend12.md`
  - first co-joined baseline run record (fixed-risk, no open-delay).
- `08_run_record_2026-03-01_knyc_kmia_cojoined_blend12_fractionalkelly_no_outlier_gt2000.md`
  - historical strict run (fractional-Kelly + outlier recalc, no open-delay).
- `09_run_record_2026-03-02_knyc_kmia_cojoined_blend12_openplus30m_fractionalkelly_no_outlier_gt2000.md`
  - current strict reference (open+30m + strict filters + post-processed recalc).
- `10_run_record_2026-03-02_model_export_and_live_inference.md`
  - explicit model lineage for `ev0p30_win75_risk6_cap500_minprice10c`, retrain/export record, and live inference leakage-proof implementation details.
- `11_run_record_2026-03-02_cojoined_blend12_live_script_replay.md`
  - co-joined backtest replay where per-day forecasts are generated from the live inference script instead of parquet prediction files.
- `12_run_record_2026-03-02_ui_toggle_and_2026_live_script_replay.md`
  - 2026 co-joined replay extension and UI dataset toggle (`2024-2025` vs `2026`).
- `13_actionplan_station_full_flow_sqlite.md`
  - implementation actionplan for station onboarding with SQLite-first canonical stores and live `blend_12` parity gates.
- `14_station_full_flow_sqlite_runbook.md`
  - operator runbook for executing the full station pipeline, recovery patterns, and expected outputs.
- `15_station_full_flow_sqlite_data_contracts.md`
  - canonical schema/contract spec for NWS/MOS SQLite stores and compatibility exports.
- `16_run_record_2026-03-04_cojoined_blend12_knyc_kmia_kmdw_openplus30m_risk7p5_cap700_2024_2025.md`
  - audited 3-station co-joined fixed-risk run (`KNYC`, `KMIA`, `KMDW`) with full artifact and comparison record.
- `17_run_record_2026-03-04_ui_full_trade_table_virtualization_and_station_contribution_scoring.md`
  - UI engineering run record for table virtualization and station contribution scoring panel.
- `18_engineering_spec_2026-03-04_generic_cojoined_backtester_cli_and_station_mapping.md`
  - detailed CLI/spec refactor record for generic multi-station co-joined backtesting.
- `19_engineering_spec_2026-03-04_live_inference_station_generic_and_shared_bundle_module.md`
  - live inference station-generic refactor and shared blend_12 bundle module specification.

## Canonical Scripts

- `backtesting/mos_blend00_entry1530z_cap400_audit.py`
- `backtesting/mos_blend12_knyc_kmia_cojoined_audit.py`

Note:

- `mos_blend12_knyc_kmia_cojoined_audit.py` natively supports fixed-risk sizing.
- fractional-Kelly + outlier-filtered recalc is currently a deterministic post-processing layer on top of co-joined trade stream outputs.

## Key Output Families

### Single-Station (blend_00 entry1530z baseline)

- trades:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\trades_all_available_blend00_ev0p1_win60_risk5p5pct_entry1530z_cap400.csv`
- summary:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_all_available_blend00_ev0p1_win60_risk5p5pct_entry1530z_cap400.json`
- sanity:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\sanity_all_available_blend00_ev0p1_win60_risk5p5pct_entry1530z_cap400.json`
- deep sanity:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\deep_sanity_all_available_blend00_ev0p1_win60_risk5p5pct_entry1530z_cap400.json`

### Runtime Matrix (blend_00 vs blend_12)

- matrix comparison:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\matrix_blend00_blend12_ev0p20_win65_risk5p5_cap400_comparison.csv`
- freshness/performance:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\matrix_blend00_blend12_freshness_vs_performance.csv`
- outlier-capped recalc matrix:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\matrix_blend00_blend12_ev0p20_win65_risk5p5_cap400_pnlcap3000removed_recalc_comparison.csv`

### Co-Joined Baseline (KNYC+KMIA, fixed risk)

- run record:
  - `documentation/mos/07_run_record_2026-03-01_knyc_kmia_cojoined_blend12.md`
- summary:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_cojoined_blend12_knyc_kmia_tminus1_1200z_ev0p15_win65_risk6p5_cap400.json`

### Co-Joined Strict (historical no-open-delay variant)

- run record:
  - `documentation/mos/08_run_record_2026-03-01_knyc_kmia_cojoined_blend12_fractionalkelly_no_outlier_gt2000.md`

### Co-Joined Strict (latest open+30m variant)

- run record:
  - `documentation/mos/09_run_record_2026-03-02_knyc_kmia_cojoined_blend12_openplus30m_fractionalkelly_no_outlier_gt2000.md`
- fixed-risk base summary:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p18_win67_risk6p5_cap500_base.json`
- strict post-processed summary:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p18_win67_fractionalkelly0p15_cap500_no_outlier_gt2000_recalc.json`
- strict side-aware table:
  - `D:\Ahmed\data\kalshi\plots\all_trades_sideaware_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p18_win67_fractionalkelly0p15_cap500_no_outlier_gt2000_recalc_with_balance.csv`

### Model Export + Live Inference (KNYC + KMIA blend_12)

- run record:
  - `documentation/mos/10_run_record_2026-03-02_model_export_and_live_inference.md`
- lineage/export manifest:
  - `D:\Ahmed\data\kalshi\Experiments\MOS_RETRAIN_20260302\model_lineage_and_exports_for_cojoined_ev0p30_win75_risk6.json`

### Co-Joined Live-Script Replay (KNYC+KMIA, open+30m, fixed risk)

- run record:
  - `documentation/mos/11_run_record_2026-03-02_cojoined_blend12_live_script_replay.md`
- summary:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script.json`
- sanity:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\sanity_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script.json`
- side-aware table:
  - `D:\Ahmed\data\kalshi\plots\all_trades_sideaware_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script_with_balance.csv`

### UI Windowed Datasets (2024-2025 + 2026)

- run record:
  - `documentation/mos/12_run_record_2026-03-02_ui_toggle_and_2026_live_script_replay.md`
- 2026 summary:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script_2026.json`
- 2026 sanity:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\sanity_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script_2026.json`
- 2026 side-aware table:
  - `D:\Ahmed\data\kalshi\plots\all_trades_sideaware_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_live_script_2026_with_balance.csv`

### Co-Joined Fixed-Risk 3-Station (KNYC+KMIA+KMDW, 2024-2025)

- run record:
  - `documentation/mos/16_run_record_2026-03-04_cojoined_blend12_knyc_kmia_kmdw_openplus30m_risk7p5_cap700_2024_2025.md`
- summary:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_cojoined_blend12_knyc_kmia_kmdw_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_2024_2025.json`
- sanity:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\sanity_cojoined_blend12_knyc_kmia_kmdw_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_2024_2025.json`
- side-aware table:
  - `D:\Ahmed\data\kalshi\plots\all_trades_sideaware_cojoined_blend12_knyc_kmia_kmdw_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_2024_2025_with_balance.csv`

### UI Performance + Station Contribution Analytics

- run record:
  - `documentation/mos/17_run_record_2026-03-04_ui_full_trade_table_virtualization_and_station_contribution_scoring.md`
- core UI files:
  - `ui/result_viewer/src/App.jsx`
  - `ui/result_viewer/src/styles.css`

## Claim Hygiene Rule

A MOS performance claim is non-authoritative unless it includes all five:

1. exact entry rule (gate and any open-delay),
2. exact filters (`EV`, `model_win_prob`),
3. exact sizing policy (fixed risk or Kelly, cap),
4. exact summary JSON path,
5. exact sanity JSON path.
