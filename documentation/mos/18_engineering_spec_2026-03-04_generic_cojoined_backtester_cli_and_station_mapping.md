# 18 - Engineering Spec (2026-03-04, Generic Co-Joined Backtester CLI + Station Mapping)

## 1) Purpose

Document the `backtesting/mos_blend12_knyc_kmia_cojoined_audit.py` refactor that generalized execution from fixed `KNYC/KMIA` assumptions to configurable station sets, while keeping full sanity/audit behavior.

## 2) New CLI Surface

### 2.1 Mode and station selection

1. `--mode single|cojoined`
2. `--stations <comma-separated station ids>`

Validation:

1. `--stations` must resolve to at least one station id
2. `--mode single` requires exactly one station id

### 2.2 Per-station mapping inputs

Optional JSON mapping arguments:

1. `--pred-dev-by-station-json`
2. `--pred-test-by-station-json`
3. `--truth-csv-by-station-json`
4. `--kalshi-root-by-station-json`
5. `--file-prefix-by-station-json`

Input format:

1. inline JSON object string, or
2. file path to a JSON object on disk

Object contract:

- keys are normalized to uppercase station ids
- values are normalized string paths/prefixes

### 2.3 Legacy defaults retained

Legacy defaults remain for compatibility and are used as fallback if JSON maps are omitted:

1. `KNYC`
2. `KMIA`
3. `KMDW` (added)

## 3) Internal Refactor Summary

The following flows were made station-list aware:

1. prediction map construction
2. market index construction
3. per-day context generation
4. candidate generation and arbitration
5. backtest aggregate counters
6. sanity audit replay checks
7. summary metadata emission

Helper additions:

1. `parse_station_ids(...)`
2. `parse_json_mapping(...)`

## 4) Prediction Source Modes

`prediction_source = parquet`

1. loads station-specific dev/test parquet pairs
2. concatenates and deduplicates by `target_date_local`
3. emits `prediction_source_meta.pred_dev_by_station` and `pred_test_by_station`

`prediction_source = live-script`

1. invokes `tools/live/mos_quantile_live_inference.py` by date
2. reads inference blocks for each configured station from:
   - `inference_by_station[STATION]` (primary)
   - legacy `inference_<station_lower>` fallback
3. joins with station-specific truth map
4. records live-loader stats in summary metadata

## 5) Summary JSON Contract Changes

Added summary fields:

1. `mode`
2. `stations`
3. `kalshi_roots_by_station`
4. `file_prefix_by_station`
5. generalized counters:
   - `days_with_prediction_by_station`
   - `days_with_market_file_by_station`

Retained fields:

1. `trades`, `wins`, `losses`, `win_rate`
2. `profit_factor`, `final_balance`, `total_pnl`
3. `max_drawdown`, `station_counts`, `side_counts`
4. `prediction_source_meta`

## 6) Sanity/Audit Integrity

No audit gates were removed. Core counters remain:

1. timing checks (`entry_before_gate`, `entry_before_effective_cutoff`)
2. first-eligible timestamp check
3. tie-break policy replay check
4. bucket/price/probability/EV reconciliations
5. stake cap and pnl arithmetic reconciliation
6. one-trade/day global constraint

## 7) Example Calls

### 7.1 Single station

```powershell
python backtesting/mos_blend12_knyc_kmia_cojoined_audit.py `
  --mode single `
  --stations KMDW `
  --prediction-source parquet
```

### 7.2 Multi-station co-joined

```powershell
python backtesting/mos_blend12_knyc_kmia_cojoined_audit.py `
  --mode cojoined `
  --stations KNYC,KMIA,KMDW `
  --prediction-source parquet `
  --start-date 2024-10-01 `
  --end-date 2025-12-31
```

### 7.3 Multi-station with mapping file

```powershell
python backtesting/mos_blend12_knyc_kmia_cojoined_audit.py `
  --mode cojoined `
  --stations KNYC,KMIA,KMDW `
  --pred-dev-by-station-json D:\path\pred_dev_map.json `
  --pred-test-by-station-json D:\path\pred_test_map.json `
  --truth-csv-by-station-json D:\path\truth_map.json `
  --kalshi-root-by-station-json D:\path\kalshi_root_map.json `
  --file-prefix-by-station-json D:\path\prefix_map.json
```

## 8) Failure Modes (Refactor-Specific)

1. station listed in `--stations` without available prediction paths (parquet mode) -> hard error
2. station listed in `--stations` without truth path (live-script mode) -> hard error
3. station listed in `--stations` without Kalshi root path -> hard error
4. malformed mapping JSON -> hard error

## 9) Relationship to Existing MOS Rule Pack Docs

This refactor changes execution *configuration breadth* only. It does not alter core arithmetic or selection invariants from:

1. `documentation/mos/02_backtest_logic_and_formulas.md`
2. `documentation/mos/03_sanity_audit_framework.md`

For claim hygiene, keep using:

1. exact entry rule
2. exact stake rule
3. exact summary JSON path
4. exact sanity JSON path

## 10) Traceability

Script:

- `backtesting/mos_blend12_knyc_kmia_cojoined_audit.py`

First audited 3-station run record using this generalized path:

- `documentation/mos/16_run_record_2026-03-04_cojoined_blend12_knyc_kmia_kmdw_openplus30m_risk7p5_cap700_2024_2025.md`
