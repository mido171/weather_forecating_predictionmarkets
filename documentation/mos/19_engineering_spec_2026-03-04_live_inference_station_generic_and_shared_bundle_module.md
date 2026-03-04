# 19 - Engineering Spec (2026-03-04, Live Inference Station-Generic Refactor + Shared `blend_12` Bundle Module)

## 1) Purpose

Document the live inference refactor in:

- `tools/live/mos_quantile_live_inference.py`

and shared model module usage:

- `tools/live/mos_blend12_bundle.py`

This change removed hardcoded two-station assumptions and aligned training/loading logic through one shared bundle implementation.

## 2) Core Functional Changes

### 2.1 Shared bundle logic

`mos_quantile_live_inference.py` now imports:

- `tools.live.mos_blend12_bundle as blend12_bundle`

and uses shared constants/functions for:

1. `SLICE_DEFS` (`gfs_12`, `nam_12`)
2. feature columns
3. quantile list
4. `train_and_write_bundle(...)`
5. `load_bundle(...)`

Result:

- live inference and training/export paths use one canonical `blend_12` implementation.

### 2.2 Added KMDW defaults

New default station wiring:

1. station id: `KMDW`
2. series: `KXHIGHCHI`
3. zone: `America/Chicago`
4. model bundle default:
   - `D:\Ahmed\data\kalshi\Experiments\MOS_KMDW\03_blends\blend_12\live_model_bundle_v2_20260304`
5. MOS archive default:
   - `D:\Ahmed\data\kalshi\training_data\04_mos\archive_merged\KMDW_mos_archive_2002_2026.csv.gz`
6. truth default:
   - `D:\Ahmed\data\kalshi\training_data\02_truth\KMDW_settled_tmax_2002_2026.csv`

## 3) New Station Configuration Modes

Live script can now resolve stations via three modes:

1. `--station-configs-json` (multi-station object/array config file)
2. single-station explicit mode:
   - `--station-id`
   - `--station-zoneid`
   - `--series`
   - `--file-prefix`
   - `--bundle-dir`
   - `--mos-archive`
   - `--truth-csv`
   - optional `--market-root`
3. default multi-station mode (no explicit station args):
   - `KMIA`
   - `KMDW`
   - `KNYC`

## 4) Report Schema Changes

Primary report payload now includes:

1. `inference_by_station` map (station-keyed)
2. dynamic guardrail counters computed across configured stations

Backward compatibility preserved:

- script still writes legacy station keys when present:
  - `inference_knyc`
  - `inference_kmia`
  - `inference_kmdw`

## 5) Guardrail Behavior

Leakage guardrail accounting now scales with configured station set:

1. runtime vs quote-asof checks
2. runtime-policy equality checks
3. quantile monotonicity checks
4. non-leakage-free feature row checks

No station-specific hardcoding remains in guardrail counters.

## 6) CLI Surface Additions

New arguments documented in script:

1. `--bundle-dir-kmdw`
2. `--mos-archive-kmdw`
3. `--truth-csv-kmdw`
4. `--market-root-kmdw`
5. `--station-configs-json`
6. single-station explicit args (`--station-id`, `--station-zoneid`, `--series`, `--file-prefix`, `--bundle-dir`, `--mos-archive`, `--truth-csv`, `--market-root`)

## 7) Why This Matters for MOS Operations

1. New stations can be onboarded without editing core inference loops.
2. Backtest live-script replay can consume inference reports with arbitrary station sets.
3. Model parity drift risk is reduced by centralizing blend_12 bundle code in one module.

## 8) Contract With Co-Joined Backtester

`backtesting/mos_blend12_knyc_kmia_cojoined_audit.py` live-script mode reads:

1. `inference_by_station[STATION]` first
2. falls back to legacy `inference_<station_lower>` key

This keeps replay compatibility across old and new inference report formats.

## 9) Known Operational Note

When default mode is used (no explicit station override), live inference now includes three stations (`KMIA`, `KMDW`, `KNYC`) rather than the previous two-station default.

## 10) Traceability

Primary run records tied to this change family:

1. `documentation/mos/13_actionplan_station_full_flow_sqlite.md`
2. `documentation/mos/14_station_full_flow_sqlite_runbook.md`
3. `documentation/mos/15_station_full_flow_sqlite_data_contracts.md`
4. `documentation/mos/16_run_record_2026-03-04_cojoined_blend12_knyc_kmia_kmdw_openplus30m_risk7p5_cap700_2024_2025.md`
5. `documentation/mos/18_engineering_spec_2026-03-04_generic_cojoined_backtester_cli_and_station_mapping.md`
