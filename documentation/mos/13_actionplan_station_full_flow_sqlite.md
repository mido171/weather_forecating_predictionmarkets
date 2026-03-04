# 13 - Action Plan: Station Full Flow (SQLite-First, Live `blend_12` Parity)

## Objective
Implement a one-command station onboarding flow that:
1. Downloads all required data first.
2. Stores canonical MOS and NWS truth in station-specific SQLite stores under `D:\Ahmed\data\sqlite\...`.
3. Exports compatibility CSV artifacts for existing train/live/backtest scripts.
4. Trains the same `blend_12` live model strategy used by `tools/live/mos_quantile_live_inference.py`.
5. Runs parity validation against the live inference path.
6. Supports optional live inference and backtesting (single/cojoined).

Mandatory canonical-read statement:

1. MOS/NWS data must be read canonically from `D:\Ahmed\data\sqlite` stores.
2. CSV artifacts are compatibility outputs derived from SQLite, not the primary source-of-truth.

## Required Inputs
`tools/station_flow/run_station_full_flow.py` required inputs:
1. `--station-id` (Kalshi series ticker, e.g. `KXHIGHTATL`)
2. `--nws-code` (USW station code, e.g. `USW00013874`)

Primary optional inputs:
1. `--data-root` (default `D:\Ahmed\data`)
2. `--mos-station-id`, `--station-zoneid` (for fail-closed metadata overrides)
3. `--truth-start-date`, `--truth-end-date`
4. `--mos-start-year`, `--mos-end-year`
5. `--kalshi-start-date`, `--kalshi-end-date`
6. `--dev-start`, `--dev-end`, `--test-start`, `--test-end`
7. `--target-date`
8. `--backtest-mode single|cojoined`, `--run-cojoined`, `--cojoined-stations`
9. `--resume`, `--log-level`

## Locked Live Strategy Parity
Per implementation, station model export/loading is centralized in `tools/live/mos_blend12_bundle.py` and used by live inference:
1. Runtime policy: `blend12_tminus1_1200z`
2. Slice family: `gfs_12` + `nam_12`
3. Residual modeling target: `resid = y_tmax - mos_tmax_raw`
4. LightGBM point objective `l1`, quantile objective `quantile`
5. Quantiles: `0.05,0.10,0.25,0.50,0.75,0.90,0.95`
6. Dev tuning: `2022-01-01..2023-12-31`
7. Full train end: `2023-12-31`
8. Blend weights tuned on grid `[0, 0.25, 0.5, 0.75, 1.0]`
9. Bundle schema: `mos_blend12_live_bundle_v1`

## Data-First Execution Order (Mandatory)
### Phase 0 - Preflight
1. Resolve station metadata from Kalshi series.
2. Fail closed if station id/timezone cannot be resolved and no override is provided.
3. Create run root:
`D:\Ahmed\data\runs\station_full_flow\<series>\<timestamp>\`
4. Persist `resolved_inputs.json`.

### Phase 1 - Mandatory Bootstrap First
1. NWS truth bootstrap from NCEI daily summaries for requested window.
2. MOS bootstrap (`GFS`, `NAM`) for requested year window.
3. Kalshi minute bootstrap for series/date window.
4. Write canonical SQLite stores:
   - `D:\Ahmed\data\sqlite\NWS\<STATION>\<STATION>_nws_truth_....sqlite`
   - `D:\Ahmed\data\sqlite\MOS\<STATION>\<STATION>_mos_....sqlite`
5. Use SQLite as the canonical read source for MOS/NWS in downstream phase contracts.
6. Export compatibility files from SQLite:
   - `D:\Ahmed\data\kalshi\training_data\02_truth\<STATION>_settled_tmax_2002_2026.csv`
   - `D:\Ahmed\data\kalshi\training_data\04_mos\archive_merged\<STATION>_mos_archive_2002_2026.csv.gz`

### Phase 2 - Train + Export
1. Train station pipeline (station/timezone generic runner).
2. Export live bundle:
`D:\Ahmed\data\kalshi\Experiments\MOS_<STATION>\03_blends\blend_12\live_model_bundle_v2_<YYYYMMDD>`

### Phase 3 - Parity Gate (Mandatory)
1. Load bundle via live loader.
2. Validate schema/artifact set for `gfs_12` + `nam_12`.
3. Run smoke inference using the new bundle.
4. Verify runtime evidence equals policy runtime (`T-1 12:00Z`) and guardrails pass.

### Phase 4 - Optional Live Inference
If `--target-date` provided, run inference using the new station bundle.

### Phase 5 - Backtest
1. `single`: station-only
2. `cojoined`: one trade/day global arbitration across requested stations
3. Current locked default rule pack:
   - Entry gate: `entry_timestamp >= max(T-1 12:00:00Z, market_open_utc + 30m)`
   - Stake: `stake = min(balance_before * 0.065, 500)`

### Phase 6 - Run Manifest
Write `run_manifest.json` with:
1. Inputs and resolved metadata
2. Commands executed
3. SQLite paths and export paths
4. Bundle info + hashes
5. Inference/backtest summary/sanity paths

### Phase 7 - Explicit Documentation Output
Persist run notes and update MOS docs.

## Required Refactors Included
1. `tools/live/mos_blend12_bundle.py` created and wired.
2. `tools/live/mos_quantile_live_inference.py` refactored for dynamic station configs.
3. `ml/run_knyc_mos_first_plan.py` made station/timezone generic.
4. `backtesting/mos_blend12_knyc_kmia_cojoined_audit.py` made station-list + `single|cojoined` capable.
5. `tools/station_flow/station_metadata.py` resolver added.
6. `tools/station_flow/run_station_full_flow.py` orchestrator added.

## Acceptance Targets
1. Data bootstrap always runs before model training.
2. Canonical NWS/MOS station data persists in SQLite-first layout under `D:\Ahmed\data\sqlite`.
3. Exported model path matches live strategy family (`blend_12`, `gfs_12+nam_12`, `T-1 12:00Z`).
4. Bundle parity gate is pass/fail mandatory.
5. Backtest supports `single` and `cojoined`.
6. Run manifest is produced every run.

## Non-Negotiable Documentation Requirement
Absolutely **everything** must be deeply and explicitly documented, including:
1. Decisions, assumptions, fallback behavior, and fail-closed gates.
2. CLI inputs/defaults and output paths.
3. SQLite schemas and export contracts.
4. Leakage controls and parity checks.
5. Backtest rules, outputs, sanity interpretation, and reproducibility artifacts.
