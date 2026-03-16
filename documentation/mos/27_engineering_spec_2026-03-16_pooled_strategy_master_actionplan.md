# Pooled Strategy Master Action Plan

## Objective

Implement the full pooled MOS-first residual system described in the pooled-strategy brief with:

1. A station-universe backbone of 140 active stations plus 20 reserves.
2. One canonical SQLite system-of-record per station under `D:\Ahmed\data\sqlite\pooled_strategy\<STATION>\`.
3. Leakage-safe dataset construction using exact station-local date assignment and exact runtime policy.
4. Two-stage pooled modeling:
   - pooled residual slice models for `gfs_12` and `nam_12`
   - pooled out-of-fold meta-blend and calibration layer
5. Shared train/live feature code, bundle hashing, and strict runtime parity.

## Non-Negotiable Invariants

1. MOS remains the physical anchor. Base targets are residuals versus raw MOS Tmax, not direct weather from scratch.
2. Exact runtime policy is preserved. No silent fallback runtimes.
3. Station-local date and timezone handling must come from the registry, not hardcoded logic.
4. All climatology, bias priors, target encodings, and regime artifacts must be fold-safe.
5. WU and sidecar data are optional augmentation only until availability and leakage audits are passed.
6. Train and live must call the same feature builder and validate the same feature contract hash.

## Stage Breakdown

### Stage 1 - Canonical Station Data Backbone

Status:
Implemented in `tools/pooled_strategy/`.

Deliverables:
1. Programmatic station universe and registry resolution.
2. One SQLite DB per station with:
   - station registry
   - ingest runs and events
   - NWS truth tables
   - MOS raw and normalized tables
   - WU manifest and 30-minute observation tables
   - Kalshi minute-history tables
3. Compatibility exports back to:
   - `D:\Ahmed\data\kalshi\training_data\02_truth`
   - `D:\Ahmed\data\kalshi\training_data\04_mos\archive_merged`
4. Detailed per-station and per-run JSON summaries.

Acceptance:
1. Every station has its own folder and SQLite file.
2. Ingest is restartable and prefers existing local artifacts before network fetches.
3. WU blocks cleanly when no API key is available.
4. Registry crosswalk CSVs are written under `D:\Ahmed\data\sqlite\pooled_strategy\_registry`.

### Stage 2 - Station Catalog and Static Metadata Product

Deliverables:
1. Versioned station crosswalk:
   - `market_station_id`
   - `mos_station_id`
   - `truth_station_id`
   - `obs_station_id`
   - `display_name`
   - `timezone`
   - `lat`
   - `lon`
   - `elevation_m`
2. Static station feature set:
   - water exposure class
   - distance to saltwater / Great Lakes
   - primary coast-normal bearing
   - urban proxy
   - traded flag
   - regime family
   - health score
3. Reserve-station admission and replacement logic.

Acceptance:
1. No hand-maintained timezone list remains in pooled training/live code.
2. Alias-risk stations such as `KNYC` are explicit in the crosswalk.
3. The station universe is frozen and versioned per model release.

### Stage 3 - Canonical Normalized Training Tables

Deliverables:
1. Canonical base-slice row grain:
   - `station_id`
   - `target_date_local`
   - `required_runtime_utc`
   - `model_slice`
2. Canonical meta row grain:
   - same station/date/runtime key
   - OOF base predictions
   - disagreement features
   - priors
   - regime features
3. Exact local-day assignment logic using station IANA timezone.
4. Runtime-complete inclusion and exclusion rules.

Acceptance:
1. Rows are built only from the exact `T-1 12:00Z` runtime for each station/day.
2. Forecast times are assigned to local-day windows using the station timezone.
3. Duplicate runtime handling and payload conflict logging are explicit.

### Stage 4 - Feature Store Build

Deliverables:
1. Fold-safe daily normals.
2. Fold-safe hourly normals.
3. Station and family bias priors with shrinkage.
4. Feature families:
   - station metadata and priors
   - solar geometry
   - temp curve summaries
   - dewpoint and moisture
   - solar-weighted cloud burden
   - wind direction and onshore/offshore components
   - precipitation / thunder / visibility / ceiling
   - cross-model disagreement
   - revision features from prior runs
   - regime flags and cluster features
   - missingness and QC features
5. Explicit feature-family contracts and ablation toggles.

Acceptance:
1. Each feature family is produced by shared reusable builders.
2. Each feature is provably available under the runtime policy.
3. QC flags are emitted for sparse or suspect fields instead of silently coercing everything.

### Stage 5 - Pooled Model Training

Primary architecture:
1. Pooled `gfs_12` residual point model.
2. Pooled `gfs_12` residual quantile heads.
3. Pooled `nam_12` residual point model.
4. Pooled `nam_12` residual quantile heads.
5. OOF generation for both slices.
6. Pooled meta point model on residual versus simple slice consensus.
7. Pooled meta quantile heads.
8. Monotonic repair and pooled location-scale calibration layer.

Fallback architecture:
1. Single joint cross-slice pooled residual GBDT versus raw blended MOS.

Acceptance:
1. Base models do not rely heavily on raw station ID.
2. Meta and calibration layers may use station identity, but only after slice predictions exist.
3. No training artifact is built from non-OOF predictions.

### Stage 6 - Evaluation and Research Harness

Deliverables:
1. Rolling-origin folds for 2014-2023.
2. Locked final test on 2024-2025.
3. Core-traded station scorecards for:
   - `KNYC`
   - `KLAX`
   - `KMIA`
   - `KPHL`
4. Regime slice scorecards.
5. Universe-size sweep and station-ID ablation tests.
6. Family-wise SHAP and permutation analysis.

Acceptance:
1. Every experiment records exact feature families, universe version, and bundle inputs.
2. Claims always report traded-station aggregate and per-station results.
3. Calibration coverage, CRPS, MAE, and tail pinball are all present.

### Stage 7 - Live Bundle and Parity Layer

Deliverables:
1. Shared slice feature builder used by both training and live inference.
2. Bundle manifest with hashes for:
   - feature contract
   - station catalog
   - crosswalk
   - climatology store
   - parser versions
   - runtime policy
3. Runtime-proof and leakage-proof reports.
4. Drift diagnostics and station/date replay suite.

Acceptance:
1. Live inference refuses prediction if the exact runtime is missing.
2. Critical feature-family loss causes fail-closed output.
3. Quantiles are monotonic after calibration.

### Stage 8 - Side-Channel Promotion Pipeline

Deliverables:
1. WU field audit and availability classification:
   - Green
   - Yellow
   - Red
2. Raw MOS token audit and parser conflict catalog.
3. Optional promotion of Green-zone sidecar features into augmentation experiments only.

Acceptance:
1. No side-channel feature enters production without timestamp and leakage proof.
2. UV and sparse/corrupt WU fields remain quarantined until fixed.

### Stage 9 - Operations, Monitoring, and Retraining

Deliverables:
1. Slice-availability dashboard.
2. Station-health and source-health monitoring.
3. Drift alerts for:
   - missingness
   - disagreement
   - spread
   - parser changes
   - stale bulletin hashes
4. Fast retrain pathway for upstream MOS or service changes.

Acceptance:
1. Station universe changes are versioned, not ad hoc.
2. Source outages are surfaced as explicit degraded states.
3. Train/live parity failures block promotion.

## Implementation Order

1. Finish Stage 1 for the full universe.
2. Add station static-feature products and health scoring.
3. Build canonical base/meta dataset writers with exact local-day logic.
4. Implement fold-safe climatology and priors.
5. Build the shared feature family library.
6. Train pooled slice models and OOF pipelines.
7. Train meta and calibration layers.
8. Add evaluation harness and ablation runner.
9. Add bundle export, parity replay, and drift diagnostics.
10. Audit and optionally promote side channels.

## Immediate Next Commands

### Stage 1 full-universe backfill
```powershell
python tools/pooled_strategy/stage1_backfill.py `
  --stations all `
  --log-level INFO
```

### Stage 1 with Kalshi backfill enabled
```powershell
python tools/pooled_strategy/stage1_backfill.py `
  --stations all `
  --download-missing-kalshi `
  --log-level INFO
```

### Stage 1 scoped validation run
```powershell
python tools/pooled_strategy/stage1_backfill.py `
  --stations KNYC,KATL `
  --log-level INFO
```

## Current Known Blockers

1. `WEATHERCOM_API_KEY` is not available in the current environment, so WU downloads can only use existing local station SQLite caches.
2. Full-universe stage-1 truth and MOS backfill for stations without local artifacts will still take time and should be run with the low-concurrency defaults already baked into the stage-1 runner.
3. Stages 2 through 9 remain to be implemented after the stage-1 universe is materially populated.
