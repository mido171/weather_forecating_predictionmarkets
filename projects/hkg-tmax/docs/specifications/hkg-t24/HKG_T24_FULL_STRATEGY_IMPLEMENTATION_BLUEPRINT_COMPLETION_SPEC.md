# HKG T24 / H24N Full Strategy Implementation Blueprint — Completion Specification

**Document role:** mandatory implementation complement to `HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT.md`.

**Audience:** Codex implementation agent.

**Status:** directive implementation specification. This document contains the final defaults for the first full implementation. Codex must not make architecture, modelling, data-selection, validation, workflow, threshold, promotion, or schema decisions outside this specification.

**Primary target:** Hong Kong Observatory daily maximum temperature for target date `T`, in degrees Celsius.

**Primary decision cutoff:** `H24N`, meaning the forecast is made at **15:00 HKT on T-1**.

**Primary goal:** produce a leakage-safe, trustable, audited, out-of-fold-measured HKG Tmax forecasting system built from the existing database, the corrected near-continuous HKO official forecast archive, the tactical GribStream NWP backfill, and approved proxy/live-shadow sources.

---

## 0. Global directives

### 0.1 Mandatory language

All implementation requirements below are binding.

`MUST` means required.

`MUST NOT` means forbidden.

`STRICT` means eligible for the first deployable historical scoreboard.

`RESEARCH_PROXY` means useful for research and proxy scoreboards but not eligible for the strict deployable leaderboard.

`LIVE_SHADOW` means collected and scored prospectively but not allowed to affect strict production forecasts until the promotion gate is passed.

`BLOCKED` means stored only for provenance/diagnostics and never used as a Tmax predictor in the first full implementation.

### 0.2 Conservative default rule

If a source, column, timestamp, row, model output, publication timing, station field, or feature fails an eligibility check, Codex MUST fail closed:

```text
row is excluded from strict features
reason is recorded in audit table
prediction proceeds only if fallback experts are available
```

Codex MUST NOT infer availability from archive presence alone.

Codex MUST NOT use target-date outcome fields, same-row residual fields, target-day finalized climate rows, post-cutoff model runs, or post-cutoff observations.

### 0.3 H24N time convention

All database storage and model logic MUST use UTC timestamps internally.

All human reporting may display HKT alongside UTC.

For target date `target_date_hkt`:

```sql
formal_cutoff_utc = ((target_date_hkt::date - INTERVAL '1 day') + TIME '15:00') AT TIME ZONE 'Asia/Hong_Kong'
operational_freeze_utc = formal_cutoff_utc - INTERVAL '15 minutes'
```

For GribStream historical rows, the strict first implementation MUST additionally apply the project inventory's conservative publication/indexing buffer:

```sql
run_time_utc + INTERVAL '6 hours' <= formal_cutoff_utc
```

The NWP `6 hour` buffer is used for historical strict NWP extraction. Prospective first-seen collection may supersede that buffer only when `first_seen_at_utc <= operational_freeze_utc` is recorded.

### 0.4 Strict target-memory correction

At `15:00 HKT on T-1`, the finalized daily Tmax for `T-1` is not assumed known. Therefore, finalized daily target-label memory MUST use **T-2 or older**.

`target_lag1_final_tmax_c` is forbidden in the first full implementation.

`target_lag2_tmax_c` is the latest allowed finalized daily HKO Tmax lag.

T-1 intraday temperature/high-frequency features are excluded from first strict implementation unless an exact-vintage intraday HKO feed with first-seen timestamps exists and passes a separate live-shadow promotion gate. The existing high-frequency/live feeds remain outside the first strict historical model.

---

## 1. Definition of "full implementation complete"

The first full implementation is complete only when all required artifacts in this section exist and all blocking phases pass.

### 1.1 Included in first full implementation

The first full implementation MUST include:

1. Source registry, schema verification, and source eligibility audit.
2. H24N canonical snapshot builder.
3. Strict target labels and target-memory feature builder using T-2 or older daily labels.
4. HKO official forecast anchor, revision features, and official residual-memory state using corrected `public.hko_historical_forecasts_2000_2026`.
5. GribStream NWP feature builder for the tactical fetched datasets, using only `full_tactical_backfill_ok_tmax` rows and the H24N leakage-safe filter.
6. Strict expert models:
   - `E0_OFFICIAL_RAW`
   - `E1_OFFICIAL_RESIDUAL`
   - `E2_TARGET_MEMORY`
   - `E4_GFS_MOS`
   - `E5_GEFS_PROB_MOS`
7. Research-proxy expert models:
   - `E3_STATION_PROXY`, only in proxy scoreboards.
   - `E10_DIAGNOSTIC_PROXY`, only if built from allowed lagged/proxy mechanisms and never from blocked same-day diagnostic values.
8. Shadow/capped challenger expert models:
   - `E6_IFS_OPER_SHADOW`
   - `E7_IFS_ENS_SHADOW`
   - `E8_AI_CHALLENGERS_SHADOW`
   - `E9_CWA_WRF_LIVE_SHADOW`
   - `E11_ARWF_LIVE_SHADOW` if ARWF exact-vintage data exists.
9. Genuine out-of-fold prediction factory for every eligible expert on its permitted rows.
10. Routers:
    - `R0_OFFICIAL_LONG_HISTORY`
    - `R1_CORE_GFS_GEFS`
    - `R2_IFS_SHADOW_ADAPTER`
    - `R3_AI_SHADOW_ADAPTER`
    - `R4_LIVE_SHADOW_ADAPTER`
11. Specialist system:
    - marine suppression
    - weak-wind heat buildup
    - MAM transition
    - cloud/rain suppression
    - dry subsidence/ridge heating
    - high-error tail prevention
12. Distributional layer for P10/P25/P50/P75/P90, expected absolute error, threshold probabilities, confidence state, and no-trade flag.
13. Strict development OOF scoreboard through `2023-12-31`.
14. Research-proxy OOF scoreboard through `2023-12-31`.
15. Shadow feature/prediction coverage report for 2024+ sources without using sealed target outcomes.
16. Negative-control and leakage-test report.
17. Frozen candidate manifest produced from pre-2024 development only.
18. Sealed validation code and commands implemented but not executed by default.
19. Live/replay inference command implemented. Scheduling is out of scope.

### 1.2 Not included by default

The first full implementation MUST NOT automatically open sealed target outcomes for 2024 or 2025.

The first full implementation MUST end with:

```text
status = READY_FOR_SEALED_VALIDATION
```

Sealed validation is a separate explicit command that requires:

```text
--open-sealed --sealed-release-token <token>
```

### 1.3 Required final done condition

The implementation is complete only when these files and tables exist:

```text
reports/source_registry.md
reports/schema_contract_report.md
reports/leakage_audit.md
reports/h24n_snapshot_coverage.md
reports/feature_dictionary.md
reports/feature_availability_matrix.md
reports/expert_oof_scoreboard_strict.csv
reports/expert_oof_scoreboard_proxy.csv
reports/router_scoreboard_strict.csv
reports/specialist_scoreboard_strict.csv
reports/system_scoreboard_strict.csv
reports/system_scoreboard_proxy.csv
reports/ablation_matrix.csv
reports/negative_control_report.md
reports/ready_for_sealed_validation.md
artifacts/frozen_candidate_manifest.json
artifacts/final_system_config.yaml
```

and the following database tables contain non-empty, validated rows:

```text
model_core.run_manifest
model_core.source_registry
model_features.h24n_snapshot
model_features.official_features
model_features.target_memory_features
model_features.nwp_daily_features
model_features.snapshot_feature_matrix_strict
model_oof.expert_prediction
model_oof.system_prediction
model_validation.scoreboard
model_validation.negative_control_result
```

---

## 2. Exact implementation order

Codex MUST implement and run the phases in the order below. A blocking phase failure MUST stop the pipeline.

### Phase 0 — Repository, environment, and source verification

**Purpose:** verify the database, source tables, row counts, and required docs before building features.

**Inputs:**

```text
PostgreSQL connection from environment variable HKG_TMAX_DB_DSN
GRIBSTREAM_FETCHED_DATA_INVENTORY_20260626.md
HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT.md
this completion specification
```

**Scripts/modules:**

```text
src/hkg_tmax/cli.py
src/hkg_tmax/config.py
src/hkg_tmax/db.py
src/hkg_tmax/audit/source_registry.py
src/hkg_tmax/audit/schema_contracts.py
```

**CLI:**

```bash
python -m hkg_tmax.cli phase0-verify-sources --cutoff-id H24N
```

**Outputs:**

```text
reports/source_registry.md
reports/schema_contract_report.md
reports/phase0_validation.json
model_core.source_registry
model_core.run_manifest
```

**Pass condition:**

- database connection succeeds;
- all required strict tables are present or discoverable by the specified contracts;
- `public.hko_historical_forecasts_2000_2026` has at least `100000` rows with `row_quality_status='usable_local_minmax'` and `product_type='local'`;
- `nwp_tactical.forecast_wide` and `nwp_tactical.raw_response_object` exist;
- the full-run GribStream source filter returns at least `1,900,000` rows;
- no table has ambiguous date key columns after schema contract resolution.

**Blocks later phases:** yes.

### Phase 1 — Create new schemas and tables

**Purpose:** create the permanent, rebuildable, and audit tables defined in Section 5.

**Inputs:** Phase 0 schema contracts.

**Scripts/modules:**

```text
src/hkg_tmax/db/ddl.py
sql/create_model_schemas.sql
```

**CLI:**

```bash
python -m hkg_tmax.cli phase1-create-schema --if-exists keep
```

**Outputs:** all schemas/tables in Section 5.

**Pass condition:**

- all schemas exist;
- DDL idempotency test passes;
- every table has primary key, required indexes, and provenance columns.

**Blocks later phases:** yes.

### Phase 2 — Build H24N cutoff calendar and target labels

**Purpose:** construct canonical target dates, H24N cutoff timestamps, and settled labels.

**Inputs:** target label table, official archive target-date range.

**Scripts/modules:**

```text
src/hkg_tmax/features/cutoff_calendar.py
src/hkg_tmax/features/target_labels.py
```

**CLI:**

```bash
python -m hkg_tmax.cli phase2-build-target-calendar --cutoff-id H24N --start-date 2000-01-02 --end-date 2026-06-21
```

**Outputs:**

```text
model_core.cutoff_calendar
model_core.target_label
reports/target_label_coverage.md
```

**Pass condition:**

- at least `9660` distinct target dates from `2000-01-02` to `2026-06-21` are represented;
- the known missing official target date `2003-02-02` is recorded as an official-anchor gap, not a target-label failure if label exists;
- every target date has formal cutoff UTC and freeze UTC;
- target labels are in degrees Celsius.

**Blocks later phases:** yes.

### Phase 3 — Build official forecast anchor and official features

**Purpose:** select the latest eligible official HKO local min/max forecast before the H24N freeze and compute revision/context features.

**Inputs:**

```text
public.hko_historical_forecasts_2000_2026
model_core.cutoff_calendar
model_core.target_label
```

**Scripts/modules:**

```text
src/hkg_tmax/features/official_anchor.py
src/hkg_tmax/features/official_text.py
src/hkg_tmax/features/online_state.py
```

**CLI:**

```bash
python -m hkg_tmax.cli phase3-build-official-features --cutoff-id H24N --end-date 2026-06-21
```

**Outputs:**

```text
model_features.official_features
model_features.official_revision_features
model_features.online_residual_state_seed
reports/official_anchor_coverage.md
reports/official_revision_feature_dictionary.md
```

**Pass condition:**

- no official anchor row has `issue_at_utc > operational_freeze_utc`;
- no official row has invalid Tmax outside `[5, 45]` °C;
- anchor coverage for `2000-01-02` through `2026-06-21` is at least `99.9%` excluding documented missing dates;
- multiple eligible pre-cutoff rows are resolved to exactly one latest anchor per target date.

**Blocks later phases:** yes for official models, no for NWP-only feature diagnostics.

### Phase 4 — Build target-memory features

**Purpose:** build causal HKO target-memory features using only finalized daily labels from `T-2` or older.

**Inputs:**

```text
model_core.cutoff_calendar
model_core.target_label
```

**Scripts/modules:**

```text
src/hkg_tmax/features/target_memory.py
```

**CLI:**

```bash
python -m hkg_tmax.cli phase4-build-target-memory --cutoff-id H24N
```

**Outputs:**

```text
model_features.target_memory_features
reports/target_memory_feature_dictionary.md
```

**Pass condition:**

- no feature references `target_date >= T-1` finalized daily label;
- earliest rows with insufficient history have missing feature values and explicit missingness flags;
- rolling windows match formulas in Section 8;
- leakage test confirms shifting target labels by one day changes the feature matrix.

**Blocks later phases:** yes.

### Phase 5 — Build GribStream NWP daily features

**Purpose:** convert tactical GribStream forecast rows into daily model features under H24N strict filtering.

**Inputs:**

```text
nwp_tactical.forecast_wide
nwp_tactical.raw_response_object
model_core.cutoff_calendar
GRIBSTREAM_FETCHED_DATA_INVENTORY_20260626.md
```

**Scripts/modules:**

```text
src/hkg_tmax/features/nwp_filter.py
src/hkg_tmax/features/nwp_daily.py
src/hkg_tmax/features/nwp_ensemble.py
src/hkg_tmax/features/nwp_spatial.py
```

**CLI:**

```bash
python -m hkg_tmax.cli phase5-build-nwp-features --cutoff-id H24N --source-scope full_tactical_backfill_ok_tmax
```

**Outputs:**

```text
model_features.nwp_safe_row_ledger
model_features.nwp_daily_features
model_features.nwp_ensemble_features
reports/nwp_feature_coverage.md
reports/nwp_leakage_filter_report.md
reports/nwp_feature_dictionary.md
```

**Pass condition:**

- raw smoke rows outside `full_tactical_backfill_ok_tmax` are excluded;
- `run_time_utc + interval '6 hours' <= formal_cutoff_utc` for every strict NWP feature row;
- `nbmoc`, `aigfspres`, and `aigefssfc` are excluded from Tmax source features;
- `ifsenfo` member-0 gap is flagged but days with at least 50 members remain usable;
- `fourcastnetgfs` has no generated features after its observed archive end;
- every generated feature carries source dataset code and availability status.

**Blocks later phases:** yes for R1/R2/R3, no for R0.

### Phase 6 — Build station and diagnostic proxy feature tables

**Purpose:** build proxy features from station/daily-climate/IGRA/TC/static sources without allowing them into strict deployable scoreboards.

**Inputs:**

```text
NOAA ISD station tables discovered by contract
HKO daily climate table discovered by contract
IGRA tables discovered by contract
TC best track table discovered by contract
static geospatial inventory discovered by contract
```

**Scripts/modules:**

```text
src/hkg_tmax/features/station_proxy.py
src/hkg_tmax/features/diagnostic_proxy.py
src/hkg_tmax/features/static_geography.py
```

**CLI:**

```bash
python -m hkg_tmax.cli phase6-build-proxy-features --cutoff-id H24N
```

**Outputs:**

```text
model_features.station_proxy_features
model_features.diagnostic_proxy_features
model_features.static_geospatial_features
reports/station_proxy_feature_dictionary.md
reports/proxy_source_eligibility_report.md
```

**Pass condition:**

- station wind-direction fields known to be constant `20` degrees are not used;
- ISD/IGRA/HKO daily climate finalized same-day values are not used as strict predictors;
- every proxy feature has `strict_allowed=false`;
- proxy features appear only in proxy feature matrices and proxy scoreboards.

**Blocks later phases:** no for strict system, yes for proxy system.

### Phase 7 — Build feature matrices

**Purpose:** join target dates, labels, snapshots, official features, target-memory features, NWP features, proxy features, and availability masks.

**Inputs:** Phase 2 through Phase 6 outputs.

**Scripts/modules:**

```text
src/hkg_tmax/features/matrix_builder.py
```

**CLI:**

```bash
python -m hkg_tmax.cli phase7-build-feature-matrices --cutoff-id H24N
```

**Outputs:**

```text
model_features.h24n_snapshot
model_features.snapshot_feature_matrix_strict
model_features.snapshot_feature_matrix_proxy
model_features.feature_availability_matrix
reports/feature_availability_matrix.md
reports/feature_dictionary.md
```

**Pass condition:**

- strict matrix contains no proxy-only columns;
- proxy matrix contains explicit proxy flags;
- rows are unique by `(snapshot_id)`;
- target label is separated from feature columns;
- every feature has provenance and availability grade.

**Blocks later phases:** yes.

### Phase 8 — Train experts and generate OOF predictions

**Purpose:** train every expert according to Section 13 and write OOF predictions.

**Inputs:** feature matrices.

**Scripts/modules:**

```text
src/hkg_tmax/models/experts.py
src/hkg_tmax/models/oof.py
src/hkg_tmax/models/preprocessing.py
```

**CLI:**

```bash
python -m hkg_tmax.cli phase8-train-experts-oof --cutoff-id H24N --mode strict_and_proxy
```

**Outputs:**

```text
model_oof.expert_prediction
model_oof.expert_artifact
model_oof.expert_scoreboard
reports/expert_oof_scoreboard_strict.csv
reports/expert_oof_scoreboard_proxy.csv
```

**Pass condition:**

- every OOF prediction row was generated by a model trained on strictly earlier target dates;
- preprocessing artifacts are fold-local;
- strict experts meet coverage requirements;
- failed shadow experts are marked unavailable and do not stop strict training.

**Blocks later phases:** yes.

### Phase 9 — Train routers

**Purpose:** train R0/R1/R2/R3/R4 routers using only OOF expert predictions and context features.

**Inputs:**

```text
model_oof.expert_prediction
model_features.snapshot_feature_matrix_strict
model_features.snapshot_feature_matrix_proxy
```

**Scripts/modules:**

```text
src/hkg_tmax/models/router.py
src/hkg_tmax/models/static_weights.py
src/hkg_tmax/models/expected_error.py
```

**CLI:**

```bash
python -m hkg_tmax.cli phase9-train-routers --cutoff-id H24N
```

**Outputs:**

```text
model_router.router_prediction
model_router.router_weight
model_router.router_scoreboard
reports/router_scoreboard_strict.csv
reports/router_scoreboard_proxy.csv
```

**Pass condition:**

- no in-sample expert predictions enter router training;
- R1 is trained only on rows where official, GFS, and GEFS core experts have OOF predictions;
- short-history challengers have capped or zero weights according to Section 15;
- router beats the relevant static blend by at least the promotion threshold or is demoted to static blend.

**Blocks later phases:** yes.

### Phase 10 — Train specialists

**Purpose:** train specialist detectors, corrections, and benefit gates.

**Inputs:** router OOF predictions, feature matrices, expert predictions.

**Scripts/modules:**

```text
src/hkg_tmax/models/specialists.py
```

**CLI:**

```bash
python -m hkg_tmax.cli phase10-train-specialists --cutoff-id H24N
```

**Outputs:**

```text
model_router.specialist_prediction
model_router.specialist_scoreboard
reports/specialist_scoreboard_strict.csv
reports/specialist_scoreboard_proxy.csv
```

**Pass condition:**

- each promoted specialist passes support, lift, and no-harm gates;
- non-promoted specialists output zero correction in final strict formula;
- total specialist correction cap is enforced.

**Blocks later phases:** yes.

### Phase 11 — Train distributional layer

**Purpose:** produce calibrated prediction intervals, expected absolute error, threshold probabilities, confidence, and no-trade flag.

**Inputs:** final OOF system predictions before distributional correction.

**Scripts/modules:**

```text
src/hkg_tmax/models/distribution.py
```

**CLI:**

```bash
python -m hkg_tmax.cli phase11-train-distribution --cutoff-id H24N
```

**Outputs:**

```text
model_validation.distribution_prediction
reports/distribution_scoreboard.csv
reports/calibration_report.md
```

**Pass condition:**

- P10/P25/P50/P75/P90 are monotonic;
- P50 MAE does not worsen strict final point MAE by more than `0.005°C`;
- threshold probability calibration Brier score beats climatological probabilities on development rows by at least `1%` relative.

**Blocks later phases:** yes.

### Phase 12 — Build final system predictions and ablations

**Purpose:** run full historical replay and compare components.

**Inputs:** all previous outputs.

**Scripts/modules:**

```text
src/hkg_tmax/validation/scoreboard.py
src/hkg_tmax/validation/ablation.py
```

**CLI:**

```bash
python -m hkg_tmax.cli phase12-score-system --cutoff-id H24N --end-date 2023-12-31
```

**Outputs:**

```text
model_oof.system_prediction
model_validation.scoreboard
reports/system_scoreboard_strict.csv
reports/system_scoreboard_proxy.csv
reports/ablation_matrix.csv
```

**Pass condition:**

- strict final system is scored against official raw on identical rows;
- all required metrics and slices exist;
- no source-specific row count mismatch appears in same-row comparisons.

**Blocks later phases:** yes.

### Phase 13 — Negative controls and leakage tests

**Purpose:** prove the system is not winning through leakage or accidental outcome fields.

**Inputs:** feature matrices, predictions, schema contracts.

**Scripts/modules:**

```text
src/hkg_tmax/validation/leakage_tests.py
src/hkg_tmax/validation/negative_controls.py
```

**CLI:**

```bash
python -m hkg_tmax.cli phase13-run-leakage-tests --cutoff-id H24N
```

**Outputs:**

```text
model_validation.negative_control_result
model_validation.leakage_audit_event
reports/negative_control_report.md
reports/leakage_audit.md
```

**Pass condition:** all Section 19 tests pass.

**Blocks later phases:** yes.

### Phase 14 — Freeze candidate and create live/replay command

**Purpose:** write immutable config for the pre-2024 development champion and implement live/replay prediction.

**Inputs:** Phase 12 and Phase 13 outputs.

**Scripts/modules:**

```text
src/hkg_tmax/live/inference.py
src/hkg_tmax/live/online_state.py
src/hkg_tmax/artifacts/freeze.py
```

**CLI:**

```bash
python -m hkg_tmax.cli phase14-freeze-candidate --cutoff-id H24N --candidate-source pre2024_oof
python -m hkg_tmax.cli predict-replay --cutoff-id H24N --target-date 2023-12-31
```

**Outputs:**

```text
artifacts/frozen_candidate_manifest.json
artifacts/final_system_config.yaml
reports/ready_for_sealed_validation.md
model_live.prediction
```

**Pass condition:**

- candidate manifest includes git hash, data hashes, table row counts, feature list, model artifacts, random seeds, and scoreboards;
- replay command produces one prediction JSON for a known historical date without reading target label until score mode;
- status is `READY_FOR_SEALED_VALIDATION`.

**Blocks later phases:** no.

### Phase 15 — Sealed validation commands

**Purpose:** implement but not run by default.

**CLI:**

```bash
python -m hkg_tmax.cli sealed-score --cutoff-id H24N --year 2024 --open-sealed --sealed-release-token <token>
python -m hkg_tmax.cli sealed-score --cutoff-id H24N --year 2025 --open-sealed --sealed-release-token <token>
```

**Pass condition for command implementation:** guard rejects execution without both `--open-sealed` and a non-empty token.

**Blocks first full implementation:** no.

---

## 3. Repository and code layout

Codex MUST create or conform to this package layout. If an existing `src/` package exists, Codex MUST place these modules under the existing package root while preserving the submodule names below.

```text
src/hkg_tmax/
  __init__.py
  cli.py
  config.py
  db.py
  timeutils.py
  constants.py

  audit/
    __init__.py
    source_registry.py
    schema_contracts.py
    leakage_events.py
    reports.py

  db/
    __init__.py
    ddl.py
    migrations.py

  features/
    __init__.py
    cutoff_calendar.py
    target_labels.py
    official_anchor.py
    official_text.py
    target_memory.py
    nwp_filter.py
    nwp_daily.py
    nwp_ensemble.py
    nwp_spatial.py
    station_proxy.py
    diagnostic_proxy.py
    static_geography.py
    matrix_builder.py
    feature_dictionary.py

  models/
    __init__.py
    preprocessing.py
    folds.py
    experts.py
    oof.py
    static_weights.py
    expected_error.py
    router.py
    specialists.py
    distribution.py
    artifact_store.py

  validation/
    __init__.py
    metrics.py
    scoreboard.py
    slices.py
    ablation.py
    leakage_tests.py
    negative_controls.py
    sealed.py

  live/
    __init__.py
    inference.py
    online_state.py
    post_settlement.py

  utils/
    __init__.py
    sql.py
    hashing.py
    json.py
    io.py

sql/
  create_model_schemas.sql
  source_contract_queries.sql
  h24n_snapshot_views.sql
  nwp_safe_row_filter.sql

tests/
  unit/
    test_timeutils.py
    test_target_memory.py
    test_metrics.py
    test_nwp_units.py
  integration/
    test_schema_contracts.py
    test_h24n_snapshot.py
    test_official_anchor.py
    test_nwp_filter.py
    test_oof_integrity.py
    test_router_weights.py
    test_specialist_caps.py
    test_sealed_guard.py
  smoke/
    test_full_pipeline_smoke.py

reports/
artifacts/
configs/
  h24n_system_config.yaml
  model_hyperparameters.yaml
  feature_whitelist_strict.yaml
  feature_whitelist_proxy.yaml
```

### 3.1 Required CLI commands

The CLI MUST expose these commands exactly:

```text
phase0-verify-sources
phase1-create-schema
phase2-build-target-calendar
phase3-build-official-features
phase4-build-target-memory
phase5-build-nwp-features
phase6-build-proxy-features
phase7-build-feature-matrices
phase8-train-experts-oof
phase9-train-routers
phase10-train-specialists
phase11-train-distribution
phase12-score-system
phase13-run-leakage-tests
phase14-freeze-candidate
run-full-pre2024
predict-replay
predict-live
score-prediction
sealed-score
```

The command `run-full-pre2024` MUST execute phases 0 through 14 in order and stop at the first failure.

---

## 4. Existing source-of-truth tables and schema contracts

Codex MUST resolve source tables using the exact priority below. If the primary table name exists, use it. If it does not exist, use the discovery contract. If the discovery contract returns zero or multiple ambiguous tables, Phase 0 fails.

### 4.1 HKO target Tmax labels

**Primary table:**

```text
public.hko_daily_tmax_target_labels
```

**Required columns:**

| Column | Type contract | Required | Meaning |
|---|---|---:|---|
| `local_date` | date or castable text date | yes | HKT target date |
| `target_tmax_c` | numeric | yes | settled HKO daily Tmax in °C |
| `target_station` | text | no | expected `Hong Kong Observatory` |
| `target_source_id` | text | no | source provenance |
| `raw_retrieved_at_utc` | timestamptz or text timestamp | no | archive retrieval timestamp |
| `availability_tier` | text | no | expected `TARGET_ONLY` |
| `operational_input_allowed` | boolean | no | expected false |

**Discovery contract if primary table absent:**

Codex MUST run:

```sql
SELECT table_schema, table_name
FROM information_schema.columns
WHERE column_name IN ('target_tmax_c', 'local_date')
GROUP BY table_schema, table_name
HAVING COUNT(DISTINCT column_name) = 2;
```

The result MUST contain exactly one table whose `target_tmax_c` range is within `[0, 45]` °C and whose date range starts before `1900-01-01`.

### 4.2 HKO official forecast archive

**Primary table:**

```text
public.hko_historical_forecasts_2000_2026
```

**Required columns:**

| Column | Type contract | Required | Meaning |
|---|---|---:|---|
| `target_date` | date | yes | HKT target date forecasted |
| `issue_at_utc` | timestamptz | yes | issue timestamp UTC |
| `issue_at_hkt` | timestamptz or timestamp | yes | issue timestamp HKT or equivalent |
| `product_type` | text | yes | use only `local` |
| `row_quality_status` | text | yes | use only `usable_local_minmax` |
| `forecast_min_c` | numeric | yes | official min °C |
| `forecast_max_c` | numeric | yes | official max °C |
| `forecast_range_c` | numeric | no | max-min °C; recompute if absent |
| `forecast_midpoint_c` | numeric | no | midpoint °C; recompute if absent |
| `target_issue_lead_days` | numeric | no | expected 0 or 1 in raw archive |
| `forecast_text` | text | no | free text forecast |
| `weather_text` | text | no | free text weather description |
| `wind_text` | text | no | free text wind description |
| `humidity_min_pct` | numeric | no | forecast RH min |
| `humidity_max_pct` | numeric | no | forecast RH max |
| `source_url` | text | no | provenance |
| `source_hash` | text | no | provenance |
| `parser_version` | text | no | parser lineage |

**Expected corrected clean subset:**

```text
row_quality_status = 'usable_local_minmax'
product_type = 'local'
rows >= 115000
distinct target dates >= 9660
issue range begins 2000-01-01
forecast_max_c in [7, 39] in corrected clean subset
```

### 4.3 GribStream forecast tables

**Primary tables:**

```text
nwp_tactical.forecast_wide
nwp_tactical.raw_response_object
nwp_tactical.acquisition_chunk
nwp_tactical.validation_issue
```

**`nwp_tactical.forecast_wide` required columns:**

```text
dataset_code
acquisition_version
target_date_hkt
cutoff_id
run_time_utc
valid_time_utc
lead_hours
location_code
requested_latitude
requested_longitude
returned_latitude
returned_longitude
returned_grid_distance_km
member_number
raw_values_jsonb
source_response_object_id
quality_status
temperature_2m_k
interval_tmax_2m_k
dewpoint_2m_k
relative_humidity_2m_pct
u_wind_10m_mps
v_wind_10m_mps
mslp_pa
low_cloud_pct
accumulated_precip_kg_m2
downward_shortwave_w_m2
net_shortwave_w_m2
total_precip_m
shortwave_down_j_m2
total_column_water_vapour_kg_m2
pwat_kg_m2
temperature_925_k
temperature_850_k
relative_humidity_700_pct
geopotential_height_500_m
```

Columns that are absent MUST be created as NULL projection columns in a view named:

```text
model_features.v_nwp_forecast_wide_compat
```

**`nwp_tactical.raw_response_object` required columns:**

```text
response_object_id
object_uri
row_count
byte_size
response_sha256
created_at_utc
```

The exact column names for byte/hash timestamps may differ. If absent, Codex MUST create compatibility view `model_features.v_raw_response_object_compat` with these names. Missing `response_sha256` is a Phase 5 warning, not a Phase 0 failure, if raw objects exist and `response_object_id` joins to `forecast_wide`.

### 4.4 NOAA ISD station observations

**Primary tables:**

```text
public.noaa_isd_core_observations
public.noaa_isd_station_day_cutoff_summary
```

**Required columns for `noaa_isd_core_observations`:**

```text
station_id
observed_at_utc
observed_at_hkt
report_type
latitude
longitude
elevation_m
wind_direction_deg
wind_speed_mps
air_temperature_c
dew_point_c
sea_level_pressure_hpa
temperature_quality_code
dew_point_quality_code
sea_level_pressure_quality_code
source_time_policy
availability_tier
operational_input_allowed
```

**Required columns for `noaa_isd_station_day_cutoff_summary`:**

```text
station_id
local_date
obs_count
latest_before_1500_hkt
air_temperature_c_latest_before_1500
dew_point_c_latest_before_1500
sea_level_pressure_hpa_latest_before_1500
wind_direction_deg_latest_before_1500
wind_speed_mps_latest_before_1500
daily_air_temperature_min_c
daily_air_temperature_max_c
availability_tier
operational_input_allowed
```

**Use status:** `RESEARCH_PROXY` only in first full implementation.

**Forbidden field:** `wind_direction_deg` and `wind_direction_deg_latest_before_1500` are forbidden because the profiled dataset shows constant `20` degree values. Wind speed remains allowed as proxy. Wind direction may be promoted only after a separate repair artifact proves non-constant, source-derived direction.

### 4.5 HKO daily climate elements

**Primary table:**

```text
public.hko_daily_climate_elements
```

**Required columns:**

```text
source_id
station_or_domain
variable
unit
local_date
year
month
day
value
value_precision
completeness
parse_issue
availability_tier
operational_input_allowed
source_time_policy
```

**Use status:** `RESEARCH_PROXY` and `DIAGNOSTIC_ONLY`. Same-day values and target-day values are forbidden. Lagged values may enter proxy scoreboards only when lagged by at least 2 days and labelled proxy.

### 4.6 IGRA upper-air

**Primary tables:**

```text
public.noaa_igra_hkm00045004_key_pressure_levels
public.noaa_igra_hkm00045004_sounding_features
```

**Required columns:**

```text
station_id
valid_at_utc
valid_at_hkt
nominal_hour_utc
latitude
longitude
availability_tier
operational_input_allowed
release_latency_proven
source_time_policy
pressure_hpa
temperature_c
relative_humidity_pct
dewpoint_depression_c
wind_direction_deg
wind_speed_mps
pressure_level_tag
```

**Use status:** `DIAGNOSTIC_ONLY` in first full implementation. IGRA fields MUST NOT enter strict or proxy production feature matrices unless a cleaned rebuilt table and release-latency proof exists. Existing sentinel-contaminated values remain blocked.

### 4.7 Tropical cyclone best track

**Primary table:**

```text
public.hko_tropical_cyclone_best_track
```

**Required columns:**

```text
valid_at_utc
valid_at_hkt
latitude
longitude
intensity_or_wind_field columns if present
```

**Use status:** `DIAGNOSTIC_ONLY`. It may appear in diagnostic reports explaining high-error cases but not in first strict/proxy model features.

### 4.8 ARWF station forecasts

**Primary table:**

```text
public.hko_arwf_station_daily_forecasts
```

**Required columns:**

```text
raw_retrieved_at_utc
forecast_date
station_or_location
forecast_temperature columns if present
forecast_humidity columns if present
forecast_wind columns if present
```

**Use status:** `LIVE_SHADOW`. Current history is too short for first strict historical training. If table is absent, create live-shadow placeholder masks.

### 4.9 Static geospatial inventory

**Primary table:**

```text
public.static_geospatial_package_inventory
```

**Use status:** may produce deterministic static station context features in proxy matrices. It does not create weather predictors by itself.

### 4.10 Experiment outputs

Any historical experiment-output tables or parquet files are `RESEARCH_EVIDENCE_ONLY`. They MUST NOT be used as model features in the first implementation. Their scoreboards may be cited in reports, but old feature matrices and old predictions are not source-of-truth inputs.

---

## 5. New database schemas and tables

Codex MUST create the following schemas:

```sql
CREATE SCHEMA IF NOT EXISTS model_core;
CREATE SCHEMA IF NOT EXISTS model_features;
CREATE SCHEMA IF NOT EXISTS model_oof;
CREATE SCHEMA IF NOT EXISTS model_router;
CREATE SCHEMA IF NOT EXISTS model_validation;
CREATE SCHEMA IF NOT EXISTS model_live;
CREATE SCHEMA IF NOT EXISTS model_audit;
```

All tables MUST include:

```text
created_at_utc timestamptz not null default now()
updated_at_utc timestamptz not null default now()
run_id text not null
```

### 5.1 `model_core.run_manifest`

Permanent table.

```sql
CREATE TABLE IF NOT EXISTS model_core.run_manifest (
  run_id text PRIMARY KEY,
  run_kind text NOT NULL,
  cutoff_id text NOT NULL,
  started_at_utc timestamptz NOT NULL DEFAULT now(),
  ended_at_utc timestamptz NULL,
  status text NOT NULL,
  git_commit text NULL,
  code_version text NOT NULL,
  config_sha256 text NOT NULL,
  db_dsn_hash text NOT NULL,
  notes text NULL,
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  updated_at_utc timestamptz NOT NULL DEFAULT now()
);
```

### 5.2 `model_core.source_registry`

Permanent table.

```sql
CREATE TABLE IF NOT EXISTS model_core.source_registry (
  source_key text PRIMARY KEY,
  source_family text NOT NULL,
  table_schema text NULL,
  table_name text NULL,
  source_file text NULL,
  strict_status text NOT NULL,
  first_data_date date NULL,
  last_data_date date NULL,
  row_count bigint NULL,
  eligibility_rule text NOT NULL,
  blocker text NULL,
  run_id text NOT NULL REFERENCES model_core.run_manifest(run_id),
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  updated_at_utc timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_source_registry_family ON model_core.source_registry(source_family);
```

### 5.3 `model_core.cutoff_calendar`

Rebuildable permanent table.

```sql
CREATE TABLE IF NOT EXISTS model_core.cutoff_calendar (
  cutoff_id text NOT NULL,
  target_date_hkt date NOT NULL,
  formal_cutoff_utc timestamptz NOT NULL,
  operational_freeze_utc timestamptz NOT NULL,
  hkt_target_start_utc timestamptz NOT NULL,
  hkt_target_end_utc timestamptz NOT NULL,
  development_partition text NOT NULL,
  sealed_status text NOT NULL,
  run_id text NOT NULL REFERENCES model_core.run_manifest(run_id),
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  updated_at_utc timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (cutoff_id, target_date_hkt)
);
CREATE INDEX IF NOT EXISTS idx_cutoff_calendar_partition ON model_core.cutoff_calendar(development_partition);
```

### 5.4 `model_core.target_label`

Rebuildable permanent table.

```sql
CREATE TABLE IF NOT EXISTS model_core.target_label (
  target_date_hkt date PRIMARY KEY,
  target_tmax_c double precision NOT NULL,
  target_station text NOT NULL DEFAULT 'Hong Kong Observatory',
  source_table text NOT NULL,
  source_row_hash text NULL,
  label_visible_for_development boolean NOT NULL,
  sealed_status text NOT NULL,
  run_id text NOT NULL REFERENCES model_core.run_manifest(run_id),
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  updated_at_utc timestamptz NOT NULL DEFAULT now()
);
```

### 5.5 `model_features.h24n_snapshot`

Permanent rebuildable table.

```sql
CREATE TABLE IF NOT EXISTS model_features.h24n_snapshot (
  snapshot_id text PRIMARY KEY,
  cutoff_id text NOT NULL,
  target_date_hkt date NOT NULL,
  formal_cutoff_utc timestamptz NOT NULL,
  operational_freeze_utc timestamptz NOT NULL,
  strict_feature_available boolean NOT NULL,
  official_available boolean NOT NULL,
  gfs_available boolean NOT NULL,
  gefs_available boolean NOT NULL,
  station_proxy_available boolean NOT NULL,
  ifs_shadow_available boolean NOT NULL,
  ai_shadow_available boolean NOT NULL,
  arwf_live_shadow_available boolean NOT NULL,
  cwa_live_shadow_available boolean NOT NULL,
  missing_source_jsonb jsonb NOT NULL DEFAULT '{}'::jsonb,
  leakage_status text NOT NULL,
  run_id text NOT NULL REFERENCES model_core.run_manifest(run_id),
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  updated_at_utc timestamptz NOT NULL DEFAULT now(),
  UNIQUE (cutoff_id, target_date_hkt)
);
CREATE INDEX IF NOT EXISTS idx_h24n_snapshot_target_date ON model_features.h24n_snapshot(target_date_hkt);
```

### 5.6 `model_features.official_features`

Permanent rebuildable table.

```sql
CREATE TABLE IF NOT EXISTS model_features.official_features (
  snapshot_id text PRIMARY KEY REFERENCES model_features.h24n_snapshot(snapshot_id),
  target_date_hkt date NOT NULL,
  anchor_issue_at_utc timestamptz NULL,
  anchor_row_id text NULL,
  official_forecast_min_c double precision NULL,
  official_forecast_max_c double precision NULL,
  official_forecast_range_c double precision NULL,
  official_forecast_midpoint_c double precision NULL,
  eligible_issue_count integer NOT NULL,
  first_eligible_issue_at_utc timestamptz NULL,
  latest_eligible_issue_at_utc timestamptz NULL,
  issue_span_hours double precision NULL,
  max_first_c double precision NULL,
  max_latest_c double precision NULL,
  max_revision_latest_minus_first_c double precision NULL,
  max_revision_abs_sum_c double precision NULL,
  min_revision_latest_minus_first_c double precision NULL,
  text_latest text NULL,
  weather_text_latest text NULL,
  wind_text_latest text NULL,
  text_token_jsonb jsonb NOT NULL DEFAULT '{}'::jsonb,
  official_available boolean NOT NULL,
  missing_reason text NULL,
  source_row_jsonb jsonb NOT NULL DEFAULT '{}'::jsonb,
  run_id text NOT NULL REFERENCES model_core.run_manifest(run_id),
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  updated_at_utc timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_official_features_date ON model_features.official_features(target_date_hkt);
```

### 5.7 `model_features.target_memory_features`

Permanent rebuildable table.

```sql
CREATE TABLE IF NOT EXISTS model_features.target_memory_features (
  snapshot_id text PRIMARY KEY REFERENCES model_features.h24n_snapshot(snapshot_id),
  target_date_hkt date NOT NULL,
  target_lag2_tmax_c double precision NULL,
  target_lag3_tmax_c double precision NULL,
  target_lag4_tmax_c double precision NULL,
  target_lag5_tmax_c double precision NULL,
  target_lag7_tmax_c double precision NULL,
  target_lag14_tmax_c double precision NULL,
  target_lag30_tmax_c double precision NULL,
  target_lag60_tmax_c double precision NULL,
  target_lag365_tmax_c double precision NULL,
  target_roll7_mean_lag2_c double precision NULL,
  target_roll14_mean_lag2_c double precision NULL,
  target_roll30_mean_lag2_c double precision NULL,
  target_roll60_mean_lag2_c double precision NULL,
  target_roll7_std_lag2_c double precision NULL,
  target_roll14_std_lag2_c double precision NULL,
  target_roll30_std_lag2_c double precision NULL,
  target_roll7_min_lag2_c double precision NULL,
  target_roll7_max_lag2_c double precision NULL,
  target_roll14_range_lag2_c double precision NULL,
  target_slope_7_30_lag2_c_per_day double precision NULL,
  target_slope_3_14_lag2_c_per_day double precision NULL,
  target_curvature_7_30_lag2_c double precision NULL,
  target_anomaly_vs_clim_lag2_c double precision NULL,
  target_climatology_doy_c double precision NULL,
  target_climatology_doy_n integer NULL,
  hot_spell_len_lag2 integer NULL,
  cool_spell_len_lag2 integer NULL,
  volatility_iqr_14_lag2_c double precision NULL,
  volatility_mad_14_lag2_c double precision NULL,
  missing_feature_jsonb jsonb NOT NULL DEFAULT '{}'::jsonb,
  run_id text NOT NULL REFERENCES model_core.run_manifest(run_id),
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  updated_at_utc timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_target_memory_date ON model_features.target_memory_features(target_date_hkt);
```

### 5.8 `model_features.nwp_safe_row_ledger`

Permanent audit table.

```sql
CREATE TABLE IF NOT EXISTS model_features.nwp_safe_row_ledger (
  ledger_id bigserial PRIMARY KEY,
  dataset_code text NOT NULL,
  source_response_object_id text NOT NULL,
  target_date_hkt date NOT NULL,
  run_time_utc timestamptz NOT NULL,
  valid_time_utc timestamptz NOT NULL,
  lead_hours double precision NOT NULL,
  location_code text NOT NULL,
  member_number integer NOT NULL,
  row_is_safe_h24n boolean NOT NULL,
  row_excluded_reason text NULL,
  source_scope text NOT NULL,
  run_id text NOT NULL REFERENCES model_core.run_manifest(run_id),
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  updated_at_utc timestamptz NOT NULL DEFAULT now(),
  UNIQUE (dataset_code, source_response_object_id, target_date_hkt, run_time_utc, valid_time_utc, location_code, member_number)
);
CREATE INDEX IF NOT EXISTS idx_nwp_safe_ledger_date_dataset ON model_features.nwp_safe_row_ledger(target_date_hkt, dataset_code);
```

### 5.9 `model_features.nwp_daily_features`

Permanent rebuildable table.

```sql
CREATE TABLE IF NOT EXISTS model_features.nwp_daily_features (
  snapshot_id text NOT NULL REFERENCES model_features.h24n_snapshot(snapshot_id),
  dataset_code text NOT NULL,
  source_status text NOT NULL,
  target_date_hkt date NOT NULL,
  center_tmax_c double precision NULL,
  center_inst_tmax_c double precision NULL,
  center_interval_tmax_c double precision NULL,
  center_tmin_c double precision NULL,
  center_dewpoint_peak_c double precision NULL,
  center_temp_dewpoint_spread_peak_c double precision NULL,
  center_u10_mean_mps double precision NULL,
  center_v10_mean_mps double precision NULL,
  center_wind_speed_mean_mps double precision NULL,
  center_mslp_mean_hpa double precision NULL,
  center_low_cloud_mean_pct double precision NULL,
  center_precip_window_mm double precision NULL,
  center_shortwave_window_mj_m2 double precision NULL,
  center_temp_925_peak_c double precision NULL,
  center_temp_850_peak_c double precision NULL,
  center_rh700_mean_pct double precision NULL,
  center_z500_mean_m double precision NULL,
  spatial_tmax_mean_c double precision NULL,
  spatial_tmax_max_c double precision NULL,
  spatial_tmax_min_c double precision NULL,
  spatial_tmax_range_c double precision NULL,
  inland_nw_far_minus_center_tmax_c double precision NULL,
  marine_s_far_minus_center_tmax_c double precision NULL,
  marine_e_far_minus_center_tmax_c double precision NULL,
  north_minus_south_tmax_c double precision NULL,
  east_minus_west_tmax_c double precision NULL,
  local_gradient_abs_c double precision NULL,
  valid_row_count integer NOT NULL,
  location_count integer NOT NULL,
  lead_count integer NOT NULL,
  missing_reason text NULL,
  quality_flags_jsonb jsonb NOT NULL DEFAULT '{}'::jsonb,
  run_id text NOT NULL REFERENCES model_core.run_manifest(run_id),
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  updated_at_utc timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (snapshot_id, dataset_code)
);
CREATE INDEX IF NOT EXISTS idx_nwp_daily_dataset_date ON model_features.nwp_daily_features(dataset_code, target_date_hkt);
```

### 5.10 `model_features.nwp_ensemble_features`

Permanent rebuildable table.

```sql
CREATE TABLE IF NOT EXISTS model_features.nwp_ensemble_features (
  snapshot_id text NOT NULL REFERENCES model_features.h24n_snapshot(snapshot_id),
  dataset_code text NOT NULL,
  source_status text NOT NULL,
  target_date_hkt date NOT NULL,
  member_count_expected integer NOT NULL,
  member_count_available integer NOT NULL,
  member0_available boolean NULL,
  ens_tmax_mean_c double precision NULL,
  ens_tmax_median_c double precision NULL,
  ens_tmax_p10_c double precision NULL,
  ens_tmax_p25_c double precision NULL,
  ens_tmax_p75_c double precision NULL,
  ens_tmax_p90_c double precision NULL,
  ens_tmax_iqr_c double precision NULL,
  ens_tmax_spread_p90_p10_c double precision NULL,
  ens_tmax_std_c double precision NULL,
  ens_prob_ge_25c double precision NULL,
  ens_prob_ge_26c double precision NULL,
  ens_prob_ge_27c double precision NULL,
  ens_prob_ge_28c double precision NULL,
  ens_prob_ge_29c double precision NULL,
  ens_prob_ge_30c double precision NULL,
  ens_prob_ge_31c double precision NULL,
  ens_prob_ge_32c double precision NULL,
  ens_prob_ge_33c double precision NULL,
  ens_prob_ge_34c double precision NULL,
  ens_prob_ge_35c double precision NULL,
  missing_reason text NULL,
  quality_flags_jsonb jsonb NOT NULL DEFAULT '{}'::jsonb,
  run_id text NOT NULL REFERENCES model_core.run_manifest(run_id),
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  updated_at_utc timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (snapshot_id, dataset_code)
);
CREATE INDEX IF NOT EXISTS idx_nwp_ens_dataset_date ON model_features.nwp_ensemble_features(dataset_code, target_date_hkt);
```

### 5.11 Proxy feature tables

`model_features.station_proxy_features` and `model_features.diagnostic_proxy_features` are permanent rebuildable tables.

```sql
CREATE TABLE IF NOT EXISTS model_features.station_proxy_features (
  snapshot_id text PRIMARY KEY REFERENCES model_features.h24n_snapshot(snapshot_id),
  target_date_hkt date NOT NULL,
  station_obs_count_total integer NULL,
  station_count_available integer NULL,
  station_temp_latest_mean_c double precision NULL,
  station_temp_latest_max_c double precision NULL,
  station_temp_latest_min_c double precision NULL,
  station_temp_latest_range_c double precision NULL,
  station_dewpoint_latest_mean_c double precision NULL,
  station_temp_dewpoint_spread_mean_c double precision NULL,
  station_pressure_latest_mean_hpa double precision NULL,
  station_wind_speed_latest_mean_mps double precision NULL,
  station_temp_anomaly_14d_mean_c double precision NULL,
  station_dewpoint_change_1d_mean_c double precision NULL,
  station_pressure_change_1d_mean_hpa double precision NULL,
  coastal_inland_temp_spread_proxy_c double precision NULL,
  station_disagreement_index double precision NULL,
  wind_direction_used boolean NOT NULL DEFAULT false,
  strict_allowed boolean NOT NULL DEFAULT false,
  proxy_reason text NOT NULL,
  run_id text NOT NULL REFERENCES model_core.run_manifest(run_id),
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  updated_at_utc timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_features.diagnostic_proxy_features (
  snapshot_id text PRIMARY KEY REFERENCES model_features.h24n_snapshot(snapshot_id),
  target_date_hkt date NOT NULL,
  climate_lag2_dewpoint_c double precision NULL,
  climate_lag2_mslp_hpa double precision NULL,
  climate_lag2_rainfall_mm double precision NULL,
  climate_lag2_cloud_amount_pct double precision NULL,
  climate_lag2_sunshine_hours double precision NULL,
  climate_lag2_sea_temp_c double precision NULL,
  diagnostic_upper_air_teacher_available boolean NOT NULL DEFAULT false,
  strict_allowed boolean NOT NULL DEFAULT false,
  proxy_reason text NOT NULL,
  run_id text NOT NULL REFERENCES model_core.run_manifest(run_id),
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  updated_at_utc timestamptz NOT NULL DEFAULT now()
);
```

### 5.12 Feature matrices

Permanent rebuildable wide tables.

```sql
CREATE TABLE IF NOT EXISTS model_features.snapshot_feature_matrix_strict (
  snapshot_id text PRIMARY KEY REFERENCES model_features.h24n_snapshot(snapshot_id),
  target_date_hkt date NOT NULL,
  label_visible boolean NOT NULL,
  target_tmax_c double precision NULL,
  feature_jsonb jsonb NOT NULL,
  availability_jsonb jsonb NOT NULL,
  feature_schema_version text NOT NULL,
  run_id text NOT NULL REFERENCES model_core.run_manifest(run_id),
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  updated_at_utc timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_features.snapshot_feature_matrix_proxy (
  snapshot_id text PRIMARY KEY REFERENCES model_features.h24n_snapshot(snapshot_id),
  target_date_hkt date NOT NULL,
  label_visible boolean NOT NULL,
  target_tmax_c double precision NULL,
  feature_jsonb jsonb NOT NULL,
  availability_jsonb jsonb NOT NULL,
  feature_schema_version text NOT NULL,
  proxy_feature_count integer NOT NULL,
  run_id text NOT NULL REFERENCES model_core.run_manifest(run_id),
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  updated_at_utc timestamptz NOT NULL DEFAULT now()
);
```

Codex may materialize additional rectangular parquet files for model training, but PostgreSQL remains the source of truth for row identity and feature provenance.

### 5.13 OOF, router, specialist, validation, live tables

```sql
CREATE TABLE IF NOT EXISTS model_oof.expert_prediction (
  prediction_id text PRIMARY KEY,
  snapshot_id text NOT NULL REFERENCES model_features.h24n_snapshot(snapshot_id),
  target_date_hkt date NOT NULL,
  expert_id text NOT NULL,
  router_scope text NOT NULL,
  fold_id text NOT NULL,
  model_artifact_id text NOT NULL,
  prediction_tmax_c double precision NULL,
  prediction_residual_c double precision NULL,
  expected_abs_error_c double precision NULL,
  available boolean NOT NULL,
  unavailable_reason text NULL,
  train_start_date date NULL,
  train_end_date date NULL,
  feature_schema_version text NOT NULL,
  run_id text NOT NULL REFERENCES model_core.run_manifest(run_id),
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  updated_at_utc timestamptz NOT NULL DEFAULT now(),
  UNIQUE (snapshot_id, expert_id, router_scope)
);
CREATE INDEX IF NOT EXISTS idx_expert_prediction_date ON model_oof.expert_prediction(target_date_hkt);

CREATE TABLE IF NOT EXISTS model_oof.expert_artifact (
  model_artifact_id text PRIMARY KEY,
  expert_id text NOT NULL,
  fold_id text NOT NULL,
  artifact_uri text NOT NULL,
  train_start_date date NOT NULL,
  train_end_date date NOT NULL,
  feature_list_sha256 text NOT NULL,
  model_params_jsonb jsonb NOT NULL,
  preprocessing_jsonb jsonb NOT NULL,
  run_id text NOT NULL REFERENCES model_core.run_manifest(run_id),
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  updated_at_utc timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_oof.system_prediction (
  system_prediction_id text PRIMARY KEY,
  snapshot_id text NOT NULL REFERENCES model_features.h24n_snapshot(snapshot_id),
  target_date_hkt date NOT NULL,
  system_version text NOT NULL,
  router_version text NOT NULL,
  base_forecast_c double precision NULL,
  specialist_total_correction_c double precision NOT NULL DEFAULT 0,
  final_point_tmax_c double precision NULL,
  final_p10_c double precision NULL,
  final_p25_c double precision NULL,
  final_p50_c double precision NULL,
  final_p75_c double precision NULL,
  final_p90_c double precision NULL,
  expected_abs_error_c double precision NULL,
  confidence_state text NULL,
  no_trade_flag boolean NOT NULL DEFAULT false,
  component_jsonb jsonb NOT NULL,
  run_id text NOT NULL REFERENCES model_core.run_manifest(run_id),
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  updated_at_utc timestamptz NOT NULL DEFAULT now(),
  UNIQUE (snapshot_id, system_version)
);

CREATE TABLE IF NOT EXISTS model_router.router_prediction (
  router_prediction_id text PRIMARY KEY,
  snapshot_id text NOT NULL REFERENCES model_features.h24n_snapshot(snapshot_id),
  target_date_hkt date NOT NULL,
  router_version text NOT NULL,
  router_scope text NOT NULL,
  fold_id text NOT NULL,
  base_forecast_c double precision NULL,
  static_weight_jsonb jsonb NOT NULL,
  dynamic_weight_jsonb jsonb NOT NULL,
  final_weight_jsonb jsonb NOT NULL,
  expected_error_jsonb jsonb NOT NULL,
  availability_mask_jsonb jsonb NOT NULL,
  selected_tau double precision NOT NULL,
  selected_lambda double precision NOT NULL,
  available boolean NOT NULL,
  unavailable_reason text NULL,
  run_id text NOT NULL REFERENCES model_core.run_manifest(run_id),
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  updated_at_utc timestamptz NOT NULL DEFAULT now(),
  UNIQUE (snapshot_id, router_version, router_scope)
);

CREATE TABLE IF NOT EXISTS model_router.specialist_prediction (
  specialist_prediction_id text PRIMARY KEY,
  snapshot_id text NOT NULL REFERENCES model_features.h24n_snapshot(snapshot_id),
  target_date_hkt date NOT NULL,
  specialist_id text NOT NULL,
  fold_id text NOT NULL,
  regime_probability double precision NULL,
  predicted_correction_c double precision NULL,
  applied_correction_c double precision NOT NULL DEFAULT 0,
  expected_benefit_c double precision NULL,
  activated boolean NOT NULL,
  activation_reason text NOT NULL,
  support_count integer NULL,
  no_harm_pass boolean NULL,
  run_id text NOT NULL REFERENCES model_core.run_manifest(run_id),
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  updated_at_utc timestamptz NOT NULL DEFAULT now(),
  UNIQUE (snapshot_id, specialist_id)
);

CREATE TABLE IF NOT EXISTS model_validation.scoreboard (
  scoreboard_id text PRIMARY KEY,
  scoreboard_scope text NOT NULL,
  candidate_id text NOT NULL,
  baseline_id text NULL,
  row_count integer NOT NULL,
  first_target_date date NOT NULL,
  last_target_date date NOT NULL,
  mae_c double precision NOT NULL,
  rmse_c double precision NOT NULL,
  bias_c double precision NOT NULL,
  median_abs_error_c double precision NOT NULL,
  p75_abs_error_c double precision NOT NULL,
  p90_abs_error_c double precision NOT NULL,
  p95_abs_error_c double precision NOT NULL,
  large_error_ge_1c_rate double precision NOT NULL,
  large_error_ge_2c_rate double precision NOT NULL,
  delta_mae_vs_baseline_c double precision NULL,
  slice_jsonb jsonb NOT NULL DEFAULT '{}'::jsonb,
  pass_fail_status text NOT NULL,
  run_id text NOT NULL REFERENCES model_core.run_manifest(run_id),
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  updated_at_utc timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_validation.negative_control_result (
  control_id text PRIMARY KEY,
  control_name text NOT NULL,
  candidate_id text NOT NULL,
  row_count integer NOT NULL,
  mae_c double precision NULL,
  expected_behavior text NOT NULL,
  pass_fail_status text NOT NULL,
  details_jsonb jsonb NOT NULL DEFAULT '{}'::jsonb,
  run_id text NOT NULL REFERENCES model_core.run_manifest(run_id),
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  updated_at_utc timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_validation.leakage_audit_event (
  event_id text PRIMARY KEY,
  severity text NOT NULL,
  source_table text NULL,
  source_column text NULL,
  snapshot_id text NULL,
  target_date_hkt date NULL,
  rule_id text NOT NULL,
  message text NOT NULL,
  fail_closed_action text NOT NULL,
  run_id text NOT NULL REFERENCES model_core.run_manifest(run_id),
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  updated_at_utc timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_live.prediction (
  live_prediction_id text PRIMARY KEY,
  target_date_hkt date NOT NULL,
  cutoff_id text NOT NULL,
  formal_cutoff_utc timestamptz NOT NULL,
  prediction_created_at_utc timestamptz NOT NULL,
  final_point_tmax_c double precision NOT NULL,
  final_point_tmax_rounded_0p1c double precision NOT NULL,
  p10_c double precision NULL,
  p25_c double precision NULL,
  p50_c double precision NULL,
  p75_c double precision NULL,
  p90_c double precision NULL,
  expected_abs_error_c double precision NULL,
  threshold_prob_jsonb jsonb NOT NULL DEFAULT '{}'::jsonb,
  confidence_state text NOT NULL,
  no_trade_flag boolean NOT NULL,
  source_availability_jsonb jsonb NOT NULL,
  component_jsonb jsonb NOT NULL,
  audit_sha256 text NOT NULL,
  run_id text NOT NULL REFERENCES model_core.run_manifest(run_id),
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  updated_at_utc timestamptz NOT NULL DEFAULT now(),
  UNIQUE (target_date_hkt, cutoff_id)
);
```

---

## 6. H24N snapshot builder

### 6.1 Snapshot ID format

```text
snapshot_id = H24N:<YYYY-MM-DD>
```

Example:

```text
H24N:2023-07-15
```

### 6.2 Target-date ranges

Codex MUST build snapshots for:

```text
2000-01-02 through 2026-06-21
```

The pipeline MUST support extending beyond `2026-06-21` for live inference.

Partitions:

```text
pre2024_development: 2000-01-02 through 2023-12-31
sealed_2024:         2024-01-01 through 2024-12-31
sealed_2025:         2025-01-01 through 2025-12-31
prospective_2026:    2026-01-01 onward
```

For dates where target labels are unavailable, `label_visible=false` and scoring is not performed.

### 6.3 Source inclusion rules

For each snapshot, Codex MUST build availability flags:

```text
official_available
gfs_available
gefs_available
ifs_shadow_available
ai_shadow_available
station_proxy_available
arwf_live_shadow_available
cwa_live_shadow_available
```

A missing source does not invalidate the snapshot. It sets the corresponding availability flag to false and forces that expert's weight to zero.

### 6.4 Duplicate handling

For each source family and target date, duplicate candidate source rows MUST be resolved as follows:

1. Exact duplicate rows with identical source hash are collapsed.
2. Multiple official forecast rows are retained for revision features, and the latest eligible pre-freeze row becomes the anchor.
3. Multiple NWP rows with identical dataset/run/valid/location/member keys are invalid unless raw response hashes are identical. Non-identical duplicates are quarantined.
4. Multiple station rows at same station/timestamp use the row with the best quality code; if quality ties, the latest ingested row is used and the tie is audited.

### 6.5 Fail-closed leakage behavior

If a feature references any timestamp after `operational_freeze_utc`, Codex MUST:

```text
exclude feature
write model_validation.leakage_audit_event severity='ERROR'
mark source unavailable for that snapshot
continue only if remaining experts can produce prediction
```

If a target outcome is accessed before scoring mode, Codex MUST stop with failure.

### 6.6 Required audit fields

`reports/h24n_snapshot_coverage.md` MUST include:

```text
total snapshots by partition
formal cutoff formula
operational freeze formula
source availability by year
official anchor coverage by year
NWP availability by model and year
proxy availability by year
count of fail-closed leakage exclusions
count of duplicate resolutions
count of snapshots with no strict forecast source
```

---

## 7. Official HKO forecast anchor

### 7.1 Usable row filter

The anchor input subset is:

```sql
SELECT *
FROM public.hko_historical_forecasts_2000_2026
WHERE row_quality_status = 'usable_local_minmax'
  AND product_type = 'local'
  AND forecast_min_c IS NOT NULL
  AND forecast_max_c IS NOT NULL
  AND forecast_min_c BETWEEN 0 AND 35
  AND forecast_max_c BETWEEN 5 AND 45
  AND forecast_max_c >= forecast_min_c
  AND target_date IS NOT NULL
  AND issue_at_utc IS NOT NULL;
```

Rows failing this filter are not features. They are counted in the source audit.

### 7.2 Latest pre-cutoff anchor selection

For each `snapshot_id`, select:

```sql
WITH eligible AS (
  SELECT
    f.*,
    c.formal_cutoff_utc,
    c.operational_freeze_utc,
    ROW_NUMBER() OVER (
      PARTITION BY c.target_date_hkt
      ORDER BY f.issue_at_utc DESC, f.forecast_max_c DESC, f.forecast_min_c DESC
    ) AS rn_latest,
    COUNT(*) OVER (PARTITION BY c.target_date_hkt) AS eligible_issue_count,
    MIN(f.issue_at_utc) OVER (PARTITION BY c.target_date_hkt) AS first_issue_at_utc,
    MAX(f.issue_at_utc) OVER (PARTITION BY c.target_date_hkt) AS latest_issue_at_utc,
    FIRST_VALUE(f.forecast_max_c) OVER (
      PARTITION BY c.target_date_hkt
      ORDER BY f.issue_at_utc ASC
      ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS max_first_c,
    FIRST_VALUE(f.forecast_max_c) OVER (
      PARTITION BY c.target_date_hkt
      ORDER BY f.issue_at_utc DESC
      ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS max_latest_c
  FROM model_core.cutoff_calendar c
  JOIN public.hko_historical_forecasts_2000_2026 f
    ON f.target_date::date = c.target_date_hkt
  WHERE c.cutoff_id = 'H24N'
    AND f.row_quality_status = 'usable_local_minmax'
    AND f.product_type = 'local'
    AND f.forecast_min_c IS NOT NULL
    AND f.forecast_max_c IS NOT NULL
    AND f.issue_at_utc <= c.operational_freeze_utc
)
SELECT *
FROM eligible
WHERE rn_latest = 1;
```

### 7.3 Revision features

For all eligible pre-freeze official forecast rows for the target date, compute:

```text
eligible_issue_count
first_eligible_issue_at_utc
latest_eligible_issue_at_utc
issue_span_hours = latest - first in hours
max_first_c
max_latest_c
max_revision_latest_minus_first_c = max_latest_c - max_first_c
max_revision_abs_sum_c = sum(abs(forecast_max_c_i - forecast_max_c_{i-1})) ordered by issue_at_utc
min_revision_latest_minus_first_c
forecast_range_latest_c = forecast_max_c - forecast_min_c
forecast_midpoint_latest_c = (forecast_max_c + forecast_min_c)/2
```

If only one eligible row exists, revision deltas are `0` and `eligible_issue_count=1`.

### 7.4 Text features

For `forecast_text`, `weather_text`, and `wind_text`, Codex MUST create deterministic token flags using lowercase normalized text.

Required token flags:

```text
text_has_sunny
text_has_fine
text_has_cloudy
text_has_overcast
text_has_showers
text_has_rain
text_has_thunder
text_has_mist
text_has_fog
text_has_haze
text_has_hot
text_has_very_hot
text_has_cool
text_has_warm
text_has_easterly
text_has_southeasterly
text_has_southerly
text_has_southwesterly
text_has_westerly
text_has_northerly
text_has_monsoon
text_has_light_winds
text_has_moderate_winds
text_has_fresh_winds
text_has_strong_winds
```

Fold-local learned text representations are not included in the first strict implementation. Deterministic token flags only.

### 7.5 Missing official forecast

If no eligible official forecast exists for a snapshot:

```text
official_available=false
E0_OFFICIAL_RAW unavailable
E1_OFFICIAL_RESIDUAL unavailable
router masks official experts to zero
fallback uses target-memory and available NWP experts
missing reason = NO_PRE_FREEZE_OFFICIAL_LOCAL_MINMAX
```

No same-day or post-freeze forecast may be substituted.

---

## 8. Target label and target-memory features

### 8.1 Label contract

The label is:

```text
target_tmax_c = settled HKO daily maximum temperature in °C for target_date_hkt
```

The label is never a feature for the same target date.

### 8.2 Safe lag rules

For H24N, finalized daily target memory uses:

```text
allowed dates <= T-2
forbidden dates >= T-1
```

Thus:

```text
target_lag2_tmax_c = target_tmax_c on T-2
target_lag3_tmax_c = target_tmax_c on T-3
...
```

`target_lag1_tmax_c` is not created in strict or proxy matrices.

### 8.3 Exact target-memory formulas

Let `y[d]` be settled HKO Tmax on local date `d`.

For target date `T`:

```text
target_lagK_tmax_c = y[T-K], K in {2,3,4,5,7,14,30,60,365}
```

Rolling windows:

```text
target_roll7_mean_lag2_c  = mean(y[T-8]  ... y[T-2])
target_roll14_mean_lag2_c = mean(y[T-15] ... y[T-2])
target_roll30_mean_lag2_c = mean(y[T-31] ... y[T-2])
target_roll60_mean_lag2_c = mean(y[T-61] ... y[T-2])

target_roll7_std_lag2_c  = sample_std(y[T-8]  ... y[T-2])
target_roll14_std_lag2_c = sample_std(y[T-15] ... y[T-2])
target_roll30_std_lag2_c = sample_std(y[T-31] ... y[T-2])

target_roll7_min_lag2_c = min(y[T-8] ... y[T-2])
target_roll7_max_lag2_c = max(y[T-8] ... y[T-2])
target_roll14_range_lag2_c = max(y[T-15]...y[T-2]) - min(y[T-15]...y[T-2])
```

Trend features:

```text
target_slope_7_30_lag2_c_per_day =
    (mean(y[T-8]...y[T-2]) - mean(y[T-31]...y[T-9])) / 23

target_slope_3_14_lag2_c_per_day =
    (mean(y[T-4]...y[T-2]) - mean(y[T-15]...y[T-5])) / 10

target_curvature_7_30_lag2_c =
    target_slope_3_14_lag2_c_per_day - target_slope_7_30_lag2_c_per_day
```

Volatility features:

```text
volatility_iqr_14_lag2_c = percentile75(y[T-15]...y[T-2]) - percentile25(y[T-15]...y[T-2])
volatility_mad_14_lag2_c = median(abs(y[d] - median(y[T-15]...y[T-2]))) for d in T-15...T-2
```

Causal climatology:

```text
clim_pool(T) = all dates d such that:
    year(d) < year(T)
    abs(day_of_year_circular(d) - day_of_year_circular(T)) <= 15
    target_tmax_c exists

target_climatology_doy_c = mean(y[d] for d in clim_pool(T))
target_climatology_doy_n = count(clim_pool(T))
target_anomaly_vs_clim_lag2_c = target_lag2_tmax_c - target_climatology_doy_c
```

Minimum climatology history:

```text
if target_climatology_doy_n < 20:
    target_climatology_doy_c = NULL
    target_anomaly_vs_clim_lag2_c = NULL
    missing flag climatology_insufficient_history = true
```

Spell features:

```text
hot_threshold(T) = causal 85th percentile for day-of-year ±15 from years < year(T)
cool_threshold(T) = causal 15th percentile for day-of-year ±15 from years < year(T)

hot_spell_len_lag2 = count consecutive dates ending T-2 with y[d] >= hot_threshold(T)
cool_spell_len_lag2 = count consecutive dates ending T-2 with y[d] <= cool_threshold(T)
```

Minimum rolling coverage:

```text
7-day windows require >= 6 observed days
14-day windows require >= 12 observed days
30-day windows require >= 25 observed days
60-day windows require >= 50 observed days
```

Missing windows produce NULL plus a missingness flag.

---

## 9. GribStream/NWP dataset treatment

The inventory states that the current full tactical backfill source scope is `full_tactical_backfill_ok_tmax`, with `1,964,157` normalized rows in `nwp_tactical.forecast_wide`. It also states that raw forecast rows are not automatically feature-safe and must be filtered by H24N cutoff with a 6-hour buffer.

Codex MUST implement the treatment table below exactly.

### 9.1 Dataset treatment table

| Dataset | First implementation status | Training use | Date range in current inventory | Enters first strict pre-2024 system | Router weight rule |
|---|---|---|---|---:|---|
| `gfs` | STRICT_CORE | GFS MOS expert | target dates 2021-03-23 to 2026-06-23 | yes | normal R1 weight when available |
| `gefsatmosmean` | STRICT_CORE | GEFS mean features | target dates 2020-10-03 to 2026-06-23 | yes | feeds E5/R1 context |
| `gefsatmos` | STRICT_CORE | GEFS member ensemble expert | target dates 2020-10-03 to 2026-06-23 | yes | normal R1 weight when available |
| `ifsoper` | SHADOW_CAPPED_CHALLENGER | IFS deterministic shadow | target dates 2024-02-29 to 2026-06-23 | no pre-2024 | zero until sealed adapter pass; cap 0.10 first year after pass |
| `ifsenfo` | SHADOW_CAPPED_CHALLENGER | IFS ensemble shadow | target dates 2024-03-03 to 2026-06-23 | no pre-2024 | zero until sealed adapter pass; cap 0.10 first year after pass |
| `cwawrf15` | LIVE_SHADOW | rolling prospective expert | target dates 2026-06-23 to 2026-06-26 in current inventory | no | zero until 365 prospective scored days; then cap 0.05 |
| `aifsoper` | SHADOW_CAPPED_CHALLENGER | AI deterministic shadow | target dates 2025-02-26 to 2026-06-23 | no | zero until 365 scored days; cap 0.05 |
| `aifsenfo` | SHADOW_CAPPED_CHALLENGER | AI ensemble shadow | target dates 2025-07-04 to 2026-06-23 | no | zero until 365 scored days; cap 0.05 |
| `aigfssfc` | SHADOW_SHORT_RANGE | AI/GFS surface deterministic shadow | target dates 2026-04-22 to 2026-06-23 | no | zero in first implementation |
| `aigfspres` | SUPPORT_ONLY_BLOCKED_AS_TMAX | upper-air support only | target dates 2026-04-22 to 2026-06-23 | no | forced zero |
| `aigefssfc` | BLOCKED_AS_TMAX | poor surface coverage | 373 seen, only 67 days with Tmax candidate | no | forced zero |
| `graphcast` | SHADOW_HISTORICAL_AI | AI deterministic shadow | target dates 2024-04-26 to 2026-05-06 | no pre-2024 | zero until sealed/shadow report; cap 0.05 after pass |
| `fourcastnetgfs` | SHADOW_HISTORICAL_AI | AI deterministic shadow | target dates 2024-05-03 to 2026-02-20 | no pre-2024 | zero after archive end; cap 0.05 after pass |
| `nbmoc` | BLOCKED_EMPTY | none | zero rows | no | forced zero |

### 9.2 Training and validation ranges by dataset

```text
gfs:
  pre2024 training: 2021-03-23 through 2023-12-31
  sealed validation: 2024-01-01 through 2024-12-31
  sealed final test: 2025-01-01 through 2025-12-31
  prospective: 2026 onward

gefsatmosmean / gefsatmos:
  pre2024 training: 2021-03-23 through 2023-12-31 for R1 common row training
  source-specific diagnostics may start 2020-10-03
  sealed validation: 2024
  sealed final test: 2025
  prospective: 2026 onward

ifsoper / ifsenfo:
  no pre2024 training
  sealed adapter calibration: 2024 only after explicit sealed open
  sealed final test: 2025 only after frozen adapter rules
  prospective: 2026 onward

ai challengers:
  no pre2024 training
  sealed/shadow only according to available dates
  no strict router impact until promotion gates pass

cwawrf15 / arwf:
  no historical strict training in first implementation
  live shadow only
```

### 9.3 Availability representation

For every dataset and target date:

```text
available = true only when all required features for that dataset are present and safe
available = false with unavailable_reason otherwise
```

Missing source features MUST remain NULL in the feature table and have mask fields in `availability_jsonb`.

No imputation may convert a missing expert into a real forecast. Router weight for unavailable experts is exactly zero.

---

## 10. NWP feature formulas

### 10.1 Safe row SQL filter

Codex MUST implement the canonical NWP safe row view:

```sql
CREATE OR REPLACE VIEW model_features.v_nwp_h24n_safe_rows AS
SELECT
  fw.*,
  r.object_uri,
  c.formal_cutoff_utc,
  c.operational_freeze_utc,
  CASE
    WHEN r.object_uri NOT LIKE '%full_tactical_backfill_ok_tmax%' THEN false
    WHEN fw.run_time_utc + INTERVAL '6 hours' > c.formal_cutoff_utc THEN false
    WHEN fw.dataset_code IN ('nbmoc', 'aigfspres', 'aigefssfc') THEN false
    WHEN fw.cutoff_id <> 'H24N' THEN false
    ELSE true
  END AS row_is_safe_h24n,
  CASE
    WHEN r.object_uri NOT LIKE '%full_tactical_backfill_ok_tmax%' THEN 'NOT_FULL_TACTICAL_SCOPE'
    WHEN fw.run_time_utc + INTERVAL '6 hours' > c.formal_cutoff_utc THEN 'POST_CUTOFF_OR_INSUFFICIENT_PUBLICATION_BUFFER'
    WHEN fw.dataset_code IN ('nbmoc', 'aigfspres', 'aigefssfc') THEN 'DATASET_BLOCKED_FOR_TMAX'
    WHEN fw.cutoff_id <> 'H24N' THEN 'WRONG_CUTOFF_ID'
    ELSE NULL
  END AS exclusion_reason
FROM nwp_tactical.forecast_wide fw
JOIN nwp_tactical.raw_response_object r
  ON r.response_object_id = fw.source_response_object_id
JOIN model_core.cutoff_calendar c
  ON c.cutoff_id = 'H24N'
 AND c.target_date_hkt = fw.target_date_hkt::date;
```

Feature extraction MUST use only `row_is_safe_h24n=true` for strict NWP features.

### 10.2 Unit conversions

```text
Kelvin to Celsius: K - 273.15
Pa to hPa: Pa / 100
kg/m2 precipitation to mm: numeric value unchanged
m precipitation to mm: m * 1000
J/m2 to MJ/m2: J/m2 / 1,000,000
W/m2 remains W/m2 for instantaneous/rate fields
```

### 10.3 Target valid-time window

NWP features use rows whose `valid_time_utc` maps to target date HKT:

```sql
(valid_time_utc AT TIME ZONE 'Asia/Hong_Kong')::date = target_date_hkt
```

The full local day is used. No narrower daytime-only filter is used for primary Tmax aggregation.

### 10.4 Candidate daily Tmax formula

For each model/location/member/target date:

```text
instant_tmax_c = max(temperature_2m_k - 273.15) over target-date HKT valid times where temperature_2m_k is present
interval_tmax_c = max(interval_tmax_2m_k - 273.15) over target-date HKT valid times where interval_tmax_2m_k is present
candidate_tmax_c = max(instant_tmax_c, interval_tmax_c) when both exist
candidate_tmax_c = instant_tmax_c when interval_tmax_c missing
candidate_tmax_c = interval_tmax_c when instant_tmax_c missing
candidate_tmax_c = NULL when both missing
```

Reject values outside `[-10, 50]` °C and record a quality flag.

### 10.5 Deterministic 12-point spatial features

For 12-point datasets (`gfs`, `gefsatmosmean`, `ifsoper`, `aifsoper`, `aigfssfc`, `graphcast`, `fourcastnetgfs`, `cwawrf15`), compute:

```text
center_tmax_c = candidate_tmax_c at location_code='hko_center'
spatial_tmax_mean_c = mean(candidate_tmax_c over available 12 locations)
spatial_tmax_max_c = max(candidate_tmax_c over available 12 locations)
spatial_tmax_min_c = min(candidate_tmax_c over available 12 locations)
spatial_tmax_range_c = spatial_tmax_max_c - spatial_tmax_min_c
inland_nw_far_minus_center_tmax_c = tmax(inland_nw_far) - tmax(hko_center)
marine_s_far_minus_center_tmax_c = tmax(marine_s_far) - tmax(hko_center)
marine_e_far_minus_center_tmax_c = tmax(marine_e_far) - tmax(hko_center)
north_minus_south_tmax_c = tmax(local_n) - tmax(local_s)
east_minus_west_tmax_c = tmax(local_e) - tmax(local_w)
local_gradient_abs_c = sqrt((east_minus_west_tmax_c)^2 + (north_minus_south_tmax_c)^2)
```

If fewer than 9 of 12 deterministic locations are available, the dataset is unavailable for that snapshot.

### 10.6 Non-temperature NWP features

At `hko_center`, compute over target-date HKT valid times:

```text
center_tmin_c = min(temperature_2m_c)
center_dewpoint_peak_c = max(dewpoint_2m_c)
center_temp_dewpoint_spread_peak_c = max(temperature_2m_c - dewpoint_2m_c)
center_u10_mean_mps = mean(u_wind_10m_mps)
center_v10_mean_mps = mean(v_wind_10m_mps)
center_wind_speed_mean_mps = mean(sqrt(u10^2 + v10^2))
center_mslp_mean_hpa = mean(mslp_hpa)
center_low_cloud_mean_pct = mean(low_cloud_pct)
center_temp_925_peak_c = max(temperature_925_c)
center_temp_850_peak_c = max(temperature_850_c)
center_rh700_mean_pct = mean(relative_humidity_700_pct)
center_z500_mean_m = mean(geopotential_height_500_m)
```

Precipitation:

```text
For accumulated_precip_kg_m2:
  sort by valid_time_utc within same model run and location
  precip_window_mm = max(accumulation) - min(accumulation)
  if negative, set NULL and flag accumulation_reset

For total_precip_m:
  precip_window_mm = max(total_precip_m) * 1000 only when provider semantics are verified for that model in source registry
  for aifsoper and ifsoper first implementation: preserve value but set precip_window_mm=NULL unless source registry has unit_semantics_verified=true
```

Radiation:

```text
DSWRF W/m2: center_shortwave_window_mj_m2 = mean(DSWRF) * daylight_hours / 1000, using daylight_hours = count(non-null DSWRF valid rows)
SSRD J/m2: center_shortwave_window_mj_m2 = max(SSRD) / 1e6 - min(SSRD) / 1e6 within same run when non-decreasing; else NULL with reset flag
```

### 10.7 Ensemble handling

For `gefsatmos`, use all expected 31 members at `hko_center`.

For `ifsenfo`, use expected 51 members at `hko_center`. If member `0` is missing but at least 50 members are available, keep the day and set:

```text
member0_available=false
member_count_available=<count>
quality flag = MEMBER0_MISSING_ACCEPTED
```

If fewer than 50 of 51 IFS ENS members are available, mark `ifsenfo` unavailable for that snapshot.

For `aifsenfo`, require at least 50 of 51 members.

For `aigefssfc`, no first-implementation Tmax features are generated because the source is blocked as a Tmax source.

Ensemble daily member Tmax:

```text
member_daily_tmax_c[m] = max temperature candidate over target-date HKT valid times for member m
```

Ensemble features:

```text
ens_tmax_mean_c = mean(member_daily_tmax_c)
ens_tmax_median_c = median(member_daily_tmax_c)
ens_tmax_p10_c = percentile 10
ens_tmax_p25_c = percentile 25
ens_tmax_p75_c = percentile 75
ens_tmax_p90_c = percentile 90
ens_tmax_iqr_c = p75 - p25
ens_tmax_spread_p90_p10_c = p90 - p10
ens_tmax_std_c = sample standard deviation
ens_prob_ge_Xc = count(member_daily_tmax_c >= X) / available_member_count for X in 25..35
```

### 10.8 Short-history sources

Short-history sources generate features and shadow predictions only.

The first strict router masks these experts to zero:

```text
ifsoper until sealed adapter pass
aifsenfo/aifsoper/aigfssfc/graphcast/fourcastnet until shadow promotion pass
cwawrf15 until 365 prospective scored days
arwf until 365 prospective scored days
```

---

## 11. Combining datasets with different availability ranges

### 11.1 Common-row policy

Every scoreboard MUST state its row universe.

Definitions:

```text
source_specific_rows = all dates where a single expert has enough features and visible labels
common_router_rows = intersection of dates where all experts in that router are available and labels visible
strict_system_rows = dates where final strict system can produce prediction and label visible
proxy_system_rows = dates where proxy system can produce prediction and label visible
sealed_shadow_rows = dates where features are available but labels are sealed
```

Same-row comparison is mandatory for comparing any two candidates.

### 11.2 Router stages

#### R0 — `R0_OFFICIAL_LONG_HISTORY`

Experts:

```text
E0_OFFICIAL_RAW
E1_OFFICIAL_RESIDUAL
E2_TARGET_MEMORY
```

Proxy-only comparison may add:

```text
E3_STATION_PROXY
```

Training rows:

```text
2005-01-01 through 2023-12-31
```

Reason for 2005 start: requires enough early official history for OOF training and residual-memory state.

#### R1 — `R1_CORE_GFS_GEFS`

Experts:

```text
E0_OFFICIAL_RAW
E1_OFFICIAL_RESIDUAL
E2_TARGET_MEMORY
E4_GFS_MOS
E5_GEFS_PROB_MOS
```

Training rows:

```text
2021-03-23 through 2023-12-31
```

Rows require official anchor, target-memory, GFS features, GEFS features, and visible labels.

#### R2 — `R2_IFS_SHADOW_ADAPTER`

Experts:

```text
R1_CORE output
E6_IFS_OPER_SHADOW
E7_IFS_ENS_SHADOW
```

Training rows:

```text
none before sealed validation
```

The adapter is implemented but not trained for strict use until 2024 sealed validation is opened.

Initial production cap after 2024 pass:

```text
combined IFS weight <= 0.10
```

#### R3 — `R3_AI_SHADOW_ADAPTER`

Experts:

```text
R2 output
E8_AIFS_OPER_SHADOW
E8_AIFS_ENS_SHADOW
E8_AIGFS_SHADOW
E8_GRAPHCAST_SHADOW
E8_FOURCASTNET_SHADOW
```

Training rows are source-specific shadow rows only. No strict effect before promotion.

Initial production cap after pass:

```text
combined AI challenger weight <= 0.05
```

#### R4 — `R4_LIVE_SHADOW_ADAPTER`

Experts:

```text
R3 output
E9_CWA_WRF_LIVE_SHADOW
E11_ARWF_LIVE_SHADOW
```

Training rows:

```text
prospective exact-first-seen days only
```

Initial cap after 365 scored days:

```text
CWA WRF weight <= 0.05
ARWF weight <= 0.05
```

Full normal eligibility requires:

```text
730 prospective scored days
positive lift in at least two warm seasons and two cool seasons
negative controls pass
```

### 11.3 Comparing 2024+ models

IFS, AI, CWA, and ARWF sources MUST NOT be compared against pre-2024 strict champions as if they had the same history.

They receive separate shadow scoreboards:

```text
shadow_ifs_2024plus
shadow_ai_2025plus
shadow_cwa_live
shadow_arwf_live
```

They may enter the strict frozen system only after the sealed protocol promotes them.

---

## 12. Station and proxy data policy

### 12.1 First implementation status

Station-network features are included only in the `RESEARCH_PROXY` implementation.

They are not included in the first strict deployable historical system because:

```text
NOAA ISD archive is quality-controlled historical archive, not exact operational vintage.
existing ISD wind direction is corrupted/constant at 20 degrees.
release latency and live availability proof are not established.
```

### 12.2 Station features to build

Use only these fields:

```text
air_temperature_c_latest_before_1500
dew_point_c_latest_before_1500
sea_level_pressure_hpa_latest_before_1500
wind_speed_mps_latest_before_1500
obs_count
latest_before_1500_hkt
station_id
latitude
longitude
elevation_m
```

Forbidden:

```text
wind_direction_deg
wind_direction_deg_latest_before_1500
daily_air_temperature_max_c for target date T
daily_air_temperature_min_c for target date T
any station observation after operational_freeze_utc
```

### 12.3 Station groups

Create station groups by deterministic metadata rules:

```text
hko_like: stations within 20 km of HKO and elevation <= 100 m
coastal_proxy: stations within 10 km of coast if coastline distance exists; otherwise station IDs beginning 4500xx with known HK region are coastal_proxy only after metadata proof
inland_proxy: stations north or northwest of HKO by at least 0.25 degrees latitude/longitude proxy
marine_proxy: station metadata indicates island/marine exposure, or station is assigned manually in station dossier
unknown_role: all other stations
```

If station metadata is insufficient, group as `unknown_role`. Do not invent station names.

### 12.4 Station formulas

For target date `T`, station daily cutoff summaries must correspond to observations available before `operational_freeze_utc`.

Proxy formulas:

```text
station_temp_latest_mean_c = mean(latest air temp across stations)
station_temp_latest_range_c = max(latest air temp) - min(latest air temp)
station_dewpoint_latest_mean_c = mean(latest dewpoint)
station_temp_dewpoint_spread_mean_c = mean(air temp - dewpoint)
station_pressure_latest_mean_hpa = mean(sea-level pressure)
station_wind_speed_latest_mean_mps = mean(wind speed)
station_disagreement_index = station_temp_latest_range_c + 0.5 * std(latest dewpoint)
station_temp_anomaly_14d_mean_c = mean(station latest temp - station rolling 14-day prior latest temp mean)
station_dewpoint_change_1d_mean_c = mean(station latest dewpoint - station prior-day latest dewpoint)
station_pressure_change_1d_mean_hpa = mean(station latest pressure - station prior-day latest pressure)
coastal_inland_temp_spread_proxy_c = mean(coastal_proxy temp) - mean(inland_proxy temp), NULL unless both groups have >=2 stations
```

Minimum station support:

```text
at least 5 total stations for network features
at least 2 stations per group for group spread features
```

### 12.5 Scoreboard treatment

Station features may appear only in:

```text
expert_oof_scoreboard_proxy.csv
system_scoreboard_proxy.csv
station_proxy_ablation.csv
```

They must not appear in:

```text
system_scoreboard_strict.csv
strict frozen candidate formula
strict sealed validation formula
```

Promotion to strict requires a separate artifact proving exact operational vintage or live first-seen availability over at least 730 days.

---

## 13. Expert models

All models MUST use deterministic random seed:

```text
20260626
```

All preprocessing MUST be fold-local.

All numeric features use:

```text
median imputation fitted on training fold only
missingness indicator for every imputed feature
standardization fitted on training fold only for linear models
no standardization for LightGBM
```

All categorical/string flags are one-hot encoded using categories observed in the training fold only. Unknown categories in prediction fold map to all-zero plus `unknown_category_flag`.

### 13.1 Common model library

Use these model classes:

```text
Ridge regression
ElasticNet regression
LightGBM regression
LightGBM quantile regression
LightGBM binary classifier
```

If `lightgbm` is missing, Codex MUST add it to project requirements. The first full implementation requires LightGBM.

### 13.2 Hyperparameter grids

Regression grid for small/medium feature sets:

```yaml
ridge:
  alpha: [0.1, 1.0, 10.0, 100.0]

elastic_net:
  alpha: [0.001, 0.01, 0.1]
  l1_ratio: [0.1, 0.5]

lightgbm_l1:
  objective: regression_l1
  n_estimators: [100, 250]
  learning_rate: [0.03]
  num_leaves: [7, 15]
  max_depth: [3, 5]
  min_child_samples: [50, 100]
  subsample: [0.8]
  colsample_bytree: [0.8]
  reg_lambda: [1.0, 10.0]
  random_state: 20260626
```

For each expert/fold, select the best model by inner validation MAE. If two candidates differ by less than `0.005°C` MAE, pick the simpler model in this order:

```text
Ridge
ElasticNet
LightGBM with fewer leaves
LightGBM with more leaves
```

### 13.3 E0 — `E0_OFFICIAL_RAW`

Target variable:

```text
target_tmax_c
```

Prediction:

```text
official_forecast_max_c
```

Training: none.

Available when official anchor exists.

Promotion: always baseline, never removed.

### 13.4 E1 — `E1_OFFICIAL_RESIDUAL`

Target variable:

```text
official_residual_c = target_tmax_c - official_forecast_max_c
```

Input features:

```text
official_forecast_min_c
official_forecast_max_c
official_forecast_range_c
official_forecast_midpoint_c
revision features
text token flags
target-memory features
online residual-memory features
month, day-of-year sin/cos, season
source-era flags if present
```

Training range:

```text
R0: 2005-01-01 through 2023-12-31 OOF folds
R1: 2021-03-23 through 2023-12-31 OOF folds
```

Output:

```text
prediction_residual_c
prediction_tmax_c = official_forecast_max_c + clipped(prediction_residual_c, -0.7, +0.7)
```

Promotion condition:

```text
OOF MAE improves official raw by >= 0.010°C on identical rows
P90 absolute error worsens by <= 0.020°C
negative controls pass
```

If promotion fails, E1 remains available for diagnostics but its router cap is `0`.

### 13.5 E2 — `E2_TARGET_MEMORY`

Target variable:

```text
target_tmax_c
```

Input features:

```text
all target-memory features from Section 8
calendar sin/cos
causal climatology
```

Training range:

```text
1884-derived history is used for features.
OOF scoring begins 2005-01-01 for R0 and 2021-03-23 for R1.
```

Model class:

```text
Ridge, ElasticNet, LightGBM_l1 grid
```

Promotion condition:

```text
prediction exists for >= 99% of R0 rows
beats causal climatology baseline by >= 0.050°C MAE
negative controls pass
```

### 13.6 E3 — `E3_STATION_PROXY`

Status:

```text
RESEARCH_PROXY only
strict router weight forced to zero
```

Target variable:

```text
official_residual_c when official exists
target_tmax_c only for station-only diagnostic report
```

Input features:

```text
station_proxy_features
selected target-memory features
official anchor features for residual version
calendar features
```

Training range:

```text
source-specific station rows through 2023, with OOF scoring on rows where official anchor exists
```

Model class:

```text
LightGBM_l1 with max_depth <= 3
```

Output:

```text
station_proxy_residual_c
station_proxy_tmax_c = official_forecast_max_c + clipped(station_proxy_residual_c, -0.5, +0.5)
```

Promotion condition for proxy scoreboard:

```text
improves official raw by >= 0.010°C on proxy identical rows
no strict promotion in first implementation
```

### 13.7 E4 — `E4_GFS_MOS`

Target variable:

```text
gfs_residual_c = target_tmax_c - gfs_center_tmax_c
```

Input features:

```text
gfs center daily features
gfs spatial features
target-memory features
calendar features
official-minus-gfs contradiction when official exists
```

Training range:

```text
2021-03-23 through 2023-12-31
```

OOF folds:

```text
Fold R1_A: train 2021-03-23..2021-12-31, test 2022-01-01..2022-06-30
Fold R1_B: train 2021-03-23..2022-06-30, test 2022-07-01..2022-12-31
Fold R1_C: train 2021-03-23..2022-12-31, test 2023-01-01..2023-06-30
Fold R1_D: train 2021-03-23..2023-06-30, test 2023-07-01..2023-12-31
```

Prediction:

```text
gfs_mos_tmax_c = gfs_center_tmax_c + clipped(predicted_gfs_residual_c, -1.0, +1.0)
```

Promotion condition:

```text
beats raw gfs_center_tmax_c by >= 0.030°C MAE
has OOF coverage >= 95% of R1 common rows
```

### 13.8 E5 — `E5_GEFS_PROB_MOS`

Target variable:

```text
gefs_residual_c = target_tmax_c - gefs_ens_tmax_median_c
```

Input features:

```text
gefsatmos ensemble features
gefsatmosmean deterministic/mean fields
GEFS spread features
calendar features
target-memory features
official-minus-GEFS contradiction
```

Training range:

```text
2021-03-23 through 2023-12-31 for R1 common rows
```

Prediction:

```text
gefs_mos_tmax_c = gefs_ens_tmax_median_c + clipped(predicted_gefs_residual_c, -1.0, +1.0)
```

Promotion condition:

```text
beats GEFS raw median by >= 0.030°C MAE
probability features have no missing member count below 31 except documented accepted anomalies
```

### 13.9 E6 — `E6_IFS_OPER_SHADOW`

Status:

```text
SHADOW_CAPPED_CHALLENGER
strict pre2024 weight = 0
```

Target variable:

```text
ifs_residual_c = target_tmax_c - ifs_center_tmax_c
```

Training:

```text
not trained on visible labels in first implementation because IFS starts in 2024.
features and shadow predictions are generated.
```

After sealed 2024 open, train on 2024 only with fixed hyperparameter grid and evaluate on 2025 final test. Weight cap `0.10` after pass.

### 13.10 E7 — `E7_IFS_ENS_SHADOW`

Same status as E6. Uses `ifsenfo` ensemble features. Missing member 0 is allowed only with at least 50 available members and a flag.

### 13.11 E8 — AI challengers

Expert IDs:

```text
E8_AIFS_OPER_SHADOW
E8_AIFS_ENS_SHADOW
E8_AIGFS_SFC_SHADOW
E8_GRAPHCAST_SHADOW
E8_FOURCASTNET_SHADOW
```

All are `SHADOW_CAPPED_CHALLENGER` or `SHADOW_SHORT_RANGE`.

Strict pre-2024 weight is zero.

First promotion cap after pass is `0.05` combined.

`aigfspres` and `aigefssfc` are not Tmax experts.

### 13.12 E9 — `E9_CWA_WRF_LIVE_SHADOW`

Status: `LIVE_SHADOW`.

No historical strict training in first implementation.

After 365 prospective exact-first-seen scored days, train a capped live adapter. Until then weight is zero.

### 13.13 E10 — `E10_DIAGNOSTIC_PROXY`

Status: `RESEARCH_PROXY` only.

Uses lagged HKO daily climate and static/diagnostic proxy features only. IGRA direct upper-air values remain excluded unless a cleaned and release-proven table exists.

Weight in strict router: zero.

### 13.14 E11 — `E11_ARWF_LIVE_SHADOW`

Status: `LIVE_SHADOW`.

No historical strict training in first implementation unless exact-vintage ARWF history of at least 365 scored days is already present. If fewer than 365 scored days exist, output placeholder unavailable predictions.

---

## 14. OOF generation mechanics

### 14.1 Fold boundaries

#### R0 folds

Use yearly expanding folds:

```text
R0_2005: train 2000-01-02..2004-12-31, test 2005-01-01..2005-12-31
R0_2006: train 2000-01-02..2005-12-31, test 2006-01-01..2006-12-31
...
R0_2023: train 2000-01-02..2022-12-31, test 2023-01-01..2023-12-31
```

Rows without official anchor are excluded from official-expert OOF but retained for target-memory diagnostics.

#### R1 folds

Use fixed half-year expanding folds:

```text
R1_A: train 2021-03-23..2021-12-31, test 2022-01-01..2022-06-30
R1_B: train 2021-03-23..2022-06-30, test 2022-07-01..2022-12-31
R1_C: train 2021-03-23..2022-12-31, test 2023-01-01..2023-06-30
R1_D: train 2021-03-23..2023-06-30, test 2023-07-01..2023-12-31
```

Minimum training rows:

```text
R0 experts: 1000 rows
R1 experts: 250 rows
specialists: 200 active rows for promotion, otherwise zero correction
```

### 14.2 Retraining cadence

For OOF:

```text
train once per fold, predict all dates in test interval
```

For live:

```text
refit strict experts monthly using all visible non-sealed data up to the previous settled date
online residual states update daily after settlement
router weights use last frozen model until monthly refit
```

### 14.3 Failed folds

If an expert fails in a fold:

```text
write unavailable expert predictions for that fold
unavailable_reason = FOLD_TRAINING_FAILED
router masks the expert to zero
pipeline continues unless all strict experts are unavailable
```

If E0 official raw fails for a fold with official anchor available, stop the pipeline.

### 14.4 OOF coverage requirement

Router training requires:

```text
R0: at least 5000 OOF rows across official and target-memory experts
R1: at least 600 OOF common rows across official, GFS, GEFS, and target-memory experts
```

If R1 has fewer than 600 common rows, R1 is not promoted and final strict system uses R0 for all dates.

---

## 15. Router specification

### 15.1 Router context features

Allowed router context features:

```text
calendar month, season, MAM flag, JJA flag
official forecast max/min/range/midpoint
revision features
text flags
online residual-memory states
expert prediction values
pairwise expert differences
GEFS spread features
GFS spatial gradient features
NWP cloud/rain/radiation features
target-memory volatility and slope
availability masks
```

Proxy router may additionally use station proxy features.

Strict router MUST NOT use proxy-only features.

### 15.2 Expected-error model

For each expert `e`:

```text
loss_e_t = abs(target_tmax_c - expert_prediction_tmax_c)
```

Train model:

```text
expected_error_e_t = h_e(context_t)
```

Model class grid:

```yaml
ridge:
  alpha: [1.0, 10.0, 100.0]
lightgbm_l1:
  objective: regression_l1
  n_estimators: [100]
  learning_rate: [0.03]
  num_leaves: [7]
  max_depth: [3]
  min_child_samples: [75]
  subsample: [0.8]
  colsample_bytree: [0.8]
  reg_lambda: [10.0]
```

Select by inner-fold MAE on predicted absolute error. Ties within `0.005°C` use Ridge.

Clamp expected errors:

```text
min_expected_error = 0.20°C
max_expected_error = 3.00°C
```

### 15.3 Static weight optimization

For router row set, solve:

```text
minimize mean(abs(y_t - sum_e w_e * f_e_t)))
subject to:
  w_e >= 0
  sum_e w_e = 1
  w_e <= expert_cap_e
```

Use SciPy SLSQP with deterministic starting point equal weights over available experts.

Caps:

```text
E0_OFFICIAL_RAW: 0.80
E1_OFFICIAL_RESIDUAL: 0.80
E2_TARGET_MEMORY: 0.40
E4_GFS_MOS: 0.70
E5_GEFS_PROB_MOS: 0.70
E3_STATION_PROXY: 0.00 in strict, 0.40 in proxy
IFS combined: 0.00 in pre2024 strict, 0.10 after sealed adapter pass
AI combined: 0.00 in pre2024 strict, 0.05 after shadow pass
CWA/ARWF combined: 0.00 until live-shadow promotion
```

### 15.4 Dynamic weight formula

For available experts:

```text
raw_dyn_weight_e = exp(-expected_error_e / tau)
dyn_weight_e = raw_dyn_weight_e / sum(raw_dyn_weight_j)
```

Tau grid:

```text
[0.25, 0.35, 0.50, 0.75, 1.00]
```

Lambda grid:

```text
[0.00, 0.25, 0.50]
```

Final weight:

```text
final_weight_e = (1 - lambda) * static_weight_e + lambda * dyn_weight_e
```

Then apply caps again and renormalize.

If a cap changes weights, redistribute excess proportionally among available uncapped experts. If no uncapped expert remains, assign excess to `E0_OFFICIAL_RAW` if available, otherwise `E2_TARGET_MEMORY`.

### 15.5 Grid selection

Router selects `(tau, lambda)` by inner-fold MAE of final weighted router forecast.

Tie rule:

```text
If MAE difference <= 0.005°C, select lower lambda.
If lambda tie remains, select higher tau.
```

This makes the router less aggressive by default.

### 15.6 Missing expert mask

If an expert is unavailable for a snapshot:

```text
static_weight_e = 0
dynamic_weight_e = 0
final_weight_e = 0
```

Remaining weights are renormalized. If no experts are available, the system emits no forecast and writes `NO_AVAILABLE_EXPERT`.

### 15.7 Router pass/fail metrics

A router is promoted only if:

```text
MAE improvement vs best single included expert >= 0.005°C
MAE improvement vs static blend >= 0.005°C OR lambda=0 static blend is selected
P90 absolute error worsening vs best single expert <= 0.020°C
positive MAE lift in at least 50% of calendar-year or half-year OOF folds
negative controls pass
```

If these fail, use static blend if it passes; otherwise use best single expert by OOF MAE.

---

## 16. Specialists

All specialists share this implementation pattern:

```text
detector model: LightGBM binary classifier
correction model: LightGBM regression_l1
benefit model: LightGBM regression_l1
```

Classifier hyperparameters:

```yaml
n_estimators: [100]
learning_rate: [0.03]
num_leaves: [7]
max_depth: [3]
min_child_samples: [75]
subsample: [0.8]
colsample_bytree: [0.8]
reg_lambda: [10.0]
random_state: 20260626
```

Generic activation rule:

```text
activate if:
  regime_probability >= 0.70
  expected_benefit_c >= 0.03
  support_count >= 200
  correction_abs_c >= 0.05
  no_harm_pass = true
else applied_correction_c = 0
```

Individual correction cap:

```text
abs(applied_correction_c) <= 0.25°C
```

Total specialist cap:

```text
abs(sum specialist corrections) <= 0.40°C
```

If the uncapped sum exceeds the cap, scale all specialist corrections proportionally to total absolute cap.

### 16.1 Marine suppression specialist — `S1_MARINE_SUPPRESSION`

Detector target:

```text
1 if official_residual_c <= -0.30 and marine_prior_score is in top 40% of training fold
0 otherwise
```

Marine prior score:

```text
+ high onshore/easterly proxy from NWP u/v projected toward Hong Kong coast
+ high marine_s_far_minus_center cooling signal
+ high inland_nw_far_minus_center warming signal
+ high dewpoint / narrow temp-dewpoint spread
+ high low cloud
+ official forecast max warmer than GEFS median
```

Correction target:

```text
anchor_residual_c = target_tmax_c - router_base_forecast_c
```

Benefit target:

```text
abs_error_base - abs_error_after_candidate_marine_correction
```

Input features:

```text
GFS marine/inland spatial gradients
GEFS median/spread
official-minus-GFS
official-minus-GEFS
low cloud
shortwave
wind u/v
humidity/dewpoint features
target-memory volatility
month/season
```

Expected correction sign: non-positive. If predicted correction is positive, set applied correction to zero.

### 16.2 Weak-wind heat buildup — `S2_WEAK_WIND_HEAT`

Detector target:

```text
1 if official_residual_c >= +0.30 and heat_prior_score is in top 40% of training fold
0 otherwise
```

Heat prior score:

```text
+ low mean wind speed
+ high shortwave
+ low cloud
+ warm 850/925 hPa temperature
+ positive target-memory slope
+ official below GEFS median/GFS MOS
+ dry or widening temperature-dewpoint spread
```

Expected correction sign: non-negative. If predicted correction is negative, set applied correction to zero.

### 16.3 MAM transition — `S3_MAM_TRANSITION`

Detector target:

```text
1 if month in {3,4,5} and abs(official_residual_c) >= 0.50
0 otherwise
```

Input features:

```text
MAM flag
target-memory slopes and volatility
official-minus-target-climatology
official-minus-GFS/GEFS
GEFS spread
cloud/rain/dewpoint features
pressure/ridge features
text flags: showers, cloudy, sunny, hot
```

Correction sign is unrestricted.

Promotion requires leave-one-spring-out validation:

```text
Each spring year in 2021, 2022, 2023 is held out once for NWP-enhanced MAM.
For R0 official-only MAM, hold out each spring year 2005 through 2023.
```

### 16.4 Cloud/rain suppression — `S4_CLOUD_RAIN_SUPPRESSION`

Detector target:

```text
1 if official_residual_c <= -0.30 and cloud_rain_prior_score is in top 40% of training fold
0 otherwise
```

Features:

```text
low cloud mean
downward shortwave / shortwave energy
precip proxy
GEFS spread if present
text flags: rain, showers, thunder, cloudy
humidity/dewpoint features
```

Expected correction sign: non-positive.

### 16.5 Dry subsidence/ridge heating — `S5_DRY_RIDGE_HEAT`

Detector target:

```text
1 if official_residual_c >= +0.30 and ridge_heat_prior_score is in top 40% of training fold
0 otherwise
```

Features:

```text
500 hPa height
850/925 hPa temperature
low cloud
shortwave
low precip
wind speed
GEFS median above official
text flags: fine, sunny, very hot
```

Expected correction sign: non-negative.

### 16.6 High-error tail prevention — `S6_HIGH_ERROR_TAIL`

Detector target:

```text
1 if abs(router_base_residual_c) >= 1.00
0 otherwise
```

Correction target:

```text
target_tmax_c - router_base_forecast_c
```

Benefit target:

```text
abs_error_base - abs_error_after_candidate_tail_correction
```

Features:

```text
expert disagreement index
GEFS spread
official revision instability
text-regime conflict
NWP spatial gradient
station proxy disagreement in proxy version
target-memory volatility
recent online error volatility
```

Correction sign unrestricted.

Activation threshold is stricter:

```text
regime_probability >= 0.80
expected_benefit_c >= 0.05
support_count >= 200
```

No-harm test:

```text
P95 absolute error must improve or remain within +0.01°C of baseline.
MAE on non-activated rows must be identical to baseline.
```

---

## 17. Distributional layer

### 17.1 Training data

Use OOF final system predictions from pre-2024 development rows.

Residual:

```text
system_residual_c = target_tmax_c - final_point_pre_distribution_c
```

### 17.2 Quantile method

Train LightGBM quantile residual models for alphas:

```text
0.10, 0.25, 0.50, 0.75, 0.90
```

Feature set:

```text
final point forecast
router expected expert errors
GEFS spread
expert disagreement
month/season
MAM/JJA flags
specialist activation flags
source availability masks
```

Quantile output:

```text
pXX_c = final_point_pre_distribution_c + predicted_residual_quantile_XX
```

Monotonic repair:

```text
p10 <= p25 <= p50 <= p75 <= p90
```

If violated, sort the five quantile values ascending and record `quantile_monotonic_repair=true`.

The final point forecast for MAE is:

```text
final_point_tmax_c = p50_c
```

If P50 worsens pre-distribution point MAE by more than `0.005°C`, use pre-distribution point as P50 and train intervals around that point.

### 17.3 Expected absolute error

Train LightGBM regression_l1:

```text
target = abs(system_residual_c)
```

Clamp:

```text
expected_abs_error_c between 0.20 and 3.00
```

### 17.4 Threshold probabilities

Thresholds:

```text
20.0°C through 40.0°C in 0.5°C increments
```

For threshold `K`:

```text
prob_ge_K = mean over calibrated residual distribution of final_tmax >= K
```

First implementation approximates distribution as normal around P50:

```text
sigma = expected_abs_error_c / 0.7978845608
prob_ge_K = 1 - NormalCDF((K - p50_c) / sigma)
```

Clamp probabilities to `[0.001, 0.999]`.

### 17.5 Confidence state

```text
HIGH if expected_abs_error_c <= 0.55 and expert_disagreement <= fold-local p50
MEDIUM if expected_abs_error_c <= 0.85
LOW otherwise
```

Percentiles are computed fold-locally in OOF and from all visible training rows in live.

### 17.6 No-trade flag

```text
no_trade_flag = true if confidence_state = LOW
             OR expected_abs_error_c > 1.00
             OR source availability misses official and all core NWP
             OR leakage_status != PASS
```

No-trade flag does not suppress point forecast generation. It marks trading confidence only.

---

## 18. Metrics and scoreboards

### 18.1 Metrics

For predictions `p_i` and labels `y_i`:

```text
error_i = p_i - y_i
abs_error_i = abs(error_i)
MAE = mean(abs_error_i)
RMSE = sqrt(mean(error_i^2))
Bias = mean(error_i)
MedianAE = median(abs_error_i)
P75AE = percentile75(abs_error_i)
P90AE = percentile90(abs_error_i)
P95AE = percentile95(abs_error_i)
LargeError1Rate = mean(abs_error_i >= 1.0)
LargeError2Rate = mean(abs_error_i >= 2.0)
HotUnderforecastRate = mean((y_i >= seasonal_p80_i) AND (p_i < y_i - 0.5))
ColdOverforecastRate = mean((y_i <= seasonal_p20_i) AND (p_i > y_i + 0.5))
```

Seasonal percentiles are causal within the training fold for training metrics and full pre-period for final report.

### 18.2 Required slices

Every scoreboard MUST include:

```text
full period
year
month
season: DJF/MAM/JJA/SON
MAM only
JJA only
source availability group
official forecast max bucket: <20, 20-25, 25-30, 30-33, >=33
GEFS spread tertile when GEFS available
marine specialist active/inactive
weak-wind specialist active/inactive
cloud/rain specialist active/inactive
high-error-tail specialist active/inactive
```

### 18.3 Strict/proxy/live scoreboards

```text
strict_development_oof:
  strict features only, visible labels through 2023-12-31

proxy_development_oof:
  proxy features allowed, visible labels through 2023-12-31

shadow_2024plus_unscored:
  predictions/features only, no labels unless sealed opened

live_shadow:
  prospective predictions, labels joined after settlement only
```

### 18.4 Promotion thresholds

Strict candidate promotion requires:

```text
MAE improvement vs official raw >= 0.015°C on identical rows
RMSE not worse by more than 0.020°C
P90AE not worse by more than 0.020°C
P95AE not worse by more than 0.030°C
Bias absolute value <= official raw bias absolute value + 0.020°C
positive MAE lift in at least 60% of yearly or half-year folds
negative controls pass
```

If no candidate meets all thresholds, freeze the best safe baseline:

```text
E0 official raw or R0 static blend, whichever has lower OOF MAE and passes leakage tests
```

---

## 19. Negative controls and leakage tests

All tests are mandatory.

### 19.1 Shuffled target control

Create a copy of labels shuffled within month across years. Train the strict pipeline on the shuffled labels.

Pass condition:

```text
Shuffled-label MAE must be at least 80% of official raw MAE and must not beat official raw by more than 0.02°C.
```

### 19.2 Lag-shifted NWP control

Shift NWP features by +7 target days while keeping official and target labels unchanged.

Pass condition:

```text
Lag-shifted NWP system must not improve official raw by more than 0.02°C.
```

### 19.3 Post-cutoff injection test

Inject an artificial post-cutoff column named:

```text
leak_test_future_target_tmax_c = target_tmax_c
```

Pass condition:

```text
Feature whitelist rejects the column.
Pipeline fails closed before model training if column appears in strict feature matrix.
```

### 19.4 Outcome-derived feature scan

Scan all feature names for forbidden patterns:

```text
actual
settled
target_tmax
residual
error
overforecast
underforecast
hot_day_underforecast
cold_day_overforecast
label
outcome
```

Allowed exceptions:

```text
prediction_residual_c inside model_oof.expert_prediction output tables only
online residual-memory features computed only from prior settled dates
```

Any forbidden feature in input matrix fails Phase 13.

### 19.5 Future normalization scan

For each fold, verify preprocessing fit dates:

```text
preprocessor_fit_end_date < first_test_date
```

Pass condition: zero violations.

### 19.6 GribStream scope contamination check

Pass condition:

```text
No strict NWP feature row joins to raw_response_object.object_uri outside '%full_tactical_backfill_ok_tmax%'.
```

### 19.7 H24N NWP safety check

Pass condition:

```text
No strict NWP feature row has run_time_utc + interval '6 hours' > formal_cutoff_utc.
```

### 19.8 Sealed-year target access check

Before sealed opening, any query reading target labels for target dates >= `2024-01-01` outside feature storage and prediction-only flows fails.

Pass condition: zero sealed target reads.

### 19.9 Same-row official residual scan

Any raw source column equivalent to:

```text
target_tmax_c - official_forecast_max_c
```

in feature input fails unless it is an online residual state calculated only from dates earlier than the snapshot target date.

---

## 20. Sealed validation protocol

### 20.1 Before opening sealed years

These artifacts MUST exist and be immutable:

```text
artifacts/frozen_candidate_manifest.json
artifacts/final_system_config.yaml
reports/system_scoreboard_strict.csv
reports/negative_control_report.md
reports/leakage_audit.md
```

Manifest MUST include SHA-256 hashes for:

```text
feature schema
model artifacts
training row IDs
source registry
configuration
code git commit
```

### 20.2 Opening 2024

Command:

```bash
python -m hkg_tmax.cli sealed-score --cutoff-id H24N --year 2024 --open-sealed --sealed-release-token <token>
```

Allowed:

```text
score frozen candidate
score frozen baselines
write 2024_validation_scoreboard.csv
write sealed_validation_audit.md
```

Forbidden after seeing 2024:

```text
changing feature list
changing hyperparameter grid
changing router tau/lambda grid
changing specialist thresholds
changing NWP filtering
changing target-memory lag policy
manually excluding bad 2024 slices
```

Pass condition for 2024:

```text
final strict system improves official raw MAE by >= 0.010°C on identical rows
P90AE does not worsen by more than 0.030°C
negative controls remain passed
no sealed leakage event occurs
```

If 2024 fails, freeze failure report and stop. Do not open 2025.

### 20.3 Refit after 2024

If 2024 passes, one refit is allowed:

```text
same code
same feature list
same hyperparameter grid
same thresholds
same router rules
training data extends through 2024
```

This creates `frozen_candidate_manifest_refit_through_2024.json`.

### 20.4 Opening 2025

Command:

```bash
python -m hkg_tmax.cli sealed-score --cutoff-id H24N --year 2025 --open-sealed --sealed-release-token <token>
```

2025 is final historical test. No refit or redesign after 2025 is allowed inside this validation cycle.

Pass condition:

```text
system improves official raw MAE by >= 0.005°C
P95AE not worse by more than 0.030°C
calibration Brier score improves climatology by >= 1% relative for threshold probabilities
```

### 20.5 Treatment of 2026

2026 is prospective/live-shadow territory.

For dates already predicted before settlement, join actuals after settlement and score as prospective.

For dates where predictions were not made before settlement, use replay mode and label as:

```text
2026_historical_replay_not_live
```

---

## 21. Model promotion ladder

A candidate is promoted only if every applicable criterion passes.

### 21.1 Strict model promotion

```text
OOF MAE improvement vs official raw >= 0.015°C
OOF RMSE worsening <= 0.020°C
OOF P90AE worsening <= 0.020°C
OOF P95AE worsening <= 0.030°C
bias_abs <= official_bias_abs + 0.020°C
positive lift in >= 60% of folds
negative controls pass
leakage audit has zero ERROR events
feature provenance complete for 100% of strict features
```

### 21.2 Proxy model promotion

Proxy model can be promoted only to research status:

```text
MAE improvement vs strict system >= 0.010°C on proxy identical rows
strict_allowed=false remains recorded
report clearly labels non-deployable proxy nature
```

Proxy never enters strict frozen candidate in first implementation.

### 21.3 Specialist promotion

```text
active rows >= 200
active rows across >= 3 calendar years when history permits
positive active-slice MAE lift >= 0.030°C
overall MAE not worse by more than 0.005°C
P95AE not worse by more than 0.010°C
correction cap respected
no-harm pass true
```

### 21.4 Shadow source promotion

```text
365 prospective or sealed-scored days minimum for capped entry
730 days minimum for uncapped normal competition
at least two warm seasons and two cool seasons with non-negative lift
negative controls pass
availability proof exists
```

---

## 22. Final system formula

### 22.1 Strict first implementation formula

For target date `T`, first determine router version:

```text
If R1 experts E0/E1/E2/E4/E5 are available: use R1_CORE_GFS_GEFS.
Else if R0 experts are available: use R0_OFFICIAL_LONG_HISTORY.
Else use best available fallback expert in order E0, E2, E4, E5.
```

Base forecast:

```text
base_forecast_c = sum_e(final_weight_e * expert_prediction_tmax_c_e)
```

Specialist correction:

```text
specialist_total_raw_c = sum_s(applied_correction_c_s)
specialist_total_c = clip(specialist_total_raw_c, -0.40, +0.40)
```

Final pre-distribution point:

```text
final_pre_distribution_c = base_forecast_c + specialist_total_c
```

Hard cap relative to official anchor when official exists:

```text
final_pre_distribution_c = clip(final_pre_distribution_c,
                                official_forecast_max_c - 1.20,
                                official_forecast_max_c + 1.20)
```

Distributional median:

```text
final_point_tmax_c = p50_c
```

If distributional P50 fails its pass condition:

```text
final_point_tmax_c = final_pre_distribution_c
```

Rounding for published point:

```text
final_point_tmax_rounded_0p1c = round(final_point_tmax_c, 1)
```

Units: degrees Celsius.

### 22.2 Unavailable expert handling

Unavailable expert weight is zero. Weights are renormalized. No imputed forecast is created for a missing expert.

### 22.3 Strict first implementation included experts

Strict pre-2024 final formula may include only:

```text
E0_OFFICIAL_RAW
E1_OFFICIAL_RESIDUAL
E2_TARGET_MEMORY
E4_GFS_MOS
E5_GEFS_PROB_MOS
promoted specialists using strict features
```

Strict first implementation MUST NOT include:

```text
E3_STATION_PROXY
E6/E7 IFS before sealed promotion
E8 AI challengers before shadow promotion
E9 CWA WRF before live promotion
E10 diagnostic proxy
E11 ARWF before live promotion
```

---

## 23. Live inference

Live inference is in scope as a command, not as a scheduler.

### 23.1 Command

```bash
python -m hkg_tmax.cli predict-live --cutoff-id H24N --target-date YYYY-MM-DD
```

### 23.2 Run time

The command is intended to run between:

```text
14:45 and 14:59 HKT on T-1
```

It MUST refuse to create a live prediction after:

```text
15:00 HKT on T-1
```

unless `--replay` is supplied. Replay predictions are stored separately and are not live predictions.

### 23.3 Output JSON schema

```json
{
  "target_date_hkt": "YYYY-MM-DD",
  "cutoff_id": "H24N",
  "formal_cutoff_utc": "...",
  "prediction_created_at_utc": "...",
  "final_point_tmax_c": 32.43,
  "final_point_tmax_rounded_0p1c": 32.4,
  "p10_c": 31.6,
  "p25_c": 32.0,
  "p50_c": 32.4,
  "p75_c": 32.8,
  "p90_c": 33.2,
  "expected_abs_error_c": 0.62,
  "threshold_probabilities": {"ge_32_5": 0.44},
  "confidence_state": "MEDIUM",
  "no_trade_flag": false,
  "source_availability": {},
  "router_weights": {},
  "specialist_corrections": {},
  "audit_sha256": "..."
}
```

### 23.4 Post-settlement scoring

Command:

```bash
python -m hkg_tmax.cli score-prediction --target-date YYYY-MM-DD --cutoff-id H24N
```

This command may join target label only after the settlement label exists. It then updates online residual states.

### 23.5 Idempotency

If a live prediction already exists for `(target_date_hkt, cutoff_id)`, rerun returns the stored prediction unless `--force-replay` is supplied. `--force-replay` writes to replay tables, not live table.

---

## 24. Required tests

### 24.1 Unit tests

```text
test_timeutils.py:
  H24N cutoff for sample dates returns correct UTC
  operational freeze is exactly 15 minutes before formal cutoff

test_target_memory.py:
  lag2 uses T-2
  lag1 is absent
  rolling windows exclude T-1 and T

test_metrics.py:
  MAE/RMSE/bias/P90 formulas match hand examples

test_nwp_units.py:
  Kelvin, Pa, precipitation, and radiation conversions are correct
```

### 24.2 Integration tests

```text
test_schema_contracts.py:
  required source tables and columns resolve exactly

test_h24n_snapshot.py:
  snapshots have unique IDs and valid cutoff timestamps

test_official_anchor.py:
  latest pre-freeze official row is selected
  post-freeze row is excluded

test_nwp_filter.py:
  smoke rows excluded
  blocked datasets excluded
  unsafe post-buffer rows excluded

test_oof_integrity.py:
  train_end_date < test_start_date for every prediction

test_router_weights.py:
  weights sum to 1
  unavailable expert weight = 0
  caps respected

test_specialist_caps.py:
  individual and total correction caps respected

test_sealed_guard.py:
  sealed-score command fails without token
```

### 24.3 Smoke full pipeline

Command:

```bash
python -m hkg_tmax.cli run-full-pre2024 --cutoff-id H24N --smoke --start-date 2021-03-23 --end-date 2021-06-30
```

Expected result:

```text
pipeline completes through Phase 14
at least E0/E1/E2/E4/E5 predictions exist
scoreboards produced
negative controls run in reduced mode
```

---

## 25. Required final artifacts

Codex MUST produce exactly these top-level reports/artifacts:

```text
reports/source_registry.md
reports/schema_contract_report.md
reports/target_label_coverage.md
reports/official_anchor_coverage.md
reports/official_revision_feature_dictionary.md
reports/target_memory_feature_dictionary.md
reports/nwp_feature_coverage.md
reports/nwp_leakage_filter_report.md
reports/nwp_feature_dictionary.md
reports/station_proxy_feature_dictionary.md
reports/proxy_source_eligibility_report.md
reports/feature_availability_matrix.md
reports/feature_dictionary.md
reports/expert_oof_scoreboard_strict.csv
reports/expert_oof_scoreboard_proxy.csv
reports/router_scoreboard_strict.csv
reports/router_scoreboard_proxy.csv
reports/specialist_scoreboard_strict.csv
reports/specialist_scoreboard_proxy.csv
reports/distribution_scoreboard.csv
reports/calibration_report.md
reports/system_scoreboard_strict.csv
reports/system_scoreboard_proxy.csv
reports/ablation_matrix.csv
reports/negative_control_report.md
reports/leakage_audit.md
reports/ready_for_sealed_validation.md
artifacts/frozen_candidate_manifest.json
artifacts/final_system_config.yaml
artifacts/model_artifact_index.csv
```

Each CSV must include row count, date range, candidate ID, baseline ID, and run ID.

Each MD report must include source row counts, filters applied, exclusions, failures, and pass/fail status.

---

## 26. Coding priorities and constraints

### 26.1 Database vs parquet

PostgreSQL is the source of truth for:

```text
snapshots
features by family
feature matrices as JSONB
OOF predictions
router predictions
specialist predictions
scoreboards
live predictions
audit events
```

Parquet may be used for:

```text
training matrices exported from PostgreSQL
model-ready dense arrays
large intermediate caches
```

Every parquet file must have a manifest row containing:

```text
path
row_count
column_count
sha256
source_sql_hash
run_id
```

### 26.2 Determinism

```text
random_seed = 20260626
thread_count = fixed from config
row ordering = target_date_hkt ascending, snapshot_id ascending
```

### 26.3 Failure behavior

All scripts must be idempotent. Rebuildable tables may be truncated only for the same `run_id` or with explicit `--replace-run`.

All leakage, schema, or source-scope failures are fail-closed.

### 26.4 Provenance

Every feature value must be traceable to:

```text
source table
source row/date/model
cutoff rule
feature formula version
run_id
```

### 26.5 Forbidden implementation shortcuts

Codex MUST NOT:

```text
train one giant all-column model as the final system
fit preprocessing on full history
use 2024+ labels during pre-2024 development
use target_lag1 finalized daily Tmax
use same-row official residual as a feature
use station wind direction until repaired
use IGRA direct fields as deployable predictors
use NWP rows without full tactical source filter
use GribStream timeseries best forecasts for strict historical training
use smoke rows from GribStream forecast_wide
silently fill missing expert predictions with climatology
compare candidates on different row sets without same-row scoreboard
```

---

## 27. Final implementation summary

The first full system is a strict, leakage-safe, multi-expert forecast system.

It is not a rule-only system and not a single all-feature ML model.

The strict first implementation uses:

```text
HKO official forecast anchor
official residual model
target-memory model using T-2 or older labels
GFS MOS model
GEFS ensemble MOS model
expected-error router
promoted strict specialists
distributional calibration
```

It additionally builds, but does not strictly deploy:

```text
station proxy expert
HKO daily climate proxy diagnostics
IFS shadow experts
AI-model shadow experts
CWA WRF live shadow expert
ARWF live shadow placeholder/expert
```

The completion status after Codex finishes this specification is:

```text
READY_FOR_SEALED_VALIDATION
```

No 2024 or 2025 target outcomes are opened by default. Sealed validation is executed only by the explicit guarded command.
