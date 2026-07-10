# HKG-T24-001 — Data Contract, Source Registry, H24N Snapshot Builder, and Canonical Feature Store Foundation

> **Relocation note (2026-07-10):** Preserve the historical path text below,
> but apply the current path mapping in [the Jira index](../README.md).

## Repository Implementation Location

In this repository, the contract path `src/hkg_t24` resolves to `code/src/hkg_t24`.

All implementation code for this Jira must live under `code/src/hkg_t24/`.

Supporting files must use:

```text
code/tests/hkg_t24/          tests
config/hkg_t24/              configuration
sql/hkg_t24/                 reviewed SQL/query assets
migrations/postgres/         durable PostgreSQL migrations
schemas/hkg_t24/             machine-readable schemas
reports/hkg_t24/             report indexes and non-canonical reports
artifacts/hkg_t24/           artifact indexes and small durable metadata
```

Do not put implementation logic in this Jira folder, root files, reports, notebooks, or ad hoc scripts. Scripts may call the package, but the package owns the implementation logic.

## Objective

Implement the complete data-contract foundation for the HKG T+24 / H24N forecasting system. This ticket creates the required package layout, database connection resolution, source registry, final schema migrations, source discovery, H24N cutoff calendar, target-label contract, official-forecast source contract, GribStream safe-row ledger foundation, canonical `model_features.feature_matrix`, strict/proxy/shadow source scopes, live component tables, and immutable audit/provenance infrastructure.

This ticket must finish with a verified database and source foundation that downstream feature builders, expert models, routers, specialists, validation, sealed scoring, and live inference can use without making any architecture or schema decisions.

## Full Detailed Scope

Create the implementation foundation exactly as specified by the strategy contract and final consistency patch.

The package root must be:

```text
src/hkg_t24/
```

The canonical CLI module must be:

```text
src/hkg_t24/cli.py
```

The canonical database connection resolver must be:

```text
src/hkg_t24/db/connection.py
```

It must support environment variables in this exact priority:

```text
1. HKG_TMAX_DATABASE_URL
2. HKG_TMAX_DB_DSN
```

If neither exists, every database command must fail before doing work with exactly:

```text
ERROR: Database DSN not configured. Set HKG_TMAX_DATABASE_URL or HKG_TMAX_DB_DSN. HKG_TMAX_DATABASE_URL has priority when both are present.
```

If both exist, `HKG_TMAX_DATABASE_URL` wins and the system logs:

```text
Using HKG_TMAX_DATABASE_URL; HKG_TMAX_DB_DSN is present but ignored.
```

The final physical feature matrix table is:

```text
model_features.feature_matrix
```

The previous names are compatibility views only:

```text
model_features.snapshot_feature_matrix_strict
model_features.snapshot_feature_matrix_proxy
```

Codex must not create physical `snapshot_feature_matrix_*` tables in the final implementation. If old physical tables already exist, they must be migrated into `model_features.feature_matrix`, backed up, dropped, and replaced by views.

## Explicit Out of Scope

This ticket does not train ML models, generate expert OOF predictions, train routers, train specialists, run distributional calibration, open sealed labels, or produce final model scoreboards.

This ticket does not fetch new GribStream data. It verifies and uses existing tactical GribStream tables and source-scope rules.

This ticket does not promote station, HKO daily climate, IGRA, TC, ARWF, CWA WRF, IFS, AI, GraphCast, FourCastNet, or any blocked/support-only source into the strict feature scope.

## Required Implementation Steps

1. Create the package/module layout:

```text
src/hkg_t24/
  cli.py
  constants.py
  timeutils.py
  db/connection.py
  db/ddl.py
  db/migrations.py
  audit/source_registry.py
  audit/schema_contracts.py
  audit/leakage_events.py
  features/snapshot_builder.py
  features/source_contracts.py
  features/gribstream_safe_rows.py
  utils/hashing.py
  utils/sql.py
```

2. Implement canonical CLI commands:

```bash
python -m hkg_t24.cli phase0-preflight
python -m hkg_t24.cli build-source-registry
python -m hkg_t24.cli build-h24n-snapshots
```

3. Implement `get_database_url() -> str` in `src/hkg_t24/db/connection.py` using the final environment variable priority.

4. Create schemas:

```sql
CREATE SCHEMA IF NOT EXISTS model_core;
CREATE SCHEMA IF NOT EXISTS model_features;
CREATE SCHEMA IF NOT EXISTS model_oof;
CREATE SCHEMA IF NOT EXISTS model_router;
CREATE SCHEMA IF NOT EXISTS model_validation;
CREATE SCHEMA IF NOT EXISTS model_live;
CREATE SCHEMA IF NOT EXISTS model_audit;
CREATE SCHEMA IF NOT EXISTS model_eval;
```

5. Apply migrations idempotently using `ADD COLUMN IF NOT EXISTS` where applicable.

6. If existing incompatible columns have conflicting types, fail closed and write:

```text
reports/schema_conflict_report.md
```

7. Create final `model_core.source_registry` with these dedicated columns:

```text
source_code
source_family
source_role
feature_prefix
strict_allowed
proxy_allowed
shadow_allowed
blocked
live_only
support_only
unit_semantics_verified
availability_grade
source_time_policy
min_target_date_hkt
max_target_date_hkt
required_source_scope
blocker_reason
promotion_gate
notes
updated_at_utc
```

8. Do not use deprecated `strict_status` in new implementation code. If the old column exists, leave it only for backward compatibility and migrate its meaning into the final boolean columns.

9. Populate required `source_registry` rows exactly:

| source_code | source_family | source_role | feature_prefix | strict_allowed | proxy_allowed | shadow_allowed | blocked | live_only | support_only | unit_semantics_verified | availability_grade | required_source_scope | promotion_gate |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `hko_target_labels` | target | strict_core | `target` | true | false | false | false | false | false | true | EXACT_VINTAGE | NULL | labels and lagged target memory only |
| `hko_official_forecasts` | official | strict_core | `official` | true | false | false | false | false | false | true | EXACT_VINTAGE | NULL | always included when eligible row exists |
| `calendar` | deterministic | strict_core | `calendar` | true | false | false | false | false | false | true | EXACT_VINTAGE | NULL | always included |
| `gfs` | gribstream | strict_core | `gfs` | true | false | false | false | false | false | true | CONSERVATIVE_SCHEDULE | `full_tactical_backfill_ok_tmax` | core strict expert E4 |
| `gefsatmosmean` | gribstream | strict_core | `gefsmean` | true | false | false | false | false | false | true | CONSERVATIVE_SCHEDULE | `full_tactical_backfill_ok_tmax` | E5 context |
| `gefsatmos` | gribstream | strict_core | `gefsens` | true | false | false | false | false | false | true | CONSERVATIVE_SCHEDULE | `full_tactical_backfill_ok_tmax` | E5 ensemble |
| `ifsoper` | gribstream | shadow_challenger | `ifsoper` | false | false | true | false | false | false | true | CONSERVATIVE_SCHEDULE | `full_tactical_backfill_ok_tmax` | may enter after sealed protocol |
| `ifsenfo` | gribstream | shadow_challenger | `ifsens` | false | false | true | false | false | false | true | CONSERVATIVE_SCHEDULE | `full_tactical_backfill_ok_tmax` | may enter after sealed protocol |
| `cwawrf15` | gribstream | live_shadow | `cwawrf15` | false | false | true | false | true | false | true | LIVE_FIRST_SEEN_ONLY | `full_tactical_backfill_ok_tmax` | prospective only |
| `aifsoper` | gribstream | shadow_challenger | `aifsoper` | false | false | true | false | false | false | true | CONSERVATIVE_SCHEDULE | `full_tactical_backfill_ok_tmax` | capped after sealed protocol |
| `aifsenfo` | gribstream | shadow_challenger | `aifsens` | false | false | true | false | false | false | true | CONSERVATIVE_SCHEDULE | `full_tactical_backfill_ok_tmax` | capped after sealed protocol |
| `aigfssfc` | gribstream | shadow_challenger | `aigfssfc` | false | false | true | false | false | false | true | CONSERVATIVE_SCHEDULE | `full_tactical_backfill_ok_tmax` | short-history shadow only |
| `aigfspres` | gribstream | support_only | `aigfspres` | false | false | false | false | false | true | true | CONSERVATIVE_SCHEDULE | `full_tactical_backfill_ok_tmax` | support only, no Tmax expert |
| `aigefssfc` | gribstream | blocked | `aigefssfc` | false | false | false | true | false | false | false | BLOCKED | `full_tactical_backfill_ok_tmax` | blocked until Tmax coverage fixed |
| `graphcast` | gribstream | shadow_challenger | `graphcast` | false | false | true | false | false | false | true | MODEL_RUN_TIME_PROXY_ONLY | `full_tactical_backfill_ok_tmax` | diagnostic shadow only |
| `fourcastnetgfs` | gribstream | shadow_challenger | `fourcastnet` | false | false | true | false | false | false | true | MODEL_RUN_TIME_PROXY_ONLY | `full_tactical_backfill_ok_tmax` | diagnostic shadow through observed archive end |
| `nbmoc` | gribstream | blocked | `nbmoc` | false | false | false | true | false | false | false | BLOCKED | `full_tactical_backfill_ok_tmax` | blocked, zero rows |
| `station_network_proxy` | station | proxy_research | `station` | false | true | false | false | false | false | false | DIAGNOSTIC_ONLY | NULL | proxy reports only |
| `hko_daily_climate_proxy` | diagnostic | proxy_research | `climate` | false | true | false | false | false | false | false | DIAGNOSTIC_ONLY | NULL | lagged diagnostic features only |
| `igra_upper_air_proxy` | diagnostic | support_only | `igra` | false | true | false | false | false | true | false | DIAGNOSTIC_ONLY | NULL | diagnostic report only |
| `tc_best_track_proxy` | diagnostic | support_only | `tc` | false | true | false | false | false | true | false | DIAGNOSTIC_ONLY | NULL | diagnostic report only |
| `arwf_live` | hko_live | live_shadow | `arwf` | false | false | true | false | true | false | true | LIVE_FIRST_SEEN_ONLY | NULL | prospective shadow only |

10. Implement canonical `dataset_code` to `feature_prefix` mapping:

```text
gfs -> gfs
gefsatmosmean -> gefsmean
gefsatmos -> gefsens
ifsoper -> ifsoper
ifsenfo -> ifsens
cwawrf15 -> cwawrf15
aifsoper -> aifsoper
aifsenfo -> aifsens
aigfssfc -> aigfssfc
aigfspres -> aigfspres
aigefssfc -> aigefssfc
graphcast -> graphcast
fourcastnetgfs -> fourcastnet
nbmoc -> nbmoc
```

11. Implement source-table discovery and contract verification for the following primary tables:

```text
public.hko_daily_tmax_target_labels
public.hko_historical_forecasts_2000_2026
nwp_tactical.forecast_wide
nwp_tactical.raw_response_object
nwp_tactical.acquisition_chunk
nwp_tactical.validation_issue
public.noaa_isd_core_observations
public.noaa_isd_station_day_cutoff_summary
public.hko_daily_climate_elements
public.noaa_igra_hkm00045004_key_pressure_levels
public.noaa_igra_hkm00045004_sounding_features
public.hko_tropical_cyclone_best_track
public.hko_arwf_station_daily_forecasts
public.static_geospatial_package_inventory
```

12. If a primary table is absent, apply the discovery contract from the strategy docs. If discovery returns zero or multiple ambiguous candidates, fail Phase 0.

13. Build `model_core.cutoff_calendar` for:

```text
2000-01-02 through 2026-06-21
```

with partitions:

```text
pre2024_development
sealed_2024
sealed_2025
prospective_2026
```

14. Implement H24N time convention:

```text
formal_cutoff_utc = 07:00 UTC on T-1
operational_freeze_utc = 06:45 UTC on T-1
```

15. Use canonical field `target_date_hkt` everywhere in new model schemas. Bare `target_date` is forbidden in new model schemas except as a source alias inside ingestion SQL comments.

16. Create `model_features.h24n_snapshot` with `snapshot_id` format:

```text
H24N:YYYY-MM-DD
```

17. Add snapshot availability flags:

```text
official_available
gfs_available
gefs_available
station_proxy_available
ifs_shadow_available
ai_shadow_available
arwf_live_shadow_available
cwa_live_shadow_available
```

18. Implement canonical GribStream safe-row view `model_features.v_nwp_h24n_safe_rows`, enforcing:

```text
object_uri LIKE '%full_tactical_backfill_ok_tmax%'
run_time_utc + interval '6 hours' <= formal_cutoff_utc
cutoff_id = 'H24N'
dataset_code NOT IN ('nbmoc', 'aigfspres', 'aigefssfc')
```

19. Implement `model_features.nwp_safe_row_ledger`.

20. Implement `model_features.feature_matrix` physical table:

```sql
CREATE TABLE IF NOT EXISTS model_features.feature_matrix (
    target_date_hkt date NOT NULL,
    cutoff_id text NOT NULL,
    feature_scope text NOT NULL,
    schema_version text NOT NULL,
    snapshot_id text NOT NULL,
    features_jsonb jsonb NOT NULL,
    feature_count integer NOT NULL,
    generated_at_utc timestamptz NOT NULL DEFAULT now(),
    source_hash text NOT NULL,
    leakage_status text NOT NULL,
    matrix_status text NOT NULL DEFAULT 'active',
    PRIMARY KEY (target_date_hkt, cutoff_id, feature_scope, schema_version),
    CONSTRAINT feature_matrix_scope_chk CHECK (feature_scope IN ('strict', 'proxy', 'live_shadow')),
    CONSTRAINT feature_matrix_leakage_chk CHECK (leakage_status IN ('passed', 'failed_closed')),
    CONSTRAINT feature_matrix_status_chk CHECK (matrix_status IN ('active', 'superseded', 'failed_closed'))
);
```

21. Create compatibility views:

```sql
CREATE OR REPLACE VIEW model_features.snapshot_feature_matrix_strict AS
SELECT *
FROM model_features.feature_matrix
WHERE feature_scope = 'strict';

CREATE OR REPLACE VIEW model_features.snapshot_feature_matrix_proxy AS
SELECT *
FROM model_features.feature_matrix
WHERE feature_scope = 'proxy';
```

22. Implement `model_live.prediction`, `model_live.live_prediction_component`, and `model_eval.system_prediction_component` using the final consistency patch DDL.

23. Implement source-registry, schema, snapshot, and GribStream audits.

## Required Database Schemas / Tables / Views / Materializations

Create or migrate:

```text
model_core.run_manifest
model_core.source_registry
model_core.cutoff_calendar
model_core.target_label
model_features.h24n_snapshot
model_features.nwp_safe_row_ledger
model_features.feature_matrix
model_features.snapshot_feature_matrix_strict
model_features.snapshot_feature_matrix_proxy
model_features.v_nwp_h24n_safe_rows
model_features.v_nwp_forecast_wide_compat
model_features.v_raw_response_object_compat
model_live.prediction
model_live.live_prediction_component
model_eval.system_prediction_component
model_validation.leakage_audit_event
model_validation.negative_control_result
model_validation.scoreboard
```

Every table must include required primary keys, unique constraints, indexes, timestamps, and provenance columns from the contract.

## Required CLI Commands / Scripts / Modules

Commands:

```bash
python -m hkg_t24.cli phase0-preflight
python -m hkg_t24.cli build-source-registry
python -m hkg_t24.cli build-h24n-snapshots
```

Modules:

```text
src/hkg_t24/db/connection.py
src/hkg_t24/db/ddl.py
src/hkg_t24/db/migrations.py
src/hkg_t24/audit/source_registry.py
src/hkg_t24/audit/schema_contracts.py
src/hkg_t24/audit/leakage_events.py
src/hkg_t24/features/snapshot_builder.py
src/hkg_t24/features/source_contracts.py
src/hkg_t24/features/gribstream_safe_rows.py
```

SQL files:

```text
sql/create_model_schemas.sql
sql/source_contract_queries.sql
sql/h24n_snapshot_views.sql
sql/nwp_safe_row_filter.sql
```

## Required Feature / Model / Artifact Outputs

Foundation artifacts only:

```text
reports/source_inventory_report.md
reports/source_registry.csv
reports/schema_migration_source_registry.md
reports/schema_migration_feature_matrix.md
reports/schema_contract_report.md
reports/gribstream_source_scope_audit.csv
reports/gribstream_source_scope_audit.md
reports/leakage_audit_report.md
reports/snapshot_coverage_report.csv
reports/snapshot_coverage_report.md
reports/live_shadow_availability_report.csv
reports/live_shadow_availability_report.md
reports/phase0_preflight_report.md
```

No model scoreboards are produced by this ticket.

## Required Provenance / Audit / Logging Behavior

Every command must create a row in `model_core.run_manifest` containing:

```text
run_id
run_kind
cutoff_id
started_at_utc
ended_at_utc
status
git_commit
code_version
config_sha256
db_dsn_hash
notes
```

Every fail-closed exclusion must create an audit event in:

```text
model_validation.leakage_audit_event
```

Every GribStream safe-row exclusion must record an exclusion reason in:

```text
model_features.nwp_safe_row_ledger
```

## Required Fail-Closed / Error Behavior

Fail closed when:

```text
database DSN missing
LightGBM import missing
source table discovery ambiguous
required strict table absent with no valid discovery result
source registry cannot be populated exactly
H24N timezone calculation test fails
target-label column contract fails
official forecast clean subset has fewer than 100000 usable local rows
GribStream full tactical source filter returns fewer than 1900000 rows
nwp_tactical.forecast_wide or raw_response_object missing
canonical date field mapping ambiguous
physical snapshot_feature_matrix_* table cannot be safely migrated
source_registry column type conflicts with final DDL
```

Warnings, not failures:

```text
ARWF source table absent
cwawrf15 rows absent or too short
station proxy source absent
diagnostic proxy source absent
```

Warnings must create placeholder/mask behavior for later tickets.

## Leakage-Free / Non-Forward-Looking Requirements

This ticket must guarantee:

- no target-day observations are included in source contracts or snapshots;
- no post-cutoff official forecast row is eligible;
- no post-cutoff GribStream row is safe;
- no future labels are visible in pre-2024 development mode;
- no target-derived or outcome-derived fields are included in strict source feature scopes;
- no sealed-year labels are readable by default commands;
- no global normalization/preprocessing occurs in this ticket;
- no train/test operations occur in this ticket;
- no same-row residual fields are permitted in source feature contracts;
- no GribStream row is strict-safe unless it passes `full_tactical_backfill_ok_tmax` and H24N availability filters;
- no blocked/proxy/shadow source has `strict_allowed=true`.

## Dependencies on Earlier Jiras

None. This is the first ticket and blocks all later tickets.

## Acceptance Criteria

1. `python -m hkg_t24.cli phase0-preflight` completes successfully on the real database.
2. `python -m hkg_t24.cli build-source-registry` creates exactly the required source-registry rows.
3. `python -m hkg_t24.cli build-h24n-snapshots` creates snapshots from `2000-01-02` through `2026-06-21`.
4. `model_core.cutoff_calendar` contains valid UTC/HKT cutoffs for every snapshot date.
5. `model_features.h24n_snapshot` contains unique `snapshot_id = H24N:YYYY-MM-DD`.
6. `model_features.v_nwp_h24n_safe_rows` exists and excludes smoke rows, blocked datasets, non-H24N rows, and insufficient-buffer rows.
7. `model_features.feature_matrix` exists as the only physical feature matrix table.
8. `model_features.snapshot_feature_matrix_strict` and `model_features.snapshot_feature_matrix_proxy` exist only as views.
9. `model_core.source_registry` contains final boolean source-status columns and no new code reads deprecated `strict_status`.
10. `reports/schema_contract_report.md` passes.
11. `reports/gribstream_source_scope_audit.md` proves only `full_tactical_backfill_ok_tmax` rows are strict-eligible.
12. `reports/leakage_audit_report.md` contains zero `ERROR` events for accepted strict rows.
13. ARWF and CWA absence produce warnings and placeholder readiness, not failures.
14. No later code needs to decide source status, feature scope, field naming, or H24N cutoffs.

## Extensive Test Scenarios

Unit tests:

```text
tests/unit/test_h24n_cutoff_time.py
tests/unit/test_database_url_priority.py
tests/unit/test_source_registry_status_mapping.py
tests/unit/test_target_date_hkt_alias_mapping.py
tests/unit/test_gribstream_prefix_mapping.py
```

Integration tests:

```text
tests/integration/test_schema_migrations_realdb.py
tests/integration/test_source_contracts_realdb.py
tests/integration/test_h24n_snapshot_builder_realdb.py
tests/integration/test_nwp_safe_row_view_realdb.py
tests/integration/test_feature_matrix_migration_realdb.py
```

Synthetic tests:

```text
tests/integration/test_snapshot_builder_synthetic.py
```

## Required Smoke Tests

Run:

```bash
python -m hkg_t24.cli phase0-preflight
python -m hkg_t24.cli build-source-registry
python -m hkg_t24.cli build-h24n-snapshots --from-date 2021-04-14 --to-date 2021-05-31
```

Expected minimum real DB counts:

```text
snapshots >= 45
official source contract passed
target label source contract passed
gfs safe-row candidates >= 40 target dates
gefs safe-row candidates >= 40 target dates
zero post-cutoff accepted NWP rows
```

## Required Integration Tests

Integration tests must prove:

- all final schemas exist;
- source-registry rows are exact;
- `HKG_TMAX_DATABASE_URL` priority works;
- fallback to `HKG_TMAX_DB_DSN` works;
- missing DSN error is exact;
- source table discovery is deterministic;
- cutoff calendar converts HKT to UTC correctly;
- feature matrix physical table and views exist as specified;
- GribStream safe-row view excludes all blocked datasets;
- sealed target dates are partitioned and inaccessible to development readers.

## Leakage and Temporal Integrity Tests

This ticket must include:

```text
H24N UTC conversion test: formal_cutoff_utc = T-1 07:00 UTC.
Operational freeze test: freeze is exactly 15 minutes before formal cutoff.
Official source pre-cutoff test: reject official rows after operational_freeze_utc.
GribStream scope test: accepted strict NWP rows join to raw_response_object.object_uri LIKE '%full_tactical_backfill_ok_tmax%'.
GribStream time-buffer test: strict-safe rows satisfy run_time_utc + 6h <= formal_cutoff_utc.
Blocked source test: nbmoc, aigfspres, aigefssfc cannot be strict-safe Tmax sources.
Sealed label guard test: no pre-2024 development command can read 2024+ target labels.
Date alias test: no new model table contains bare target_date instead of target_date_hkt.
```

## Required Negative-Control Tests Where Relevant

This ticket implements the scanner foundation for later negative controls. It must reject feature names containing outcome-derived patterns:

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

Allowed exceptions at this stage:

```text
model_core.target_label.target_tmax_c
audit/report text
future model_oof output columns, not source feature columns
```

## Required Final Artifacts / Reports

```text
reports/source_inventory_report.md
reports/source_registry.csv
reports/schema_migration_source_registry.md
reports/schema_migration_feature_matrix.md
reports/schema_contract_report.md
reports/gribstream_source_scope_audit.csv
reports/gribstream_source_scope_audit.md
reports/leakage_audit_report.md
reports/snapshot_coverage_report.csv
reports/snapshot_coverage_report.md
reports/live_shadow_availability_report.csv
reports/live_shadow_availability_report.md
reports/phase0_preflight_report.md
```

## Definition of Done

This Jira is done when the final schema, source registry, source contracts, H24N cutoff calendar, snapshot table, GribStream safety view, canonical feature matrix table, compatibility views, live component tables, and foundation audits are fully implemented, tested, and pass on real DB smoke data without any leakage or temporal-integrity violation.
