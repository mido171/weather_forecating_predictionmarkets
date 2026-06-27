# HKG-T24-001 Contract Coverage

## Status

PASS

## Command

`build-h24n-snapshots`

## Binding Precedence

Final consistency patch, final clarifications, completion specification, original blueprint, Jira packet.

## Implemented Foundation

- Dedicated `code/src/hkg_t24` package and `code/tests/hkg_t24` tests.
- Exact DSN priority and missing-DSN fail-closed behavior.
- Final source registry booleans and blocked/support-only statuses.
- H24N cutoff calendar, snapshot IDs, and availability flags.
- Strict GribStream full-run source filter and 6-hour H24N buffer.
- Final physical `feature_matrix` with strict/proxy compatibility views.
- Raw GribStream response-object compatibility view with final hash/timestamp aliases.
- Validation scoreboard and negative-control result scaffolds without producing model outputs.
- Live/eval prediction-component table scaffolding required by Jira 001.

## Superseded Contract Items

- `lag1` finalized daily target-memory feature names are forbidden; lag2 is canonical.
- `snapshot_feature_matrix_strict/proxy` are views only, not physical tables.
- `HKG_TMAX_DATABASE_URL` wins over `HKG_TMAX_DB_DSN`.
- LightGBM is mandatory; there is no HistGradientBoosting fallback.

## Details

- database=postgresql://***:***@127.0.0.1:5432/hkg_tmax_research
- built snapshots for 2000-01-02..2026-06-21
- WARNING: ARWF source table absent. E11_ARWF_LIVE_SHADOW will emit placeholder rows with SOURCE_TABLE_ABSENT.

## Required Reports

- `phase0_preflight_report.md`: present
- `schema_conflict_report.md`: present
- `source_inventory_report.md`: present
- `source_registry.csv`: present
- `schema_contract_report.md`: present
- `schema_migration_source_registry.md`: present
- `schema_migration_feature_matrix.md`: present
- `gribstream_source_scope_audit.csv`: present
- `gribstream_source_scope_audit.md`: present
- `snapshot_coverage_report.csv`: present
- `snapshot_coverage_report.md`: present
- `live_shadow_availability_report.csv`: present
- `live_shadow_availability_report.md`: present
- `leakage_audit_report.md`: present
- `jira_001_contract_coverage.md`: present
