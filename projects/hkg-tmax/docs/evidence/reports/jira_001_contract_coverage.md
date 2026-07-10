# HKG-T24-001 Contract Coverage

## Status

PASS

## Command

`generate-oof`

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
- expert factory rows=26292, active=4015, placeholders=22277, artifacts=1

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
- `official_anchor_coverage.md`: present
- `online_state_audit_report.md`: present
- `feature_dictionary.md`: present
- `feature_matrix_coverage_report.md`: present
- `expert_oof_scoreboard.md`: present
- `expert_factory_report.md`: present
- `oof_integrity_report.md`: present
- `model_selection_report.md`: present
- `jira_002_contract_coverage.md`: present
- `router_report.md`: present
- `specialist_report.md`: present
- `distribution_calibration_report.md`: present
- `calibration_report.md`: present
- `system_replay_report.md`: present
- `jira_003_contract_coverage.md`: present
