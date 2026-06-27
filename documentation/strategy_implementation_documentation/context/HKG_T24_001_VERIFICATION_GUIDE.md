# HKG-T24-001 Verification Guide, Runbook, and Acceptance Evidence

## Executive Summary

This document records how the Jira foundation was verified and how to rerun it. The final verified state passed compile, lint, type check, pytest with DB-backed contracts, and all three required CLI commands. The DB-backed full snapshot command built the full target-date range from 2000-01-02 through 2026-06-21 after the command wrapper timeout was increased beyond 120 seconds.

Acceptance is not based on a single GribStream example. It covers DSN behavior, CLI manifest writing, schema migration, source registry rows, source discovery, target-label fallback, H24N calendar semantics, snapshot rows, GribStream safe rows, feature-matrix compatibility, raw response compatibility, live/eval scaffolds, validation scaffolds, reports, and tests.

## Reader Orientation

Use this file when you need to answer whether the Jira was accepted, which commands ran, which tests exist, which reports prove coverage, and how to rerun or review the foundation. The foundation deep dive explains code structure. The source safety audit explains DB objects and source eligibility.

## Scope Boundaries

The verification proves the foundation contracts and DB scaffolds. It does not prove later model accuracy, sealed validation performance, router behavior, live forecast publishing, or negative-control model outcomes. The validation and live tables are present because the Jira required scaffolding; their future rows belong to later Jira work.

The test suite intentionally mixes pure unit tests and DB-backed tests. Unit tests can run without a DSN. The DB-backed test is skipped only when both DSN variables are missing; in the final verified run, `HKG_TMAX_DATABASE_URL` was present and the DB-backed assertions ran.

## Source-of-Truth Inputs


| evidence document | bytes | lines | sha256_prefix | reason it mattered |
| --- | --- | --- | --- | --- |
| documentation/strategy_implementation_documentation/context/GRIBSTREAM_FETCHED_DATA_INVENTORY_20260626.md | 21565 | 499 | 422dfdaf095b0e0f | Inventory proving tactical forecast tables, raw response objects, full-run source scope, mixed smoke rows, model coverage, and GribStream dataset roles. |
| documentation/strategy_implementation_documentation/context/GRIBSTREAM_LEAKAGE_SAFE_DB_RETRIEVAL_LEDGER_20260626.md | 28191 | 937 | 9bfa724d3e32c697 | Ledger defining the required H24N safe predicate, source-scope filter, run-time based availability, six-hour buffer, and blocked daily Tmax sources. |
| documentation/strategy_implementation_documentation/context/POSTGRES_STRATEGY_DATASET_INVENTORY_20260626.md | 172236 | 2530 | 0b9a6fb36ea9a8fd | PostgreSQL strategy inventory covering target labels, sealed labels, official forecasts, tactical NWP, diagnostic sources, and why table existence does not equal model eligibility. |


The generated acceptance reports are also source-of-truth outputs. They were written by the CLI after applying migrations and running DB-backed checks. When reports and hand-written notes disagree, rerun the CLI and treat the fresh report plus DB row counts as the current state.

## Requirements-to-Implementation Traceability


| requirement | implementation evidence | status |
| --- | --- | --- |
| Package boundary | `code/src/hkg_t24/` contains the implementation and `code/tests/hkg_t24/` contains the tests. | Implemented |
| DSN precedence | `HKG_TMAX_DATABASE_URL` wins over `HKG_TMAX_DB_DSN`, and missing DSN exits fail-closed with the contract error. | Implemented and unit-tested |
| CLI surface | `phase0-preflight`, `build-source-registry`, and `build-h24n-snapshots` all write `model_core.run_manifest` rows. | Implemented |
| Schema creation | The required `model_*` schemas and foundation tables are created idempotently. | Implemented |
| Type conflict guard | Managed columns are checked before DDL proceeds; conflicts write `reports/schema_conflict_report.md`. | Implemented |
| Source registry | Final-patch row set uses `source_code` primary key, explicit booleans, final status semantics, blocked rows, and support-only rows. | Implemented |
| H24N calendar | Calendar rows cover 2000-01-02 through 2026-06-21 with formal cutoff 15:00 HKT and freeze 14:45 HKT on T-1. | Implemented |
| Snapshot IDs | Snapshot rows use `H24N:YYYY-MM-DD` and partitions `pre2024_development`, `sealed_2024`, `sealed_2025`, `prospective_2026`. | Implemented |
| GribStream safe rows | Forecast rows join raw response objects, filter the full tactical backfill scope, enforce the six-hour buffer, and exclude blocked daily Tmax datasets. | Implemented |
| Feature matrix | `model_features.feature_matrix` is the final physical table; legacy strict and proxy names are recreated as compatibility views. | Implemented |
| Live/eval scaffolds | `model_live.prediction`, `model_live.live_prediction_component`, and `model_eval.system_prediction_component` exist as scaffolds. | Implemented |
| Validation scaffolds | `model_validation.scoreboard` and `model_validation.negative_control_result` were added after rereading the screenshot source docs and Jira notes. | Implemented |
| Reports | All required phase, schema, registry, GribStream, snapshot, live-shadow, leakage, and contract coverage reports are written under `reports/`. | Implemented |
| Training boundary | No full model training, router, sealed validation promotion, or later Jira candidate modeling was implemented. | Intentionally out of scope |

## Change Inventory


| path | bytes | lines | sha256_prefix |
| --- | --- | --- | --- |
| pyproject.toml | 1550 | 72 | 98b1067e135c288a |
| docs/PROJECT_STRUCTURE_AND_CODE_MAP.md | 20518 | 455 | f4274317e7054ab0 |
| code/src/hkg_t24/artifacts/reports.py | 2710 | 71 | 0540ce13aed3be87 |
| code/src/hkg_t24/audit/leakage_events.py | 1025 | 37 | ac2b4d44b31d9910 |
| code/src/hkg_t24/audit/schema_contracts.py | 5144 | 160 | 037cb403051ab500 |
| code/src/hkg_t24/audit/source_registry.py | 3733 | 95 | 718c3fa59d6c9e40 |
| code/src/hkg_t24/cli.py | 10694 | 305 | 46564496d935ae5c |
| code/src/hkg_t24/constants.py | 16566 | 656 | b9a2bb7ab0fdb300 |
| code/src/hkg_t24/db/connection.py | 2000 | 69 | 37c0407f71bd7e74 |
| code/src/hkg_t24/db/ddl.py | 15406 | 370 | 1605676bc84165b2 |
| code/src/hkg_t24/db/migrations.py | 10053 | 276 | c6f8e0380e7aa5f4 |
| code/src/hkg_t24/features/gribstream_safe_rows.py | 5430 | 137 | c79835d9d62a5509 |
| code/src/hkg_t24/features/snapshot_builder.py | 13696 | 338 | e43410a5ace7ccc6 |
| code/src/hkg_t24/features/source_contracts.py | 9386 | 267 | a4856865676e32ce |
| code/src/hkg_t24/timeutils.py | 3369 | 105 | 19309f785db636d4 |
| code/src/hkg_t24/utils/hashing.py | 657 | 26 | 60344bab1ffc8878 |
| code/src/hkg_t24/utils/sql.py | 919 | 31 | d36f3cea90ce3aea |
| code/tests/hkg_t24/test_database_url_priority.py | 1764 | 58 | 7d4ed3a86f0cbe86 |
| code/tests/hkg_t24/test_h24n_contract_policy.py | 2379 | 62 | fd73f2d6ba172180 |
| code/tests/hkg_t24/test_real_db_contracts.py | 594 | 17 | c4ecae3c9de4ae02 |
| code/tests/hkg_t24/test_schema_sql_contract.py | 2338 | 45 | f059794caa4f7682 |
| code/tests/hkg_t24/test_snapshot_builder_synthetic.py | 1206 | 29 | 3875a0c18f154538 |
| config/hkg_t24/hkg_t24_001_foundation.yaml | 427 | 12 | 29e0e79f012f5774 |
| schemas/hkg_t24/hkg_t24_001_schema_versions.json | 542 | 17 | 3081daa00d0cfcfa |
| sql/hkg_t24/hkg_t24_001_foundation_schema.sql | 934 | 25 | e4785f05f6003b42 |
| sql/hkg_t24/hkg_t24_001_gribstream_safe_rows.sql | 913 | 24 | b0d198e3579410b2 |
| reports/phase0_preflight_report.md | 366 | 20 | 1567ba91d971a20c |
| reports/schema_conflict_report.md | 163 | 10 | 627e4579f591b749 |
| reports/source_inventory_report.md | 1044 | 26 | 6658114d4c02a82b |
| reports/source_registry.csv | 6386 | 24 | 80e69380f63397f4 |
| reports/schema_contract_report.md | 1219 | 26 | 798f7b42ef399ef7 |
| reports/schema_migration_source_registry.md | 298 | 10 | 65aefc8672d0fb88 |
| reports/schema_migration_feature_matrix.md | 433 | 15 | 06db191a8f0bde8f |
| reports/gribstream_source_scope_audit.csv | 453 | 15 | 3dbdb2d34ab0cb7a |
| reports/gribstream_source_scope_audit.md | 1341 | 30 | a902cc80491068fb |
| reports/snapshot_coverage_report.csv | 232 | 6 | 121896b747911119 |
| reports/snapshot_coverage_report.md | 440 | 13 | eecc3b1a06c6dc12 |
| reports/live_shadow_availability_report.csv | 155 | 4 | c7b817162e230889 |
| reports/live_shadow_availability_report.md | 344 | 14 | 71b860c4d52097ab |
| reports/leakage_audit_report.md | 244 | 14 | f5488a3e2c2beaf1 |
| reports/jira_001_contract_coverage.md | 2170 | 57 | ffc8a82152573750 |

## Architecture and Control Flow

Verification follows the same flow as operation. First the code is checked without a database using compile, lint, type, and unit tests. Then the DB-backed pytest branch opens PostgreSQL and validates the managed contract. Finally the three CLI commands execute against the same DSN and write reports. The full snapshot command should be run with a timeout above 150 seconds on this DB because the final verified full build took roughly 143.7 seconds.

```mermaid
flowchart TD
  A["compileall"] --> B["ruff"]
  B --> C["mypy"]
  C --> D["pytest with DSN"]
  D --> E["phase0-preflight"]
  E --> F["build-source-registry"]
  F --> G["build-h24n-snapshots"]
  G --> H["inspect reports and DB counts"]
```

## File-by-File Deep Dive


### `pyproject.toml`

Artifact role: durable contract, schema, SQL, config, or project-map evidence for this Jira. Current evidence: 72 lines, 1550 bytes, SHA-256 prefix `98b1067e135c288a`.


### `docs/PROJECT_STRUCTURE_AND_CODE_MAP.md`

Artifact role: durable contract, schema, SQL, config, or project-map evidence for this Jira. Current evidence: 455 lines, 20518 bytes, SHA-256 prefix `f4274317e7054ab0`.


### `code/src/hkg_t24/artifacts/reports.py`

Implementation role: Report path handling lives here. It gives the CLI deterministic paths for root reports, context docs, SQL assets, schema assets, and config assets. Current evidence: 71 lines, 2710 bytes, SHA-256 prefix `0540ce13aed3be87`.


### `code/src/hkg_t24/audit/leakage_events.py`

Implementation role: Leakage event helpers insert fail-closed events and count error events for the leakage audit report. Current evidence: 37 lines, 1025 bytes, SHA-256 prefix `ac2b4d44b31d9910`.


### `code/src/hkg_t24/audit/schema_contracts.py`

Implementation role: This module contains low-level DB discovery helpers for table existence, column discovery, row counts, and canonical source fallback selection. Current evidence: 160 lines, 5144 bytes, SHA-256 prefix `037cb403051ab500`.


### `code/src/hkg_t24/audit/source_registry.py`

Implementation role: Source-registry population lives here. It validates the final row set, upserts rows into PostgreSQL, and writes `reports/source_registry.csv`. Current evidence: 95 lines, 3733 bytes, SHA-256 prefix `718c3fa59d6c9e40`.


### `code/src/hkg_t24/cli.py`

Implementation role: Command orchestration lives here. The parser exposes the three required commands, `_run_with_manifest` wraps every DB-backed operation in a manifest row, and the failure path writes reports even when the DSN contract blocks the command before a connection is opened. Current evidence: 305 lines, 10694 bytes, SHA-256 prefix `46564496d935ae5c`.


### `code/src/hkg_t24/constants.py`

Implementation role: The contract constants live here: cutoff identifiers, date range, schema versions, source registry rows, NWP dataset allow/block sets, feature prefixes, calendar feature names, target-memory names, and the exact DSN warning/error text. Current evidence: 656 lines, 16566 bytes, SHA-256 prefix `b9a2bb7ab0fdb300`.


### `code/src/hkg_t24/db/connection.py`

Implementation role: This module isolates DSN selection and PostgreSQL connection creation. It keeps the precedence rule testable without opening a socket and redacts database URLs for reports. Current evidence: 69 lines, 2000 bytes, SHA-256 prefix `37c0407f71bd7e74`.


### `code/src/hkg_t24/db/ddl.py`

Implementation role: The foundation DDL lives here as executable SQL strings plus expected-column metadata used by the migration conflict guard. It is the code-side source for tables and views under `model_*` schemas. Current evidence: 370 lines, 15406 bytes, SHA-256 prefix `1605676bc84165b2`.


### `code/src/hkg_t24/db/migrations.py`

Implementation role: Migration code applies the schema, checks for incompatible existing columns, migrates old feature-matrix physical relations, writes migration reports, and creates or finishes run-manifest rows. Current evidence: 276 lines, 10053 bytes, SHA-256 prefix `c6f8e0380e7aa5f4`.


### `code/src/hkg_t24/features/gribstream_safe_rows.py`

Implementation role: GribStream safe-row construction lives here. It populates `nwp_safe_row_ledger` using the source-scope join, cutoff id, six-hour buffer, and blocked-source exclusions, then writes GribStream and leakage reports. Current evidence: 137 lines, 5430 bytes, SHA-256 prefix `c79835d9d62a5509`.


### `code/src/hkg_t24/features/snapshot_builder.py`

Implementation role: Snapshot construction lives here. It builds target-memory features, populates cutoff calendar rows, copies development-visible labels, writes H24N snapshot availability rows, and emits snapshot plus live-shadow reports. Current evidence: 338 lines, 13696 bytes, SHA-256 prefix `e43410a5ace7ccc6`.


### `code/src/hkg_t24/features/source_contracts.py`

Implementation role: Source preflight checks live here. It verifies LightGBM, target label fallbacks, official forecast availability, tactical NWP tables, raw response objects, and the full-run source scope. Current evidence: 267 lines, 9386 bytes, SHA-256 prefix `a4856865676e32ce`.


### `code/src/hkg_t24/timeutils.py`

Implementation role: The H24N time policy lives here. It converts target dates into formal cutoffs, operational freezes, partitions, seasons, and stable snapshot ids under the Asia/Hong_Kong timezone. Current evidence: 105 lines, 3369 bytes, SHA-256 prefix `19309f785db636d4`.


### `code/src/hkg_t24/utils/hashing.py`

Implementation role: Hash helpers create SHA-256 values for text, JSON payloads, and files. The source hash columns use this family of deterministic evidence identifiers. Current evidence: 26 lines, 657 bytes, SHA-256 prefix `60344bab1ffc8878`.


### `code/src/hkg_t24/utils/sql.py`

Implementation role: SQL formatting helpers validate identifiers, create qualified names, and render small CSV rows without depending on ad hoc string concatenation at call sites. Current evidence: 31 lines, 919 bytes, SHA-256 prefix `d36f3cea90ce3aea`.


### `code/tests/hkg_t24/test_database_url_priority.py`

Verification role: proves environment-variable precedence, exact warning/error behavior, and fail-closed CLI report creation without a DSN. Current evidence: 58 lines, 1764 bytes, SHA-256 prefix `7d4ed3a86f0cbe86`.


### `code/tests/hkg_t24/test_h24n_contract_policy.py`

Verification role: proves the final H24N clock, partitions, source-registry row compatibility, feature prefix contract, schema version contract, and lag 1 ban. Current evidence: 62 lines, 2379 bytes, SHA-256 prefix `fd73f2d6ba172180`.


### `code/tests/hkg_t24/test_real_db_contracts.py`

Verification role: opens the live PostgreSQL DB when configured and checks source discovery, migration idempotency, feature-matrix view semantics, safe-row view filters, registry population, and snapshots. Current evidence: 17 lines, 594 bytes, SHA-256 prefix `c4ecae3c9de4ae02`.


### `code/tests/hkg_t24/test_schema_sql_contract.py`

Verification role: inspects SQL strings for the physical feature matrix, compatibility views, GribStream filters, raw-response view, live scaffolds, and validation scaffolds. Current evidence: 45 lines, 2338 bytes, SHA-256 prefix `f059794caa4f7682`.


### `code/tests/hkg_t24/test_snapshot_builder_synthetic.py`

Verification role: uses a 120-label synthetic set to prove target-memory lag 2 counts and absence of finalized lag 1 feature names. Current evidence: 29 lines, 1206 bytes, SHA-256 prefix `3875a0c18f154538`.


### `config/hkg_t24/hkg_t24_001_foundation.yaml`

Artifact role: durable contract, schema, SQL, config, or project-map evidence for this Jira. Current evidence: 12 lines, 427 bytes, SHA-256 prefix `29e0e79f012f5774`.


### `schemas/hkg_t24/hkg_t24_001_schema_versions.json`

Artifact role: durable contract, schema, SQL, config, or project-map evidence for this Jira. Current evidence: 17 lines, 542 bytes, SHA-256 prefix `3081daa00d0cfcfa`.


### `sql/hkg_t24/hkg_t24_001_foundation_schema.sql`

Artifact role: durable contract, schema, SQL, config, or project-map evidence for this Jira. Current evidence: 25 lines, 934 bytes, SHA-256 prefix `e4785f05f6003b42`.


### `sql/hkg_t24/hkg_t24_001_gribstream_safe_rows.sql`

Artifact role: durable contract, schema, SQL, config, or project-map evidence for this Jira. Current evidence: 24 lines, 913 bytes, SHA-256 prefix `b0d198e3579410b2`.


### `reports/phase0_preflight_report.md`

Evidence role: DB preflight, LightGBM import, registry constant validation, and warning-level source absence. Current evidence: 20 lines, 366 bytes, SHA-256 prefix `1567ba91d971a20c`.


### `reports/schema_conflict_report.md`

Evidence role: Managed-column type conflict result before DDL proceeds. Current evidence: 10 lines, 163 bytes, SHA-256 prefix `627e4579f591b749`.


### `reports/source_inventory_report.md`

Evidence role: Source table discovery and row-count evidence. Current evidence: 26 lines, 1044 bytes, SHA-256 prefix `6658114d4c02a82b`.


### `reports/source_registry.csv`

Evidence role: Machine-readable final source registry. Current evidence: 24 lines, 6386 bytes, SHA-256 prefix `80e69380f63397f4`.


### `reports/schema_contract_report.md`

Evidence role: Target-label fallback, official forecast table, tactical forecast table, raw response object, and full-run source checks. Current evidence: 26 lines, 1219 bytes, SHA-256 prefix `798f7b42ef399ef7`.


### `reports/schema_migration_source_registry.md`

Evidence role: Registry migration surface and final-patch primary key shape. Current evidence: 10 lines, 298 bytes, SHA-256 prefix `65aefc8672d0fb88`.


### `reports/schema_migration_feature_matrix.md`

Evidence role: Feature-matrix physical-table migration and compatibility-view result. Current evidence: 15 lines, 433 bytes, SHA-256 prefix `06db191a8f0bde8f`.


### `reports/gribstream_source_scope_audit.csv`

Evidence role: Dataset-level scoped/safe/excluded row counts from the safe-row ledger. Current evidence: 15 lines, 453 bytes, SHA-256 prefix `3dbdb2d34ab0cb7a`.


### `reports/gribstream_source_scope_audit.md`

Evidence role: Human-readable GribStream scope and buffer audit. Current evidence: 30 lines, 1341 bytes, SHA-256 prefix `a902cc80491068fb`.


### `reports/snapshot_coverage_report.csv`

Evidence role: Partition-level snapshot availability counts. Current evidence: 6 lines, 232 bytes, SHA-256 prefix `121896b747911119`.


### `reports/snapshot_coverage_report.md`

Evidence role: Human-readable H24N snapshot coverage summary. Current evidence: 13 lines, 440 bytes, SHA-256 prefix `eecc3b1a06c6dc12`.


### `reports/live_shadow_availability_report.csv`

Evidence role: ARWF and CWA WRF live-shadow availability export. Current evidence: 4 lines, 155 bytes, SHA-256 prefix `c7b817162e230889`.


### `reports/live_shadow_availability_report.md`

Evidence role: Live-shadow interpretation and warning-level source absence. Current evidence: 14 lines, 344 bytes, SHA-256 prefix `71b860c4d52097ab`.


### `reports/leakage_audit_report.md`

Evidence role: Leakage-event error count and scope boundary. Current evidence: 14 lines, 244 bytes, SHA-256 prefix `f5488a3e2c2beaf1`.


### `reports/jira_001_contract_coverage.md`

Evidence role: End-to-end Jira 001 coverage claim and required report presence. Current evidence: 57 lines, 2170 bytes, SHA-256 prefix `ffc8a82152573750`.


## Public Interfaces and Contracts

Rerun verification through module commands, not by calling internal functions manually. The accepted commands are `python -m hkg_t24.cli phase0-preflight`, `python -m hkg_t24.cli build-source-registry`, and `python -m hkg_t24.cli build-h24n-snapshots`. Set `HKG_TMAX_DATABASE_URL` for the live DB smoke path. Leave `HKG_TMAX_DB_DSN` unset unless you intentionally want fallback behavior. When both are set, expect the primary URL to win and the warning to be emitted.

The report contract is part of the public interface. The required report names are fixed in `constants.py`; consumers can check presence after a run. `jira_001_contract_coverage.md` is the high-level pass/fail summary, but it should be read with `schema_contract_report.md`, `gribstream_source_scope_audit.md`, and `snapshot_coverage_report.md` for evidence detail.

## Data Model/Persistence/Migration


| object | kind | foundation responsibility |
| --- | --- | --- |
| model_core.run_manifest | table | Audit row per CLI command with command name, code version, git commit, DB hash, status, timestamps, and notes. |
| model_core.source_registry | table | Durable final-patch registry keyed by `source_code` with booleans for strict, proxy, shadow, blocked, live-only, and support-only use. |
| model_core.cutoff_calendar | table | One H24N calendar row per target date with formal cutoff, operational freeze, partition, season, month, day of year, and snapshot id. |
| model_core.target_label | table | Development-visible target Tmax labels copied from the selected canonical fallback table with source hash metadata. |
| model_features.h24n_snapshot | table | Snapshot availability ledger with target date, cutoff, partition, strict NWP flags, live-shadow flags, status, and absent-availability reason code. |
| model_features.nwp_safe_row_ledger | table | GribStream forecast-row ledger that records source scope, safety flag, exclusion reason, dataset, run time, valid time, and raw object link. |
| model_features.feature_matrix | table | Final physical feature-matrix table for later phases; Jira 001 creates the shell and migrates any legacy strict/proxy physical tables into it. |
| model_features.snapshot_feature_matrix_strict | view | Compatibility view over `feature_matrix` constrained to strict schema version. |
| model_features.snapshot_feature_matrix_proxy | view | Compatibility view over `feature_matrix` constrained to proxy schema version. |
| model_features.v_nwp_forecast_wide_compat | view | Compatibility view that exposes the tactical forecast table through the final foundation surface. |
| model_features.v_raw_response_object_compat | view | Compatibility view that maps raw response object hash and creation time to final-patch names. |
| model_features.v_nwp_h24n_safe_rows | view | Read-only safe-row view applying source-scope, cutoff, buffer, and blocked-source filters. |
| model_validation.leakage_audit_event | table | Fail-closed event log for leakage violations and contract errors. |
| model_validation.scoreboard | table | Future validation scoreboard scaffold with candidate, metric, partition, target-date span, run mode, and run manifest link. |
| model_validation.negative_control_result | table | Future negative-control scaffold with control name, candidate, MAE, expected behavior, status, details, and run link. |
| model_audit.schema_contract_audit | table | Schema audit shell for later evidence entries. |
| model_live.prediction | table | Live prediction scaffold with target date, cutoff id, issued time, model candidate, run mode, prediction value, intervals, status, and uniqueness rule. |
| model_live.live_prediction_component | table | Live component scaffold linked to a prediction with source code, component name, value, units, and contribution weight. |
| model_eval.system_prediction_component | table | Evaluation component scaffold that records the later system prediction decomposition by candidate, target, source, and component. |


| object or check | verified value | meaning |
| --- | --- | --- |
| model_core.source_registry | 22 rows | Final registry rows were inserted with `source_code` as the durable identifier. |
| model_core.cutoff_calendar | 9,668 rows | One H24N calendar row per target date from 2000-01-02 through 2026-06-21. |
| model_features.h24n_snapshot | 9,668 rows | Snapshot rows use the `H24N:YYYY-MM-DD` naming contract. |
| model_features.nwp_safe_row_ledger scoped rows | 1,964,157 rows | Rows tied to the `full_tactical_backfill_ok_tmax` acquisition scope. |
| model_features.nwp_safe_row_ledger safe rows | 1,858,133 rows | Rows that satisfy the H24N cutoff and source allow/block rules. |
| model_validation.leakage_audit_event ERROR count | 0 | No fail-closed leakage event was recorded in the final verified run. |
| model_core.target_label | 8,765 rows | Development-visible target labels from 2000-01-02 through 2023-12-31. |
| model_features.v_raw_response_object_compat | exists | Compatibility view exposes final response hash and creation timestamp aliases. |
| model_validation.scoreboard | exists | Required scaffold exists without training or scoring later Jira models. |
| model_validation.negative_control_result | exists | Required negative-control scaffold exists without executing later validation design. |


The verified database state proves that the new schema objects exist and that the full date range was built. It also proves that the final corrective pass added the raw-response compatibility view plus the validation scaffold tables. Without those, the screenshot evidence and final patch would still have an unaddressed gap.

## Error Handling/Edge Cases

Verification includes negative paths. `test_missing_database_dsn_error_is_exact` checks the exact missing-DSN error. `test_cli_fails_closed_without_dsn` changes the repo root to a temporary directory, clears both DSN variables, runs `phase0-preflight`, expects exit code 1, and confirms that reports are still written. The schema SQL tests catch accidental removal of required filters and scaffolds before a DB run hides the problem.

The full CLI run had one operational edge case: a 120 second wrapper timeout expired before the full snapshot command finished. The command was rerun with a 300 second limit and passed. That is not a data-contract failure; it is a wrapper-duration finding for this DB volume.

## Security, Privacy, and Credentials

Verification outputs must not expose the raw DSN. The reports redact credentials, and the final documentation references only the redacted form. If rerunning commands in a shell transcript, avoid pasting password-bearing connection strings into committed files. The DB hash in `run_manifest` is enough to correlate runs without leaking the secret.

## Performance Notes

Use a generous timeout for `build-h24n-snapshots`. On the verified DB it processes 9,668 snapshots and more than 1.9 million scoped GribStream rows. A quick smoke run over a narrow date range is useful before the full build when schema changes have just been made, but final acceptance requires the full 2000-01-02 through 2026-06-21 range.

## Configuration

Recommended verification environment:

```powershell
$env:PYTHONPATH='code/src'
$env:HKG_TMAX_DATABASE_URL='postgresql://***:***@127.0.0.1:5432/hkg_tmax_research'
```

Do not set `HKG_TMAX_DB_DSN` at the same time unless you are verifying the dual-env warning branch. The tested command form uses `python -m hkg_t24.cli`; the console script entry point exists in `pyproject.toml` for package-style invocation.

## Testing and Verification Evidence


| command | result | evidence |
| --- | --- | --- |
| python -m compileall -q code/src/hkg_t24 code/tests/hkg_t24 | PASS | Compiled the new package and new test tree without Python syntax errors. |
| python -m ruff check code/src/hkg_t24 code/tests/hkg_t24 | PASS | Lint completed cleanly for the HKG-T24 package and tests. |
| python -m mypy code/src/hkg_t24 code/tests/hkg_t24 | PASS | Type checking returned `Success: no issues found in 30 source files`. |
| $env:HKG_TMAX_DATABASE_URL='postgresql://***:***@127.0.0.1:5432/hkg_tmax_research'; python -m pytest code/tests/hkg_t24 -q | PASS | Pytest ran the unit tests plus the DB-backed contracts and reported 16 passed tests. |
| $env:HKG_TMAX_DATABASE_URL='postgresql://***:***@127.0.0.1:5432/hkg_tmax_research'; python -m hkg_t24.cli phase0-preflight | PASS | CLI preflight created or refreshed the managed schemas, checked LightGBM, checked source tables, and wrote the phase 0 reports. |
| $env:HKG_TMAX_DATABASE_URL='postgresql://***:***@127.0.0.1:5432/hkg_tmax_research'; python -m hkg_t24.cli build-source-registry | PASS | CLI populated the final-patch source registry and wrote registry plus source inventory reports. |
| $env:HKG_TMAX_DATABASE_URL='postgresql://***:***@127.0.0.1:5432/hkg_tmax_research'; python -m hkg_t24.cli build-h24n-snapshots | PASS | CLI built the full 2000-01-02 through 2026-06-21 H24N snapshot surface after an earlier 120 second wrapper timeout was rerun with a longer limit. |


The pytest suite has these acceptance anchors:


| test file | acceptance covered |
| --- | --- |
| code/tests/hkg_t24/test_database_url_priority.py | DSN priority, fallback, missing-DSN error text, fail-closed CLI reports. |
| code/tests/hkg_t24/test_h24n_contract_policy.py | H24N cutoff/freeze, partitions, snapshot id, registry rows, feature prefixes, schema versions, lag 1 ban. |
| code/tests/hkg_t24/test_schema_sql_contract.py | Feature-matrix physical/view contract, safe-row filters, raw-response compatibility, live/eval scaffolds, validation scaffolds. |
| code/tests/hkg_t24/test_snapshot_builder_synthetic.py | 120-label synthetic target-memory counts and lag 2 naming. |
| code/tests/hkg_t24/test_real_db_contracts.py | DB-backed source discovery, migrations, registry, views, safe rows, and snapshots when DSN exists. |

## Report Interpretation


| report path | status | bytes | what to read it for |
| --- | --- | --- | --- |
| reports/phase0_preflight_report.md | PASS | 366 | DB preflight, LightGBM import, registry constant validation, and warning-level source absence. |
| reports/schema_conflict_report.md | PASS | 163 | Managed-column type conflict result before DDL proceeds. |
| reports/source_inventory_report.md | PASS | 1044 | Source table discovery and row-count evidence. |
| reports/source_registry.csv | recorded | 6386 | Machine-readable final source registry. |
| reports/schema_contract_report.md | PASS | 1219 | Target-label fallback, official forecast table, tactical forecast table, raw response object, and full-run source checks. |
| reports/schema_migration_source_registry.md | PASS | 298 | Registry migration surface and final-patch primary key shape. |
| reports/schema_migration_feature_matrix.md | PASS | 433 | Feature-matrix physical-table migration and compatibility-view result. |
| reports/gribstream_source_scope_audit.csv | recorded | 453 | Dataset-level scoped/safe/excluded row counts from the safe-row ledger. |
| reports/gribstream_source_scope_audit.md | PASS | 1341 | Human-readable GribStream scope and buffer audit. |
| reports/snapshot_coverage_report.csv | recorded | 232 | Partition-level snapshot availability counts. |
| reports/snapshot_coverage_report.md | PASS | 440 | Human-readable H24N snapshot coverage summary. |
| reports/live_shadow_availability_report.csv | recorded | 155 | ARWF and CWA WRF live-shadow availability export. |
| reports/live_shadow_availability_report.md | PASS | 344 | Live-shadow interpretation and warning-level source absence. |
| reports/leakage_audit_report.md | PASS | 244 | Leakage-event error count and scope boundary. |
| reports/jira_001_contract_coverage.md | PASS | 2170 | End-to-end Jira 001 coverage claim and required report presence. |


Read `phase0_preflight_report.md` first. It should say `PASS`, LightGBM import passed, and source registry constants validated. Then read `schema_contract_report.md`; it should show `label_core.hko_daily_tmax` selected as the target fallback, `public.hko_historical_forecasts_2000_2026` present, tactical NWP tables present, raw response object present, and full tactical scoped rows counted. Then read `gribstream_source_scope_audit.md` to verify scoped, safe, excluded, and target-day counts by dataset. Then read `snapshot_coverage_report.md` to verify 8,765 pre-2024 development snapshots, 366 sealed 2024 snapshots, 365 sealed 2025 snapshots, and 172 prospective 2026 snapshots.

## Report-by-Report Review Guidance


### `reports/phase0_preflight_report.md`

`phase0_preflight_report.md` is part of the required Jira report set. Read the status first, then the messages and warnings. In the verified run it records a pass, LightGBM import success, source-registry constant validation, and ARWF warning-level absence. The file is regenerated by the CLI, so stale content should be resolved by rerunning the command chain rather than editing the report by hand.

### `reports/schema_conflict_report.md`

`schema_conflict_report.md` is part of the required Jira report set. A clean pass means managed columns did not conflict before migrations. If this report lists a conflict, stop and fix the DB shape before rerunning DDL. The file is regenerated by the CLI, so stale content should be resolved by rerunning the command chain rather than editing the report by hand.

### `reports/source_inventory_report.md`

`source_inventory_report.md` is part of the required Jira report set. Use this report to understand source discovery. It complements the registry; discovery says what exists, registry says how it may be used. The file is regenerated by the CLI, so stale content should be resolved by rerunning the command chain rather than editing the report by hand.

### `reports/source_registry.csv`

`source_registry.csv` is part of the required Jira report set. Use the CSV for machine-readable source policy. It should contain 22 rows in the verified foundation state. The file is regenerated by the CLI, so stale content should be resolved by rerunning the command chain rather than editing the report by hand.

### `reports/schema_contract_report.md`

`schema_contract_report.md` is part of the required Jira report set. This report records canonical target-label fallback, official forecast archive availability, tactical forecast table checks, raw response object checks, and full-run source scope counts. The file is regenerated by the CLI, so stale content should be resolved by rerunning the command chain rather than editing the report by hand.

### `reports/schema_migration_source_registry.md`

`schema_migration_source_registry.md` is part of the required Jira report set. Review this when changing registry columns or primary key semantics. The file is regenerated by the CLI, so stale content should be resolved by rerunning the command chain rather than editing the report by hand.

### `reports/schema_migration_feature_matrix.md`

`schema_migration_feature_matrix.md` is part of the required Jira report set. Review this when old strict/proxy physical matrices could exist in a DB. The file is regenerated by the CLI, so stale content should be resolved by rerunning the command chain rather than editing the report by hand.

### `reports/gribstream_source_scope_audit.csv`

`gribstream_source_scope_audit.csv` is part of the required Jira report set. Use the CSV for dataset-level scoped, safe, excluded, and target-day counts. The file is regenerated by the CLI, so stale content should be resolved by rerunning the command chain rather than editing the report by hand.

### `reports/gribstream_source_scope_audit.md`

`gribstream_source_scope_audit.md` is part of the required Jira report set. Use the Markdown report to verify the human-readable filter statement and publication-buffer caveat. The file is regenerated by the CLI, so stale content should be resolved by rerunning the command chain rather than editing the report by hand.

### `reports/snapshot_coverage_report.csv`

`snapshot_coverage_report.csv` is part of the required Jira report set. Use this CSV for partition-level snapshot counts and source availability flags. The file is regenerated by the CLI, so stale content should be resolved by rerunning the command chain rather than editing the report by hand.

### `reports/snapshot_coverage_report.md`

`snapshot_coverage_report.md` is part of the required Jira report set. Use the Markdown summary to confirm the full partition split. The file is regenerated by the CLI, so stale content should be resolved by rerunning the command chain rather than editing the report by hand.

### `reports/live_shadow_availability_report.csv`

`live_shadow_availability_report.csv` is part of the required Jira report set. Use this CSV to inspect ARWF and CWA WRF live-shadow availability state. The file is regenerated by the CLI, so stale content should be resolved by rerunning the command chain rather than editing the report by hand.

### `reports/live_shadow_availability_report.md`

`live_shadow_availability_report.md` is part of the required Jira report set. Use this report to confirm live-shadow absence is warning-level for Jira 001. The file is regenerated by the CLI, so stale content should be resolved by rerunning the command chain rather than editing the report by hand.

### `reports/leakage_audit_report.md`

`leakage_audit_report.md` is part of the required Jira report set. Use this report to confirm zero leakage error events after the final run. The file is regenerated by the CLI, so stale content should be resolved by rerunning the command chain rather than editing the report by hand.

### `reports/jira_001_contract_coverage.md`

`jira_001_contract_coverage.md` is part of the required Jira report set. Use this as the top-level acceptance summary after detailed report checks pass. The file is regenerated by the CLI, so stale content should be resolved by rerunning the command chain rather than editing the report by hand.


## Acceptance Evidence Expansion

| acceptance item | status | evidence | review note |
| --- | --- | --- | --- |
| Package boundary | Implemented | `code/src/hkg_t24/` contains the implementation and `code/tests/hkg_t24/` contains the tests. | Covered by tests, reports, DB checks, or explicit out-of-scope boundary. |
| DSN precedence | Implemented and unit-tested | `HKG_TMAX_DATABASE_URL` wins over `HKG_TMAX_DB_DSN`, and missing DSN exits fail-closed with the contract error. | Covered by tests, reports, DB checks, or explicit out-of-scope boundary. |
| CLI surface | Implemented | `phase0-preflight`, `build-source-registry`, and `build-h24n-snapshots` all write `model_core.run_manifest` rows. | Covered by tests, reports, DB checks, or explicit out-of-scope boundary. |
| Schema creation | Implemented | The required `model_*` schemas and foundation tables are created idempotently. | Covered by tests, reports, DB checks, or explicit out-of-scope boundary. |
| Type conflict guard | Implemented | Managed columns are checked before DDL proceeds; conflicts write `reports/schema_conflict_report.md`. | Covered by tests, reports, DB checks, or explicit out-of-scope boundary. |
| Source registry | Implemented | Final-patch row set uses `source_code` primary key, explicit booleans, final status semantics, blocked rows, and support-only rows. | Covered by tests, reports, DB checks, or explicit out-of-scope boundary. |
| H24N calendar | Implemented | Calendar rows cover 2000-01-02 through 2026-06-21 with formal cutoff 15:00 HKT and freeze 14:45 HKT on T-1. | Covered by tests, reports, DB checks, or explicit out-of-scope boundary. |
| Snapshot IDs | Implemented | Snapshot rows use `H24N:YYYY-MM-DD` and partitions `pre2024_development`, `sealed_2024`, `sealed_2025`, `prospective_2026`. | Covered by tests, reports, DB checks, or explicit out-of-scope boundary. |
| GribStream safe rows | Implemented | Forecast rows join raw response objects, filter the full tactical backfill scope, enforce the six-hour buffer, and exclude blocked daily Tmax datasets. | Covered by tests, reports, DB checks, or explicit out-of-scope boundary. |
| Feature matrix | Implemented | `model_features.feature_matrix` is the final physical table; legacy strict and proxy names are recreated as compatibility views. | Covered by tests, reports, DB checks, or explicit out-of-scope boundary. |
| Live/eval scaffolds | Implemented | `model_live.prediction`, `model_live.live_prediction_component`, and `model_eval.system_prediction_component` exist as scaffolds. | Covered by tests, reports, DB checks, or explicit out-of-scope boundary. |
| Validation scaffolds | Implemented | `model_validation.scoreboard` and `model_validation.negative_control_result` were added after rereading the screenshot source docs and Jira notes. | Covered by tests, reports, DB checks, or explicit out-of-scope boundary. |
| Reports | Implemented | All required phase, schema, registry, GribStream, snapshot, live-shadow, leakage, and contract coverage reports are written under `reports/`. | Covered by tests, reports, DB checks, or explicit out-of-scope boundary. |
| Training boundary | Intentionally out of scope | No full model training, router, sealed validation promotion, or later Jira candidate modeling was implemented. | Covered by tests, reports, DB checks, or explicit out-of-scope boundary. |


## DB Query Checklist for Reverification

Use these SQL checks after rerunning the CLI against PostgreSQL. They are written as review guidance; do not paste a password-bearing DSN into a committed report.

```sql
SELECT count(*) FROM model_core.source_registry;
SELECT count(*) FROM model_core.cutoff_calendar;
SELECT count(*) FROM model_features.h24n_snapshot;
SELECT count(*) FROM model_features.nwp_safe_row_ledger WHERE source_scope = 'full_tactical_backfill_ok_tmax';
SELECT count(*) FROM model_features.nwp_safe_row_ledger WHERE row_is_safe_h24n;
SELECT count(*) FROM model_validation.leakage_audit_event WHERE severity = 'ERROR';
SELECT count(*) FROM model_core.target_label WHERE label_visible_for_development;
SELECT to_regclass('model_features.v_raw_response_object_compat');
SELECT to_regclass('model_validation.scoreboard');
SELECT to_regclass('model_validation.negative_control_result');
```

Expected verified values from the final run are recorded in the live DB evidence table. A changed value is not automatically a failure, because the source DB can grow. A changed value does require explanation in the report chain before acceptance is claimed again.


## Verification Case Studies

### Missing DSN case

The missing-DSN case is verified at unit and CLI levels. The unit test calls `get_database_url` with an empty mapping and asserts that `DatabaseConfigError` contains the exact contract message. The CLI test clears both DSN environment variables, runs `main(["phase0-preflight"])`, expects exit code 1, and checks that `phase0_preflight_report.md` plus `jira_001_contract_coverage.md` are written in the temporary report directory. This matters because a foundation command with no database must not appear accepted.

### Dual DSN case

The dual-DSN case verifies precedence. The test passes a primary URL in `HKG_TMAX_DATABASE_URL` and a fallback URL in `HKG_TMAX_DB_DSN`. The resolved URL is the primary value and the warning sink receives exactly `Using HKG_TMAX_DATABASE_URL; HKG_TMAX_DB_DSN is present but ignored.` A later refactor that silently selects fallback DSN would fail this test.

### H24N clock case

The H24N clock case verifies the final patch time contract. `formal_cutoff_utc` returns T-1 15:00 HKT converted to UTC, and `operational_freeze_utc` returns T-1 14:45 HKT converted to UTC. The partition test confirms 2023-12-31 is `pre2024_development`, 2024-01-01 is `sealed_2024`, 2025-06-01 is `sealed_2025`, and 2026-06-21 is `prospective_2026`.

### Snapshot id case

The snapshot id case is deliberately simple and strict. For target date 2026-06-21, `calendar_row` returns `H24N:2026-06-21`. That exact format is then persisted in `model_core.cutoff_calendar` and `model_features.h24n_snapshot`. A later change to include hours, UTC dates, or partition names in the id would break both test expectations and DB compatibility.

### Source registry case

The registry case checks final-patch shape rather than row count alone. Tests assert that final row constants exist, dataset prefix mapping matches final values, blocked/support-only rows retain policy booleans, and no deprecated strict-status dependency appears. The live DB run then confirms 22 rows landed in `model_core.source_registry`.

### Feature matrix migration case

The schema SQL test checks that `model_features.feature_matrix` is the physical table and the strict/proxy snapshot names are compatibility views. The DB-backed test then verifies relation behavior against PostgreSQL. This closes the risk that old split physical tables remain as independent write targets.

### Raw response compatibility case

The final correction added `model_features.v_raw_response_object_compat`. The schema test checks that the view exposes `sha256 AS response_sha256` and `retrieved_at_utc AS created_at_utc`. The live DB smoke confirmed the view exists. This matters because the screenshot evidence made raw response objects central to source-scope proof.

### Validation scaffold case

The final correction also added `model_validation.scoreboard` and `model_validation.negative_control_result`. The schema test checks their table definitions without claiming later validation rows. The live DB smoke confirmed both relations exist. This closes the acceptance gap without implementing later scoring phases.

### Target-memory case

The synthetic snapshot test builds 120 ordered label rows. It expects 118 lag 2 values, 112 seven-day lag 2 rolling means, 105 fourteen-day lag 2 rolling means, 89 thirty-day lag 2 rolling means, and no finalized lag 1 target-memory name. This proves the foundation made target memory safe by construction rather than by reviewer promise.

### GribStream safe-row case

The schema SQL test checks that the safe-row view includes raw-object join, source-scope filter, cutoff id, six-hour run-time buffer, and blocked dataset exclusion. The live DB run then writes a ledger with scoped and safe row counts. That two-level verification catches both source-code drift and live data surprises.

### Report presence case

The CLI coverage report enumerates every required report name and marks whether it exists. This gives reviewers a single report-presence audit, but it does not replace reading the detailed reports. A report can exist and still contain a warning, so final review must inspect status and content.

### Timeout case

The first full snapshot command attempt hit a 120 second wrapper timeout. The command was rerun with a 300 second timeout and passed. The correct operational conclusion is to use a larger timeout for full runs on this DB volume, not to mark the data contract as failed.

### No-training case

The tests and reports verify foundation tables, snapshots, safe rows, and scaffolds. They do not verify MAE, candidate ranking, router output, sealed scoreboards, or live predictions. That absence is deliberate: later Jira phases need their own acceptance tests before those claims are made.


## Operational Runbook

1. Open PowerShell at the repo root `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex`.
2. Set `PYTHONPATH` to `code/src` if the package is not installed in editable mode.
3. Set `HKG_TMAX_DATABASE_URL` to the live PostgreSQL DSN.
4. Run compile, ruff, mypy, and pytest exactly as listed in the command evidence table.
5. Run `phase0-preflight` and inspect `reports/phase0_preflight_report.md` plus `reports/schema_conflict_report.md`.
6. Run `build-source-registry` and inspect `reports/source_registry.csv` for 22 rows.
7. Run a narrow `build-h24n-snapshots --start-date 2026-06-01 --end-date 2026-06-21` smoke if you just edited SQL.
8. Run the full `build-h24n-snapshots` command with a timeout above 150 seconds.
9. Inspect `reports/jira_001_contract_coverage.md` and confirm required reports are present.
10. Query DB row counts for source registry, cutoff calendar, snapshots, safe-row ledger, leakage error count, and target labels.

## Compatibility and Rollback

The migration is additive except for legacy feature-matrix physical tables. If old physical `snapshot_feature_matrix_strict` or `snapshot_feature_matrix_proxy` tables exist, their compatible rows are copied into `feature_matrix` before the names are recreated as views. The migration writes `schema_migration_feature_matrix.md`, so rollback review starts there. If rollback is needed, restore the database from backup rather than manually recreating old split physical tables, because the final contract expects a single physical matrix.

## Known Limitations and Follow-Up Work

Accepted foundation state does not mean later modeling is done. The feature matrix shell exists but is not populated with trained candidate features. Live prediction scaffolds exist but do not publish forecasts. Validation scaffolds exist but do not contain final negative-control runs. Official forecast archive discovery passes, but feature extraction from exact-vintage official rows remains later work. ARWF and CWA WRF are live-shadow/prospective paths.

## Reviewer Checklist

- Rerun `python -m pytest code/tests/hkg_t24 -q` with a real DSN before claiming DB acceptance.
- Rerun all three CLI commands after schema edits.
- Confirm `reports/jira_001_contract_coverage.md` says `PASS`.
- Confirm the full target-date range is present in `model_core.cutoff_calendar` and `model_features.h24n_snapshot`.
- Confirm `model_validation.leakage_audit_event` has zero `ERROR` rows after the final run.
- Confirm the safe-row ledger contains excluded rows as audit evidence and safe rows as eligible rows.
- Confirm source registry blocked/support-only booleans match final policy.
- Confirm no later-phase training claim is made from this Jira alone.

## Concrete Examples


### DSN precedence

If both environment variables are present, `get_database_url` returns `HKG_TMAX_DATABASE_URL`, sends the exact warning `Using HKG_TMAX_DATABASE_URL; HKG_TMAX_DB_DSN is present but ignored.`, and never opens the fallback value.

### Missing DSN

If neither DSN variable is present, `phase0-preflight` exits with code 1, writes the fail-closed phase 0 report, and writes contract coverage showing the command did not reach DB-backed acceptance.

### H24N clock rule

For target date 2026-06-21, the formal cutoff is 2026-06-20 15:00 HKT, the operational freeze is 2026-06-20 14:45 HKT, and the snapshot id is `H24N:2026-06-21`.

### Strict source

`gfs` enters the strict set only after the raw response object proves `full_tactical_backfill_ok_tmax` scope and the run time plus six hours is at or before the formal H24N cutoff.

### Shadow source

`ifsoper` is recorded as a shadow challenger: it can be audited and counted, yet it is not a strict v1 model input created by this foundation slice.

### Blocked source

`aigefssfc` may have rows in the tactical tables, but the source registry marks it blocked for daily Tmax because usable coverage is too sparse.

### Support-only source

`aigfspres` is kept as upper-air support evidence and is excluded from daily Tmax strict, proxy, and shadow matrices in this foundation.

### Target memory

Daily target-memory features begin at lag 2. A 120-label synthetic series yields 118 `target__lag2_tmax_c` values and zero finalized lag 1 target-memory names.

### Feature matrix compatibility

If an older DB contains physical `snapshot_feature_matrix_strict` or `snapshot_feature_matrix_proxy` tables, the migration copies compatible rows into `feature_matrix`, drops the old physical relation, and recreates the names as views.

### Live-shadow absence

ARWF absence is warning-level for Jira 001: the command records availability as absent for the foundation rather than failing the schema, because first-seen live collection history belongs to later operation.

## Source Registry Acceptance Snapshot


| source_code | role | strict | shadow | blocked | support | reason |
| --- | --- | --- | --- | --- | --- | --- |
| hko_target_labels | strict_core | True | False | False | False | always included as labels and lagged target memory only |
| hko_official_forecasts | strict_core | True | False | False | False | always included when eligible row exists |
| calendar | strict_core | True | False | False | False | always included |
| gfs | strict_core | True | False | False | False | core strict expert E4 |
| gefsatmosmean | strict_core | True | False | False | False | core strict expert E5 context |
| gefsatmos | strict_core | True | False | False | False | core strict expert E5 ensemble |
| ifsoper | shadow_challenger | False | True | False | False | may enter after sealed protocol |
| ifsenfo | shadow_challenger | False | True | False | False | may enter after sealed protocol; member-0 tracked |
| cwawrf15 | live_shadow | False | True | False | False | prospective only until two seasonal cycles |
| aifsoper | shadow_challenger | False | True | False | False | may enter after sealed protocol |
| aifsenfo | shadow_challenger | False | True | False | False | may enter after sealed protocol |
| aigfssfc | shadow_challenger | False | True | False | False | may enter after sealed protocol |
| aigfspres | support_only | False | False | False | True | support-only source |
| aigefssfc | blocked | False | False | True | False | Poor usable daily Tmax candidate coverage. |
| graphcast | shadow_challenger | False | True | False | False | may enter after sealed protocol |
| fourcastnetgfs | shadow_challenger | False | True | False | False | may enter after sealed protocol |
| nbmoc | blocked | False | False | True | False | No usable HKO-domain daily Tmax coverage. |
| station_network_proxy | proxy_research | False | False | False | False | research proxy only |
| hko_daily_climate_proxy | proxy_research | False | False | False | False | research proxy only |
| igra_upper_air_proxy | support_only | False | False | False | True | support-only proxy |
| tc_best_track_proxy | support_only | False | False | False | True | support-only proxy |
| arwf_live | live_shadow | False | True | False | False | live shadow after enough history |
