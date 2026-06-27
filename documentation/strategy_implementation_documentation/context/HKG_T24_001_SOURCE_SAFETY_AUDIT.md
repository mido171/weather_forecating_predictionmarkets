# HKG-T24-001 Data Contract, Schema, and Source Safety Audit

## Executive Summary

This document explains what the Jira foundation does to make source data safe enough for later feature engineering. The implementation does not say every table in PostgreSQL is model-ready. It classifies each source, creates schema shells with conflict checks, filters GribStream rows through the final safe-row rule, separates development labels from sealed labels, and writes reports that expose coverage, exclusions, and known warnings.

The key decision is that model eligibility is not the same as row existence. `nwp_tactical.forecast_wide` contains the full tactical backfill and older smoke/probe rows. Target labels exist in raw, canonical, and sealed families. Diagnostic station, upper-air, TC, and climate sources have research value but lack final exact-vintage guarantees. HKG-T24-001 turns those facts into explicit DB objects and source-registry booleans so later work cannot accidentally promote unsafe or blocked data.

## Reader Orientation

Use this file to audit data safety, schema shape, source classifications, GribStream scope handling, target-label treatment, and report interpretation. Use the foundation deep dive for package control flow. Use the verification guide for exact commands and acceptance evidence.

The source audit documents named in the screenshot were not optional. They define the source scope, leakage predicate, canonical target-label path, sealed-label boundary, and source families that this Jira had to encode. The implementation was adjusted after rereading those files: the raw-response compatibility view and validation scaffolds were added because the final contract expected them.

## Scope Boundaries

In scope: schemas, tables, views, source registry rows, target-date calendar, target label loading, source table discovery, tactical NWP safe-row ledger, report generation, and compatibility view creation. Out of scope: building final model features from every source family, deciding sealed-validation winners, running negative controls, assigning model weights, and publishing a forecast. This source-safety layer is the foundation for those later phases.

The DB read path is PostgreSQL-backed. The CLI reads from `label_core.hko_daily_tmax`, `public.hko_historical_forecasts_2000_2026`, `nwp_tactical.forecast_wide`, and `nwp_tactical.raw_response_object` when they are available. It writes under `model_core`, `model_features`, `model_validation`, `model_live`, `model_eval`, and `model_audit`.

## Source-of-Truth Inputs


| evidence document | bytes | lines | sha256_prefix | reason it mattered |
| --- | --- | --- | --- | --- |
| documentation/strategy_implementation_documentation/context/GRIBSTREAM_FETCHED_DATA_INVENTORY_20260626.md | 21565 | 499 | 422dfdaf095b0e0f | Inventory proving tactical forecast tables, raw response objects, full-run source scope, mixed smoke rows, model coverage, and GribStream dataset roles. |
| documentation/strategy_implementation_documentation/context/GRIBSTREAM_LEAKAGE_SAFE_DB_RETRIEVAL_LEDGER_20260626.md | 28191 | 937 | 9bfa724d3e32c697 | Ledger defining the required H24N safe predicate, source-scope filter, run-time based availability, six-hour buffer, and blocked daily Tmax sources. |
| documentation/strategy_implementation_documentation/context/POSTGRES_STRATEGY_DATASET_INVENTORY_20260626.md | 172236 | 2530 | 0b9a6fb36ea9a8fd | PostgreSQL strategy inventory covering target labels, sealed labels, official forecasts, tactical NWP, diagnostic sources, and why table existence does not equal model eligibility. |


The GribStream inventory established that `forecast_wide` alone is too broad because it still contains 933 older `batch_smoke_10w` `gefsatmos` rows. The leakage ledger established that availability must be based on `run_time_utc` plus the configured buffer, not `valid_time_utc`. The PostgreSQL inventory established that `label_core.hko_daily_tmax` is the canonical pre-2024 label source, while `sealed_confirmation.hko_daily_tmax` belongs outside development tuning.

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

Data safety is enforced in layers. Source discovery first checks what exists and which columns can support the contract. Schema migration then creates managed objects and compatibility views. Source registry population writes policy rows that tell later code whether a source is strict, proxy, shadow, blocked, live-only, or support-only. GribStream safe-row refresh materializes one ledger row per tactical forecast row with a safety decision. Snapshot generation then converts calendar and ledger facts into one H24N snapshot row per target date.

```mermaid
flowchart LR
  A["Source audits"] --> B["Source contract checks"]
  B --> C["Schema migration"]
  C --> D["Source registry"]
  C --> E["Safe-row ledger"]
  D --> F["Snapshot availability"]
  E --> F
  F --> G["Reports and manifest"]
```

## File-by-File Deep Dive


### `pyproject.toml`

Contract artifact used by the source-safety implementation and review path. Current evidence: 72 lines, 1550 bytes, SHA-256 prefix `98b1067e135c288a`.


### `docs/PROJECT_STRUCTURE_AND_CODE_MAP.md`

Contract artifact used by the source-safety implementation and review path. Current evidence: 455 lines, 20518 bytes, SHA-256 prefix `f4274317e7054ab0`.


### `code/src/hkg_t24/artifacts/reports.py`

Listed for complete coverage of the Jira implementation surface. Its detailed execution role is covered in the foundation or verification document. Current evidence: 71 lines, 2710 bytes, SHA-256 prefix `0540ce13aed3be87`.


### `code/src/hkg_t24/audit/leakage_events.py`

Listed for complete coverage of the Jira implementation surface. Its detailed execution role is covered in the foundation or verification document. Current evidence: 37 lines, 1025 bytes, SHA-256 prefix `ac2b4d44b31d9910`.


### `code/src/hkg_t24/audit/schema_contracts.py`

Provides schema discovery helpers so source-contract checks are DB-backed and not assumptions copied from static docs. Current evidence: 160 lines, 5144 bytes, SHA-256 prefix `037cb403051ab500`.


### `code/src/hkg_t24/audit/source_registry.py`

Validates and upserts the final-patch source registry and exports the CSV evidence file used by reviewers. Current evidence: 95 lines, 3733 bytes, SHA-256 prefix `718c3fa59d6c9e40`.


### `code/src/hkg_t24/cli.py`

Listed for complete coverage of the Jira implementation surface. Its detailed execution role is covered in the foundation or verification document. Current evidence: 305 lines, 10694 bytes, SHA-256 prefix `46564496d935ae5c`.


### `code/src/hkg_t24/constants.py`

Defines the final source registry, dataset-to-prefix mapping, schema versions, strict NWP source set, shadow source set, blocked daily Tmax set, calendar feature whitelist, target-memory whitelist, and forbidden finalized lag 1 terms. This is the policy center for source safety. Current evidence: 656 lines, 16566 bytes, SHA-256 prefix `b9a2bb7ab0fdb300`.


### `code/src/hkg_t24/db/connection.py`

Listed for complete coverage of the Jira implementation surface. Its detailed execution role is covered in the foundation or verification document. Current evidence: 69 lines, 2000 bytes, SHA-256 prefix `37c0407f71bd7e74`.


### `code/src/hkg_t24/db/ddl.py`

Defines the managed schema objects. The file creates source registry, calendar, target label, snapshot, safe-row ledger, feature matrix, validation scaffolds, live scaffolds, eval scaffolds, and compatibility views. It also exposes expected-column metadata for conflict detection. Current evidence: 370 lines, 15406 bytes, SHA-256 prefix `1605676bc84165b2`.


### `code/src/hkg_t24/db/migrations.py`

Runs conflict checks before DDL, applies schema SQL, migrates legacy feature-matrix physical tables into the final table, recreates strict/proxy names as views, and writes migration reports. Current evidence: 276 lines, 10053 bytes, SHA-256 prefix `c6f8e0380e7aa5f4`.


### `code/src/hkg_t24/features/gribstream_safe_rows.py`

Materializes the H24N GribStream safety decision. It records scoped rows, safe rows, excluded rows, and reasons for exclusion without treating raw forecast presence as model permission. Current evidence: 137 lines, 5430 bytes, SHA-256 prefix `c79835d9d62a5509`.


### `code/src/hkg_t24/features/snapshot_builder.py`

Builds the target-date calendar, development-visible target labels, lag 2 target-memory features, snapshot availability rows, snapshot coverage report, and live-shadow availability report. Current evidence: 338 lines, 13696 bytes, SHA-256 prefix `e43410a5ace7ccc6`.


### `code/src/hkg_t24/features/source_contracts.py`

Discovers canonical source tables and produces the schema contract report. It selects `label_core.hko_daily_tmax` as target fallback, checks official HKO forecast availability, validates tactical forecast columns, validates raw response objects, and counts the full-run scope. Current evidence: 267 lines, 9386 bytes, SHA-256 prefix `a4856865676e32ce`.


### `code/src/hkg_t24/timeutils.py`

Listed for complete coverage of the Jira implementation surface. Its detailed execution role is covered in the foundation or verification document. Current evidence: 105 lines, 3369 bytes, SHA-256 prefix `19309f785db636d4`.


### `code/src/hkg_t24/utils/hashing.py`

Listed for complete coverage of the Jira implementation surface. Its detailed execution role is covered in the foundation or verification document. Current evidence: 26 lines, 657 bytes, SHA-256 prefix `60344bab1ffc8878`.


### `code/src/hkg_t24/utils/sql.py`

Listed for complete coverage of the Jira implementation surface. Its detailed execution role is covered in the foundation or verification document. Current evidence: 31 lines, 919 bytes, SHA-256 prefix `d36f3cea90ce3aea`.


### `code/tests/hkg_t24/test_database_url_priority.py`

Listed for complete coverage of the Jira implementation surface. Its detailed execution role is covered in the foundation or verification document. Current evidence: 58 lines, 1764 bytes, SHA-256 prefix `7d4ed3a86f0cbe86`.


### `code/tests/hkg_t24/test_h24n_contract_policy.py`

Listed for complete coverage of the Jira implementation surface. Its detailed execution role is covered in the foundation or verification document. Current evidence: 62 lines, 2379 bytes, SHA-256 prefix `fd73f2d6ba172180`.


### `code/tests/hkg_t24/test_real_db_contracts.py`

Listed for complete coverage of the Jira implementation surface. Its detailed execution role is covered in the foundation or verification document. Current evidence: 17 lines, 594 bytes, SHA-256 prefix `c4ecae3c9de4ae02`.


### `code/tests/hkg_t24/test_schema_sql_contract.py`

Listed for complete coverage of the Jira implementation surface. Its detailed execution role is covered in the foundation or verification document. Current evidence: 45 lines, 2338 bytes, SHA-256 prefix `f059794caa4f7682`.


### `code/tests/hkg_t24/test_snapshot_builder_synthetic.py`

Listed for complete coverage of the Jira implementation surface. Its detailed execution role is covered in the foundation or verification document. Current evidence: 29 lines, 1206 bytes, SHA-256 prefix `3875a0c18f154538`.


### `config/hkg_t24/hkg_t24_001_foundation.yaml`

Contract artifact used by the source-safety implementation and review path. Current evidence: 12 lines, 427 bytes, SHA-256 prefix `29e0e79f012f5774`.


### `schemas/hkg_t24/hkg_t24_001_schema_versions.json`

Contract artifact used by the source-safety implementation and review path. Current evidence: 17 lines, 542 bytes, SHA-256 prefix `3081daa00d0cfcfa`.


### `sql/hkg_t24/hkg_t24_001_foundation_schema.sql`

Contract artifact used by the source-safety implementation and review path. Current evidence: 25 lines, 934 bytes, SHA-256 prefix `e4785f05f6003b42`.


### `sql/hkg_t24/hkg_t24_001_gribstream_safe_rows.sql`

Contract artifact used by the source-safety implementation and review path. Current evidence: 24 lines, 913 bytes, SHA-256 prefix `b0d198e3579410b2`.


### `reports/phase0_preflight_report.md`

Listed for complete coverage of the Jira implementation surface. Its detailed execution role is covered in the foundation or verification document. Current evidence: 20 lines, 366 bytes, SHA-256 prefix `1567ba91d971a20c`.


### `reports/schema_conflict_report.md`

Listed for complete coverage of the Jira implementation surface. Its detailed execution role is covered in the foundation or verification document. Current evidence: 10 lines, 163 bytes, SHA-256 prefix `627e4579f591b749`.


### `reports/source_inventory_report.md`

Listed for complete coverage of the Jira implementation surface. Its detailed execution role is covered in the foundation or verification document. Current evidence: 26 lines, 1044 bytes, SHA-256 prefix `6658114d4c02a82b`.


### `reports/source_registry.csv`

Machine-readable final source registry. Current evidence: 24 lines, 6386 bytes, SHA-256 prefix `80e69380f63397f4`.


### `reports/schema_contract_report.md`

Target-label fallback, official forecast table, tactical forecast table, raw response object, and full-run source checks. Current evidence: 26 lines, 1219 bytes, SHA-256 prefix `798f7b42ef399ef7`.


### `reports/schema_migration_source_registry.md`

Listed for complete coverage of the Jira implementation surface. Its detailed execution role is covered in the foundation or verification document. Current evidence: 10 lines, 298 bytes, SHA-256 prefix `65aefc8672d0fb88`.


### `reports/schema_migration_feature_matrix.md`

Listed for complete coverage of the Jira implementation surface. Its detailed execution role is covered in the foundation or verification document. Current evidence: 15 lines, 433 bytes, SHA-256 prefix `06db191a8f0bde8f`.


### `reports/gribstream_source_scope_audit.csv`

Listed for complete coverage of the Jira implementation surface. Its detailed execution role is covered in the foundation or verification document. Current evidence: 15 lines, 453 bytes, SHA-256 prefix `3dbdb2d34ab0cb7a`.


### `reports/gribstream_source_scope_audit.md`

Human-readable GribStream scope and buffer audit. Current evidence: 30 lines, 1341 bytes, SHA-256 prefix `a902cc80491068fb`.


### `reports/snapshot_coverage_report.csv`

Listed for complete coverage of the Jira implementation surface. Its detailed execution role is covered in the foundation or verification document. Current evidence: 6 lines, 232 bytes, SHA-256 prefix `121896b747911119`.


### `reports/snapshot_coverage_report.md`

Human-readable H24N snapshot coverage summary. Current evidence: 13 lines, 440 bytes, SHA-256 prefix `eecc3b1a06c6dc12`.


### `reports/live_shadow_availability_report.csv`

Listed for complete coverage of the Jira implementation surface. Its detailed execution role is covered in the foundation or verification document. Current evidence: 4 lines, 155 bytes, SHA-256 prefix `c7b817162e230889`.


### `reports/live_shadow_availability_report.md`

Listed for complete coverage of the Jira implementation surface. Its detailed execution role is covered in the foundation or verification document. Current evidence: 14 lines, 344 bytes, SHA-256 prefix `71b860c4d52097ab`.


### `reports/leakage_audit_report.md`

Leakage-event error count and scope boundary. Current evidence: 14 lines, 244 bytes, SHA-256 prefix `f5488a3e2c2beaf1`.


### `reports/jira_001_contract_coverage.md`

Listed for complete coverage of the Jira implementation surface. Its detailed execution role is covered in the foundation or verification document. Current evidence: 57 lines, 2170 bytes, SHA-256 prefix `ffc8a82152573750`.


## Public Interfaces and Contracts

The public source contract is the DB surface, not a Python function alone. Consumers should read `model_core.source_registry` before using a source family. Strict v1 source eligibility means the row has `strict_allowed = true`, is not blocked, is not support-only, and has a time policy compatible with the H24N cutoff. Shadow eligibility does not mean strict training permission. Proxy eligibility means research or diagnostic use until exact-vintage rules are repaired.

The GribStream public contract is `model_features.v_nwp_h24n_safe_rows` and `model_features.nwp_safe_row_ledger`. The safe view applies four rules together: raw-object source scope contains `full_tactical_backfill_ok_tmax`, cutoff id equals `H24N`, dataset code is not one of the blocked daily Tmax datasets, and `run_time_utc + interval '6 hours'` is at or before formal cutoff.

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


The schema migration is idempotent but not permissive. Idempotent means the command can run again without duplicating rows or recreating incompatible tables. It does not mean it overwrites mismatched types. Conflict detection checks expected managed columns first; if a DB has a pre-existing managed object with a conflicting type, the command records the issue and stops.

The source registry has 22 rows in the verified DB. The full table below is the policy surface that later feature code must obey.


| source_code | family | role | prefix | strict | proxy | shadow | blocked | live_only | support_only | grade | required_scope | promotion_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hko_target_labels | target | strict_core | target | True | False | False | False | False | False | EXACT_VINTAGE |  | always included as labels and lagged target memory only |
| hko_official_forecasts | official | strict_core | official | True | False | False | False | False | False | EXACT_VINTAGE |  | always included when eligible row exists |
| calendar | deterministic | strict_core | calendar | True | False | False | False | False | False | EXACT_VINTAGE |  | always included |
| gfs | gribstream | strict_core | gfs | True | False | False | False | False | False | CONSERVATIVE_SCHEDULE | full_tactical_backfill_ok_tmax | core strict expert E4 |
| gefsatmosmean | gribstream | strict_core | gefsmean | True | False | False | False | False | False | CONSERVATIVE_SCHEDULE | full_tactical_backfill_ok_tmax | core strict expert E5 context |
| gefsatmos | gribstream | strict_core | gefsens | True | False | False | False | False | False | CONSERVATIVE_SCHEDULE | full_tactical_backfill_ok_tmax | core strict expert E5 ensemble |
| ifsoper | gribstream | shadow_challenger | ifsoper | False | False | True | False | False | False | CONSERVATIVE_SCHEDULE | full_tactical_backfill_ok_tmax | may enter after sealed protocol |
| ifsenfo | gribstream | shadow_challenger | ifsens | False | False | True | False | False | False | CONSERVATIVE_SCHEDULE | full_tactical_backfill_ok_tmax | may enter after sealed protocol; member-0 tracked |
| cwawrf15 | gribstream | live_shadow | cwawrf15 | False | False | True | False | True | False | LIVE_FIRST_SEEN_ONLY | full_tactical_backfill_ok_tmax | prospective only until two seasonal cycles |
| aifsoper | gribstream | shadow_challenger | aifsoper | False | False | True | False | False | False | CONSERVATIVE_SCHEDULE | full_tactical_backfill_ok_tmax | may enter after sealed protocol |
| aifsenfo | gribstream | shadow_challenger | aifsens | False | False | True | False | False | False | CONSERVATIVE_SCHEDULE | full_tactical_backfill_ok_tmax | may enter after sealed protocol |
| aigfssfc | gribstream | shadow_challenger | aigfssfc | False | False | True | False | False | False | CONSERVATIVE_SCHEDULE | full_tactical_backfill_ok_tmax | may enter after sealed protocol |
| aigfspres | gribstream | support_only | aigfspres | False | False | False | False | False | True | CONSERVATIVE_SCHEDULE | full_tactical_backfill_ok_tmax | support-only source |
| aigefssfc | gribstream | blocked | aigefssfc | False | False | False | True | False | False | BLOCKED | full_tactical_backfill_ok_tmax | blocked until provider/selector fix |
| graphcast | gribstream | shadow_challenger | graphcast | False | False | True | False | False | False | MODEL_RUN_TIME_PROXY_ONLY | full_tactical_backfill_ok_tmax | may enter after sealed protocol |
| fourcastnetgfs | gribstream | shadow_challenger | fourcastnet | False | False | True | False | False | False | MODEL_RUN_TIME_PROXY_ONLY | full_tactical_backfill_ok_tmax | may enter after sealed protocol |
| nbmoc | gribstream | blocked | nbmoc | False | False | False | True | False | False | BLOCKED | full_tactical_backfill_ok_tmax | blocked until non-empty source proof |
| station_network_proxy | diagnostic_station_network | proxy_research | station | False | True | False | False | False | False | DIAGNOSTIC_ONLY |  | research proxy only |
| hko_daily_climate_proxy | diagnostic_physics | proxy_research | hko_climate | False | True | False | False | False | False | DIAGNOSTIC_ONLY |  | research proxy only |
| igra_upper_air_proxy | diagnostic_physics | support_only | igra | False | True | False | False | False | True | DIAGNOSTIC_ONLY |  | support-only proxy |
| tc_best_track_proxy | diagnostic_regime_labels | support_only | tc | False | True | False | False | False | True | DIAGNOSTIC_ONLY |  | support-only proxy |
| arwf_live | live_nwp_anchor | live_shadow | arwf | False | False | True | False | True | False | LIVE_FIRST_SEEN_ONLY |  | live shadow after enough history |

## Schema Object Audit Notes

The object list below explains why each durable object exists and what would be a review failure for that object.

### `model_core.run_manifest`

Kind: `table`. Foundation responsibility: Audit row per CLI command with command name, code version, git commit, DB hash, status, timestamps, and notes. Review failure: later code writes production claims into this scaffold before the matching Jira phase supplies acceptance tests.

### `model_core.source_registry`

Kind: `table`. Foundation responsibility: Durable final-patch registry keyed by `source_code` with booleans for strict, proxy, shadow, blocked, live-only, and support-only use. Review failure: a source is added without boolean status fields, a deprecated strict-status field becomes required, or a blocked source can be selected by name alone.

### `model_core.cutoff_calendar`

Kind: `table`. Foundation responsibility: One H24N calendar row per target date with formal cutoff, operational freeze, partition, season, month, day of year, and snapshot id. Review failure: cutoff time, freeze time, partition, or snapshot id drift from the H24N final patch.

### `model_core.target_label`

Kind: `table`. Foundation responsibility: Development-visible target Tmax labels copied from the selected canonical fallback table with source hash metadata. Review failure: sealed labels or raw audit labels are loaded as development-visible labels.

### `model_features.h24n_snapshot`

Kind: `table`. Foundation responsibility: Snapshot availability ledger with target date, cutoff, partition, strict NWP flags, live-shadow flags, status, and absent-availability reason code. Review failure: later code writes production claims into this scaffold before the matching Jira phase supplies acceptance tests.

### `model_features.nwp_safe_row_ledger`

Kind: `table`. Foundation responsibility: GribStream forecast-row ledger that records source scope, safety flag, exclusion reason, dataset, run time, valid time, and raw object link. Review failure: ledger rows no longer record excluded rows or the exclusion reason, because that would hide timing and source-policy failures.

### `model_features.feature_matrix`

Kind: `table`. Foundation responsibility: Final physical feature-matrix table for later phases; Jira 001 creates the shell and migrates any legacy strict/proxy physical tables into it. Review failure: another physical feature-matrix relation becomes authoritative for strict or proxy rows.

### `model_features.snapshot_feature_matrix_strict`

Kind: `view`. Foundation responsibility: Compatibility view over `feature_matrix` constrained to strict schema version. Review failure: this compatibility name is recreated as a table rather than a view over the final matrix.

### `model_features.snapshot_feature_matrix_proxy`

Kind: `view`. Foundation responsibility: Compatibility view over `feature_matrix` constrained to proxy schema version. Review failure: this compatibility name is recreated as a table rather than a view over the final matrix.

### `model_features.v_nwp_forecast_wide_compat`

Kind: `view`. Foundation responsibility: Compatibility view that exposes the tactical forecast table through the final foundation surface. Review failure: later code writes production claims into this scaffold before the matching Jira phase supplies acceptance tests.

### `model_features.v_raw_response_object_compat`

Kind: `view`. Foundation responsibility: Compatibility view that maps raw response object hash and creation time to final-patch names. Review failure: later code writes production claims into this scaffold before the matching Jira phase supplies acceptance tests.

### `model_features.v_nwp_h24n_safe_rows`

Kind: `view`. Foundation responsibility: Read-only safe-row view applying source-scope, cutoff, buffer, and blocked-source filters. Review failure: the view drops the raw response join, full-run source scope, cutoff id, six-hour buffer, or blocked-source exclusion.

### `model_validation.leakage_audit_event`

Kind: `table`. Foundation responsibility: Fail-closed event log for leakage violations and contract errors. Review failure: later code writes production claims into this scaffold before the matching Jira phase supplies acceptance tests.

### `model_validation.scoreboard`

Kind: `table`. Foundation responsibility: Future validation scoreboard scaffold with candidate, metric, partition, target-date span, run mode, and run manifest link. Review failure: later code writes production claims into this scaffold before the matching Jira phase supplies acceptance tests.

### `model_validation.negative_control_result`

Kind: `table`. Foundation responsibility: Future negative-control scaffold with control name, candidate, MAE, expected behavior, status, details, and run link. Review failure: later code writes production claims into this scaffold before the matching Jira phase supplies acceptance tests.

### `model_audit.schema_contract_audit`

Kind: `table`. Foundation responsibility: Schema audit shell for later evidence entries. Review failure: later code writes production claims into this scaffold before the matching Jira phase supplies acceptance tests.

### `model_live.prediction`

Kind: `table`. Foundation responsibility: Live prediction scaffold with target date, cutoff id, issued time, model candidate, run mode, prediction value, intervals, status, and uniqueness rule. Review failure: later code writes production claims into this scaffold before the matching Jira phase supplies acceptance tests.

### `model_live.live_prediction_component`

Kind: `table`. Foundation responsibility: Live component scaffold linked to a prediction with source code, component name, value, units, and contribution weight. Review failure: later code writes production claims into this scaffold before the matching Jira phase supplies acceptance tests.

### `model_eval.system_prediction_component`

Kind: `table`. Foundation responsibility: Evaluation component scaffold that records the later system prediction decomposition by candidate, target, source, and component. Review failure: later code writes production claims into this scaffold before the matching Jira phase supplies acceptance tests.


## Source-by-Source Policy Interpretation

This section turns the registry CSV into reviewer-facing guidance. Each entry states how the source may be used by this foundation and which gate must be satisfied before a later phase changes that use.

### `hko_target_labels`

`hko_target_labels` is classified as `strict_core` in family `target` with feature prefix `target`. Allowed use in Jira 001: strict v1. Availability grade is `EXACT_VINTAGE` and the time policy is `finalized daily labels may enter only as T-2-or-older target memory`. The configured target span is 2000-01-02 to 2026-06-21; required acquisition scope is `no GribStream scope requirement`. Promotion gate: `always included as labels and lagged target memory only`. Blocker state: no active blocker recorded. Reviewer meaning: Finalized target-day value is never a same-day feature. This row should be treated as policy, not merely description, because later feature code must obey the booleans before a source enters any training surface.

### `hko_official_forecasts`

`hko_official_forecasts` is classified as `strict_core` in family `official` with feature prefix `official`. Allowed use in Jira 001: strict v1. Availability grade is `EXACT_VINTAGE` and the time policy is `issue_at_utc must be <= operational freeze for the target date`. The configured target span is 2000-01-02 to 2026-06-21; required acquisition scope is `no GribStream scope requirement`. Promotion gate: `always included when eligible row exists`. Blocker state: no active blocker recorded. Reviewer meaning: Primary official HKO forecast anchor. This row should be treated as policy, not merely description, because later feature code must obey the booleans before a source enters any training surface.

### `calendar`

`calendar` is classified as `strict_core` in family `deterministic` with feature prefix `calendar`. Allowed use in Jira 001: strict v1. Availability grade is `EXACT_VINTAGE` and the time policy is `deterministic target-date metadata known before cutoff`. The configured target span is 2000-01-02 to 2026-06-21; required acquisition scope is `no GribStream scope requirement`. Promotion gate: `always included`. Blocker state: no active blocker recorded. Reviewer meaning: Only whitelisted cyclical/month/season/year-index fields may enter models. This row should be treated as policy, not merely description, because later feature code must obey the booleans before a source enters any training surface.

### `gfs`

`gfs` is classified as `strict_core` in family `gribstream` with feature prefix `gfs`. Allowed use in Jira 001: strict v1. Availability grade is `CONSERVATIVE_SCHEDULE` and the time policy is `run_time_utc + 6 hours <= formal H24N cutoff`. The configured target span is 2021-03-23 to 2026-06-23; required acquisition scope is `full_tactical_backfill_ok_tmax`. Promotion gate: `core strict expert E4`. Blocker state: no active blocker recorded. Reviewer meaning: Strict NWP source after full-run scope and H24N safe-row filters. This row should be treated as policy, not merely description, because later feature code must obey the booleans before a source enters any training surface.

### `gefsatmosmean`

`gefsatmosmean` is classified as `strict_core` in family `gribstream` with feature prefix `gefsmean`. Allowed use in Jira 001: strict v1. Availability grade is `CONSERVATIVE_SCHEDULE` and the time policy is `run_time_utc + 6 hours <= formal H24N cutoff`. The configured target span is 2021-03-23 to 2026-06-23; required acquisition scope is `full_tactical_backfill_ok_tmax`. Promotion gate: `core strict expert E5 context`. Blocker state: no active blocker recorded. Reviewer meaning: GEFS mean context source after audited safe-row filter. This row should be treated as policy, not merely description, because later feature code must obey the booleans before a source enters any training surface.

### `gefsatmos`

`gefsatmos` is classified as `strict_core` in family `gribstream` with feature prefix `gefsens`. Allowed use in Jira 001: strict v1. Availability grade is `CONSERVATIVE_SCHEDULE` and the time policy is `run_time_utc + 6 hours <= formal H24N cutoff`. The configured target span is 2021-03-23 to 2026-06-23; required acquisition scope is `full_tactical_backfill_ok_tmax`. Promotion gate: `core strict expert E5 ensemble`. Blocker state: no active blocker recorded. Reviewer meaning: HKO-center GEFS ensemble source after audited safe-row filter. This row should be treated as policy, not merely description, because later feature code must obey the booleans before a source enters any training surface.

### `ifsoper`

`ifsoper` is classified as `shadow_challenger` in family `gribstream` with feature prefix `ifsoper`. Allowed use in Jira 001: shadow challenger. Availability grade is `CONSERVATIVE_SCHEDULE` and the time policy is `run_time_utc + 6 hours <= formal H24N cutoff`. The configured target span is 2021-03-23 to 2026-06-23; required acquisition scope is `full_tactical_backfill_ok_tmax`. Promotion gate: `may enter after sealed protocol`. Blocker state: no active blocker recorded. Reviewer meaning: Shadow challenger, excluded from strict v1 features. This row should be treated as policy, not merely description, because later feature code must obey the booleans before a source enters any training surface.

### `ifsenfo`

`ifsenfo` is classified as `shadow_challenger` in family `gribstream` with feature prefix `ifsens`. Allowed use in Jira 001: shadow challenger. Availability grade is `CONSERVATIVE_SCHEDULE` and the time policy is `run_time_utc + 6 hours <= formal H24N cutoff`. The configured target span is 2021-03-23 to 2026-06-23; required acquisition scope is `full_tactical_backfill_ok_tmax`. Promotion gate: `may enter after sealed protocol; member-0 tracked`. Blocker state: no active blocker recorded. Reviewer meaning: IFS ensemble shadow challenger. This row should be treated as policy, not merely description, because later feature code must obey the booleans before a source enters any training surface.

### `cwawrf15`

`cwawrf15` is classified as `live_shadow` in family `gribstream` with feature prefix `cwawrf15`. Allowed use in Jira 001: shadow challenger, live-only tracking. Availability grade is `LIVE_FIRST_SEEN_ONLY` and the time policy is `live first-seen collection only`. The configured target span is 2026-06-23 to open-end; required acquisition scope is `full_tactical_backfill_ok_tmax`. Promotion gate: `prospective only until two seasonal cycles`. Blocker state: no active blocker recorded. Reviewer meaning: CWA WRF live shadow source; absence is warning-level for Jira 001. This row should be treated as policy, not merely description, because later feature code must obey the booleans before a source enters any training surface.

### `aifsoper`

`aifsoper` is classified as `shadow_challenger` in family `gribstream` with feature prefix `aifsoper`. Allowed use in Jira 001: shadow challenger. Availability grade is `CONSERVATIVE_SCHEDULE` and the time policy is `run_time_utc + 6 hours <= formal H24N cutoff`. The configured target span is open-start to open-end; required acquisition scope is `full_tactical_backfill_ok_tmax`. Promotion gate: `may enter after sealed protocol`. Blocker state: no active blocker recorded. Reviewer meaning: AI deterministic shadow source. This row should be treated as policy, not merely description, because later feature code must obey the booleans before a source enters any training surface.

### `aifsenfo`

`aifsenfo` is classified as `shadow_challenger` in family `gribstream` with feature prefix `aifsens`. Allowed use in Jira 001: shadow challenger. Availability grade is `CONSERVATIVE_SCHEDULE` and the time policy is `run_time_utc + 6 hours <= formal H24N cutoff`. The configured target span is open-start to open-end; required acquisition scope is `full_tactical_backfill_ok_tmax`. Promotion gate: `may enter after sealed protocol`. Blocker state: no active blocker recorded. Reviewer meaning: AI ensemble shadow source. This row should be treated as policy, not merely description, because later feature code must obey the booleans before a source enters any training surface.

### `aigfssfc`

`aigfssfc` is classified as `shadow_challenger` in family `gribstream` with feature prefix `aigfssfc`. Allowed use in Jira 001: shadow challenger. Availability grade is `CONSERVATIVE_SCHEDULE` and the time policy is `run_time_utc + 6 hours <= formal H24N cutoff`. The configured target span is open-start to open-end; required acquisition scope is `full_tactical_backfill_ok_tmax`. Promotion gate: `may enter after sealed protocol`. Blocker state: no active blocker recorded. Reviewer meaning: AI/GFS surface shadow source over short range. This row should be treated as policy, not merely description, because later feature code must obey the booleans before a source enters any training surface.

### `aigfspres`

`aigfspres` is classified as `support_only` in family `gribstream` with feature prefix `aigfspres`. Allowed use in Jira 001: support-only evidence. Availability grade is `CONSERVATIVE_SCHEDULE` and the time policy is `upper-air support only, not a daily Tmax source`. The configured target span is open-start to open-end; required acquisition scope is `full_tactical_backfill_ok_tmax`. Promotion gate: `support-only source`. Blocker state: no active blocker recorded. Reviewer meaning: Excluded from daily Tmax strict/proxy/shadow feature matrices. This row should be treated as policy, not merely description, because later feature code must obey the booleans before a source enters any training surface.

### `aigefssfc`

`aigefssfc` is classified as `blocked` in family `gribstream` with feature prefix `aigefssfc`. Allowed use in Jira 001: blocked daily Tmax input. Availability grade is `BLOCKED` and the time policy is `blocked because daily Tmax coverage is too sparse`. The configured target span is open-start to open-end; required acquisition scope is `full_tactical_backfill_ok_tmax`. Promotion gate: `blocked until provider/selector fix`. Blocker state: Poor usable daily Tmax candidate coverage.. Reviewer meaning: Rows can be leakage-safe but not usable enough for daily Tmax. This row should be treated as policy, not merely description, because later feature code must obey the booleans before a source enters any training surface.

### `graphcast`

`graphcast` is classified as `shadow_challenger` in family `gribstream` with feature prefix `graphcast`. Allowed use in Jira 001: shadow challenger. Availability grade is `MODEL_RUN_TIME_PROXY_ONLY` and the time policy is `model run time proxy only`. The configured target span is open-start to open-end; required acquisition scope is `full_tactical_backfill_ok_tmax`. Promotion gate: `may enter after sealed protocol`. Blocker state: no active blocker recorded. Reviewer meaning: Shadow source; availability proof remains weaker than exact first-seen. This row should be treated as policy, not merely description, because later feature code must obey the booleans before a source enters any training surface.

### `fourcastnetgfs`

`fourcastnetgfs` is classified as `shadow_challenger` in family `gribstream` with feature prefix `fourcastnet`. Allowed use in Jira 001: shadow challenger. Availability grade is `MODEL_RUN_TIME_PROXY_ONLY` and the time policy is `model run time proxy only; archive ends before current period`. The configured target span is open-start to open-end; required acquisition scope is `full_tactical_backfill_ok_tmax`. Promotion gate: `may enter after sealed protocol`. Blocker state: no active blocker recorded. Reviewer meaning: Shadow source available through observed archive end only. This row should be treated as policy, not merely description, because later feature code must obey the booleans before a source enters any training surface.

### `nbmoc`

`nbmoc` is classified as `blocked` in family `gribstream` with feature prefix `nbmoc`. Allowed use in Jira 001: blocked daily Tmax input. Availability grade is `BLOCKED` and the time policy is `blocked empty/probe-only source`. The configured target span is open-start to open-end; required acquisition scope is `full_tactical_backfill_ok_tmax`. Promotion gate: `blocked until non-empty source proof`. Blocker state: No usable HKO-domain daily Tmax coverage.. Reviewer meaning: Probe-only source excluded from all feature matrices. This row should be treated as policy, not merely description, because later feature code must obey the booleans before a source enters any training surface.

### `station_network_proxy`

`station_network_proxy` is classified as `proxy_research` in family `diagnostic_station_network` with feature prefix `station`. Allowed use in Jira 001: proxy research. Availability grade is `DIAGNOSTIC_ONLY` and the time policy is `proxy research only pending operational-vintage repair`. The configured target span is open-start to open-end; required acquisition scope is `no GribStream scope requirement`. Promotion gate: `research proxy only`. Blocker state: no active blocker recorded. Reviewer meaning: Station wind direction remains forbidden until repaired. This row should be treated as policy, not merely description, because later feature code must obey the booleans before a source enters any training surface.

### `hko_daily_climate_proxy`

`hko_daily_climate_proxy` is classified as `proxy_research` in family `diagnostic_physics` with feature prefix `hko_climate`. Allowed use in Jira 001: proxy research. Availability grade is `DIAGNOSTIC_ONLY` and the time policy is `finalized daily climate is not live exact-vintage`. The configured target span is open-start to open-end; required acquisition scope is `no GribStream scope requirement`. Promotion gate: `research proxy only`. Blocker state: no active blocker recorded. Reviewer meaning: Never use finalized target-day daily climate as a strict live feature. This row should be treated as policy, not merely description, because later feature code must obey the booleans before a source enters any training surface.

### `igra_upper_air_proxy`

`igra_upper_air_proxy` is classified as `support_only` in family `diagnostic_physics` with feature prefix `igra`. Allowed use in Jira 001: proxy research, support-only evidence. Availability grade is `DIAGNOSTIC_ONLY` and the time policy is `support/proxy only pending sentinel and vintage repair`. The configured target span is open-start to open-end; required acquisition scope is `no GribStream scope requirement`. Promotion gate: `support-only proxy`. Blocker state: no active blocker recorded. Reviewer meaning: IGRA contains known sentinel/scale issues in current inventory. This row should be treated as policy, not merely description, because later feature code must obey the booleans before a source enters any training surface.

### `tc_best_track_proxy`

`tc_best_track_proxy` is classified as `support_only` in family `diagnostic_regime_labels` with feature prefix `tc`. Allowed use in Jira 001: proxy research, support-only evidence. Availability grade is `DIAGNOSTIC_ONLY` and the time policy is `retrospective best-track only`. The configured target span is open-start to open-end; required acquisition scope is `no GribStream scope requirement`. Promotion gate: `support-only proxy`. Blocker state: no active blocker recorded. Reviewer meaning: Retrospective TC best-track may not be used as live strict input. This row should be treated as policy, not merely description, because later feature code must obey the booleans before a source enters any training surface.

### `arwf_live`

`arwf_live` is classified as `live_shadow` in family `live_nwp_anchor` with feature prefix `arwf`. Allowed use in Jira 001: shadow challenger, live-only tracking. Availability grade is `LIVE_FIRST_SEEN_ONLY` and the time policy is `live first-seen collection only`. The configured target span is 2026-06-19 to open-end; required acquisition scope is `no GribStream scope requirement`. Promotion gate: `live shadow after enough history`. Blocker state: no active blocker recorded. Reviewer meaning: ARWF absence is warning-level for Jira 001 and emits absent-source markers. This row should be treated as policy, not merely description, because later feature code must obey the booleans before a source enters any training surface.


## Source Safety Rules and Non-GribStream Examples

The GribStream filter is only one part of the Jira. The same safety mindset applies to target labels, official forecasts, station diagnostics, climate proxies, upper-air data, tropical cyclone labels, ARWF, and live-shadow sources. The foundation records each family so future work can decide with evidence rather than table-name guesswork.

### Target-label example

`label_core.hko_daily_tmax` is development-visible through 2023-12-31 and was selected by the source-contract report as the fallback target label table. The implementation loads those rows into `model_core.target_label` and marks them as development-visible. The final verified DB count was 8,765 target labels from 2000-01-02 through 2023-12-31. A later model may use those rows for lagged target memory at T-2 or older, but it must not use the target date value as a same-day feature.

### Sealed-label example

`sealed_confirmation.hko_daily_tmax` exists and contains post-2023 labels, but the foundation does not treat those rows as development-visible. The reason is not lack of data; it is the sealed evaluation boundary. A later sealed-validation phase can query those labels under its own protocol, but Jira 001 must not leak them into tuning or feature construction.

### Official forecast example

`public.hko_historical_forecasts_2000_2026` exists and passed source-contract checks with 324,179 total rows and 115,795 usable local min/max rows. The foundation records the archive and the strict source-registry row `hko_official_forecasts`, but the snapshot availability report still shows zero official availability counts because exact issue-time feature extraction was not implemented in this slice. That is a known foundation boundary rather than a GribStream issue.

### Station-network example

`station_network_proxy` is proxy research, not strict input. Station-derived signals can be valuable for wind, marine influence, and local gradients, yet the inventory warned that wind-direction and operational-vintage repair were not finalized. The registry therefore allows proxy use but blocks strict v1 use until a later task supplies exact-vintage handling.

### Daily climate example

`hko_daily_climate_proxy` is proxy research. Finalized daily climate data can contain target-day information and should not be used as a strict live feature for the same target date. The foundation keeps the row visible so the risk is explicit instead of buried in a notebook or old experiment.

### Upper-air example

`igra_upper_air_proxy` is support-only with proxy permission. The PostgreSQL inventory flagged sentinel and scale issues, so the row is useful for research notes and future repair work but not strict daily Tmax model input in Jira 001.

### Tropical-cyclone example

`tc_best_track_proxy` is support-only because best-track data is retrospective. It can explain regimes after the fact and support error analysis, but it cannot enter a live strict feature matrix unless a future task replaces it with an operationally available cyclone signal.

### ARWF example

`arwf_live` is live-shadow. The absence of enough ARWF live first-seen history is warning-level for this Jira because the foundation is allowed to create the schema and report the live-shadow state without training from it. Later work needs live collection depth before using it as a challenger input.

### CWA WRF example

`cwawrf15` is also live-shadow/prospective. Rows can be counted in the tactical audit, but short history prevents strict v1 use. The registry stores that status so future code must opt in through a promotion gate instead of silently treating the dataset like GFS.


## Error Handling/Edge Cases

A GribStream row can be present and still excluded. It is excluded when it belongs to a non-H24N cutoff, fails the six-hour run-time buffer, or is a blocked daily Tmax dataset. The ledger keeps both safe and excluded rows, which lets the audit report show why counts differ between scoped and safe totals.

A source can be useful and still not strict. Station-network data, HKO daily climate, IGRA upper-air, and TC best-track data are retained as proxy or support evidence because the PostgreSQL inventory showed value but not final exact-vintage safety. The source registry prevents those sources from entering strict features until later work supplies the missing operational-vintage proof.

Labels have their own boundary. `model_core.target_label` uses development-visible labels from the canonical pre-2024 source. Sealed confirmation labels are acknowledged by the source inventory, but they are not copied as development-visible labels by this foundation.

## Security, Privacy, and Credentials

The data contract docs and reports avoid storing database secrets. Source rows and schema reports contain table names, row counts, and policy decisions, not credential material. The implementation hashes source data or connection identity where a stable evidence key is useful, and it redacts usernames and passwords in human-readable reports.

## Performance Notes

The safe-row ledger is the largest persistence operation in this Jira because it records row-level safety decisions for tactical NWP data. The verified run saw 1,964,157 scoped rows and 1,858,133 safe rows. Because the ledger is refreshed by inserting from a join, DB indexes on `forecast_wide.source_response_object_id`, `forecast_wide.cutoff_id`, `forecast_wide.dataset_code`, `forecast_wide.target_date_hkt`, and `raw_response_object.response_object_id` are the likely first place to inspect if runtime grows.

## Configuration

The YAML contract fixes `cutoff_id: H24N`, `target_date_start: 2000-01-02`, `target_date_end: 2026-06-21`, `formal_cutoff_hkt: "T-1 15:00"`, `operational_freeze_hkt: "T-1 14:45"`, source scope `full_tactical_backfill_ok_tmax`, publication buffer `6`, and final schema versions. The schema JSON repeats the schema-version contract for downstream tools that should not import Python.

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


## Known Limitations and Follow-Up Work

The source contract proves foundation safety, not final model utility. Official HKO forecasts pass archive table checks but snapshot availability currently reports zero official availability rows because final feature extraction from exact issue-time rows was not part of Jira 001. ARWF and CWA WRF remain live-shadow/prospective sources. Shadow GribStream models are recorded for later challenger analysis but are not promoted into strict v1 features by this Jira.

The source registry can be extended only by a later contract update that states source code, role, feature prefix, status booleans, time policy, required scope, blocker reason when relevant, promotion gate, and unit semantics. Adding a source by inserting data into PostgreSQL is not enough.

## Reviewer Checklist

- Check that source registry rows match the final-patch policy before using feature data.
- Check that any GribStream query joins raw response objects and filters the full tactical backfill scope.
- Check that run-time, not valid-time, is the availability basis.
- Check that blocked daily Tmax datasets stay excluded from feature rows.
- Check that proxy and support sources do not enter strict training.
- Check that sealed labels remain outside development-visible labels.
- Check that compatibility views are views and `feature_matrix` is the physical table.
- Check that schema conflict report is still clean after rerunning migrations.

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

## Report Evidence Catalog


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
