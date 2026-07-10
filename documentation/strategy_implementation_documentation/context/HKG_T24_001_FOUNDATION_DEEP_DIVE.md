# HKG-T24-001 Foundation Implementation Deep Dive

## Executive Summary

HKG-T24-001 created the foundation layer for the HKG T+24 / H24N data contract. The work added a dedicated Python package, a DB-backed CLI, deterministic H24N time rules, final-patch source-registry population, idempotent schema migration, GribStream safe-row construction, H24N snapshot generation, report writing, and focused tests. The implementation is intentionally a data-contract and feature-store base slice: it prepares the surfaces later modeling work will consume, but it does not train a model, promote a router, tune sealed validation, or publish a live forecast candidate.

The most important outcome is that a later engineer now has one narrow command surface for foundation setup and verification: `python -m hkg_t24.cli phase0-preflight`, `python -m hkg_t24.cli build-source-registry`, and `python -m hkg_t24.cli build-h24n-snapshots`. Each command uses the same DSN precedence rule, applies migrations before work that depends on managed tables, writes `model_core.run_manifest`, and emits the required report set under `reports/`.

## Reader Orientation

Read this document when you need to understand the package shape, command flow, Python modules, and how the implementation maps to Jira acceptance. Read `HKG_T24_001_SOURCE_SAFETY_AUDIT.md` for the data-source and schema contract audit. Read `HKG_T24_001_VERIFICATION_GUIDE.md` for command outputs, test coverage, DB smoke evidence, report interpretation, rerun procedure, and reviewer checklist.

The implementation lives in `code/src/hkg_t24/`. Tests live in `code/tests/hkg_t24/`. SQL and schema contract assets live under `sql/hkg_t24/` and `schemas/hkg_t24/`. The three prior evidence documents from the screenshot remain in this same context folder and are treated as inputs, not outputs from the rewritten documentation pass.

## Scope Boundaries

In scope for Jira 001: foundation package structure, CLI commands, DB connection handling, schemas, tables, compatibility views, source-registry rows, source-contract preflight, cutoff calendar, target-label loading, H24N snapshot rows, GribStream safe-row ledger, required reports, and tests proving the core contracts. Out of scope: candidate training, LightGBM feature selection, router logic, OOF blending, sealed leaderboard promotion, live prediction service execution, and negative-control scoring runs. Those surfaces receive only the scaffolds specifically required by the final patch.

The live PostgreSQL database is the source of truth for source-contract acceptance. Static files document desired behavior, but the CLI commands were checked against `postgresql://***:***@127.0.0.1:5432/hkg_tmax_research` and report redacted DB identity plus row counts. When a DSN is absent, the code fails closed rather than silently substituting a fake or local in-memory source.

## Source-of-Truth Inputs

The implementation followed the binding order stated in the Jira plan: final consistency patch, final clarifications, completion spec, blueprint, and Jira packet. The source audits in the context directory were also read and used because the screenshot named them as critical inputs. Their role is summarized below.


| evidence document | bytes | lines | sha256_prefix | reason it mattered |
| --- | --- | --- | --- | --- |
| documentation/strategy_implementation_documentation/context/GRIBSTREAM_FETCHED_DATA_INVENTORY_20260626.md | 21565 | 499 | 422dfdaf095b0e0f | Inventory proving tactical forecast tables, raw response objects, full-run source scope, mixed smoke rows, model coverage, and GribStream dataset roles. |
| documentation/strategy_implementation_documentation/context/GRIBSTREAM_LEAKAGE_SAFE_DB_RETRIEVAL_LEDGER_20260626.md | 28191 | 937 | 9bfa724d3e32c697 | Ledger defining the required H24N safe predicate, source-scope filter, run-time based availability, six-hour buffer, and blocked daily Tmax sources. |
| documentation/strategy_implementation_documentation/context/POSTGRES_STRATEGY_DATASET_INVENTORY_20260626.md | 172236 | 2530 | 0b9a6fb36ea9a8fd | PostgreSQL strategy inventory covering target labels, sealed labels, official forecasts, tactical NWP, diagnostic sources, and why table existence does not equal model eligibility. |

## Requirements-to-Implementation Traceability

The matrix below maps the acceptance surface to the implemented evidence. Items marked intentionally out of scope are not omissions; they are later-phase modeling concerns that Jira 001 only scaffolds where required.


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

The following files are the implementation and artifact surface covered by this documentation. Size, line count, and hash prefix were captured from the current workspace at generation time so a reviewer can detect drift.


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

The foundation is organized as a command-driven pipeline. `cli.py` owns argument parsing and manifest lifecycle. `db.connection` resolves the database URL. `db.migrations` applies schemas, checks conflicts, and prepares compatibility surfaces. `features.source_contracts` checks external data availability. `audit.source_registry` writes source rows. `features.gribstream_safe_rows` materializes the safe-row ledger. `features.snapshot_builder` writes cutoff calendar rows, target labels, and snapshot availability rows. `artifacts.reports` keeps report paths deterministic.

```mermaid
flowchart TD
  A["CLI command"] --> B["Resolve DSN"]
  B --> C["Open PostgreSQL connection"]
  C --> D["Apply foundation migrations"]
  D --> E["Create run manifest"]
  E --> F["Run command operation"]
  F --> G["Write reports"]
  G --> H["Finish manifest"]
  F --> I["Fail-closed leakage or contract event"]
  I --> G
```

The command order matters. `phase0-preflight` proves the environment and source contracts. `build-source-registry` applies the same migrations and then upserts the final registry rows. `build-h24n-snapshots` repeats the preflight checks, refreshes the safe-row ledger, populates calendar and label tables, builds snapshots, writes the GribStream audit, writes leakage status, and writes final coverage. This repetition is deliberate because each command can be run on a fresh database and still produce a coherent report trail.

## File-by-File Deep Dive


### `pyproject.toml`

This artifact records a durable contract surface for Jira 001. It currently has 72 lines, 1550 bytes, and SHA-256 prefix `98b1067e135c288a`. Review it with the Python source because the code remains the executable source while the artifact gives operators and reviewers a stable file to inspect.


### `docs/PROJECT_STRUCTURE_AND_CODE_MAP.md`

This artifact records a durable contract surface for Jira 001. It currently has 455 lines, 20518 bytes, and SHA-256 prefix `f4274317e7054ab0`. Review it with the Python source because the code remains the executable source while the artifact gives operators and reviewers a stable file to inspect.


### `code/src/hkg_t24/artifacts/reports.py`

Report path handling lives here. It gives the CLI deterministic paths for root reports, context docs, SQL assets, schema assets, and config assets. The current file has 71 lines, 2710 bytes, and SHA-256 prefix `0540ce13aed3be87`. Its review focus is contract drift: if any constant, SQL object, command branch, or report name changes, update the tests and reports in the same commit.


### `code/src/hkg_t24/audit/leakage_events.py`

Leakage event helpers insert fail-closed events and count error events for the leakage audit report. The current file has 37 lines, 1025 bytes, and SHA-256 prefix `ac2b4d44b31d9910`. Its review focus is contract drift: if any constant, SQL object, command branch, or report name changes, update the tests and reports in the same commit.


### `code/src/hkg_t24/audit/schema_contracts.py`

This module contains low-level DB discovery helpers for table existence, column discovery, row counts, and canonical source fallback selection. The current file has 160 lines, 5144 bytes, and SHA-256 prefix `037cb403051ab500`. Its review focus is contract drift: if any constant, SQL object, command branch, or report name changes, update the tests and reports in the same commit.


### `code/src/hkg_t24/audit/source_registry.py`

Source-registry population lives here. It validates the final row set, upserts rows into PostgreSQL, and writes `reports/source_registry.csv`. The current file has 95 lines, 3733 bytes, and SHA-256 prefix `718c3fa59d6c9e40`. Its review focus is contract drift: if any constant, SQL object, command branch, or report name changes, update the tests and reports in the same commit.


### `code/src/hkg_t24/cli.py`

Command orchestration lives here. The parser exposes the three required commands, `_run_with_manifest` wraps every DB-backed operation in a manifest row, and the failure path writes reports even when the DSN contract blocks the command before a connection is opened. The current file has 305 lines, 10694 bytes, and SHA-256 prefix `46564496d935ae5c`. Its review focus is contract drift: if any constant, SQL object, command branch, or report name changes, update the tests and reports in the same commit.


### `code/src/hkg_t24/constants.py`

The contract constants live here: cutoff identifiers, date range, schema versions, source registry rows, NWP dataset allow/block sets, feature prefixes, calendar feature names, target-memory names, and the exact DSN warning/error text. The current file has 656 lines, 16566 bytes, and SHA-256 prefix `b9a2bb7ab0fdb300`. Its review focus is contract drift: if any constant, SQL object, command branch, or report name changes, update the tests and reports in the same commit.


### `code/src/hkg_t24/db/connection.py`

This module isolates DSN selection and PostgreSQL connection creation. It keeps the precedence rule testable without opening a socket and redacts database URLs for reports. The current file has 69 lines, 2000 bytes, and SHA-256 prefix `37c0407f71bd7e74`. Its review focus is contract drift: if any constant, SQL object, command branch, or report name changes, update the tests and reports in the same commit.


### `code/src/hkg_t24/db/ddl.py`

The foundation DDL lives here as executable SQL strings plus expected-column metadata used by the migration conflict guard. It is the code-side source for tables and views under `model_*` schemas. The current file has 370 lines, 15406 bytes, and SHA-256 prefix `1605676bc84165b2`. Its review focus is contract drift: if any constant, SQL object, command branch, or report name changes, update the tests and reports in the same commit.


### `code/src/hkg_t24/db/migrations.py`

Migration code applies the schema, checks for incompatible existing columns, migrates old feature-matrix physical relations, writes migration reports, and creates or finishes run-manifest rows. The current file has 276 lines, 10053 bytes, and SHA-256 prefix `c6f8e0380e7aa5f4`. Its review focus is contract drift: if any constant, SQL object, command branch, or report name changes, update the tests and reports in the same commit.


### `code/src/hkg_t24/features/gribstream_safe_rows.py`

GribStream safe-row construction lives here. It populates `nwp_safe_row_ledger` using the source-scope join, cutoff id, six-hour buffer, and blocked-source exclusions, then writes GribStream and leakage reports. The current file has 137 lines, 5430 bytes, and SHA-256 prefix `c79835d9d62a5509`. Its review focus is contract drift: if any constant, SQL object, command branch, or report name changes, update the tests and reports in the same commit.


### `code/src/hkg_t24/features/snapshot_builder.py`

Snapshot construction lives here. It builds target-memory features, populates cutoff calendar rows, copies development-visible labels, writes H24N snapshot availability rows, and emits snapshot plus live-shadow reports. The current file has 338 lines, 13696 bytes, and SHA-256 prefix `e43410a5ace7ccc6`. Its review focus is contract drift: if any constant, SQL object, command branch, or report name changes, update the tests and reports in the same commit.


### `code/src/hkg_t24/features/source_contracts.py`

Source preflight checks live here. It verifies LightGBM, target label fallbacks, official forecast availability, tactical NWP tables, raw response objects, and the full-run source scope. The current file has 267 lines, 9386 bytes, and SHA-256 prefix `a4856865676e32ce`. Its review focus is contract drift: if any constant, SQL object, command branch, or report name changes, update the tests and reports in the same commit.


### `code/src/hkg_t24/timeutils.py`

The H24N time policy lives here. It converts target dates into formal cutoffs, operational freezes, partitions, seasons, and stable snapshot ids under the Asia/Hong_Kong timezone. The current file has 105 lines, 3369 bytes, and SHA-256 prefix `19309f785db636d4`. Its review focus is contract drift: if any constant, SQL object, command branch, or report name changes, update the tests and reports in the same commit.


### `code/src/hkg_t24/utils/hashing.py`

Hash helpers create SHA-256 values for text, JSON payloads, and files. The source hash columns use this family of deterministic evidence identifiers. The current file has 26 lines, 657 bytes, and SHA-256 prefix `60344bab1ffc8878`. Its review focus is contract drift: if any constant, SQL object, command branch, or report name changes, update the tests and reports in the same commit.


### `code/src/hkg_t24/utils/sql.py`

SQL formatting helpers validate identifiers, create qualified names, and render small CSV rows without depending on ad hoc string concatenation at call sites. The current file has 31 lines, 919 bytes, and SHA-256 prefix `d36f3cea90ce3aea`. Its review focus is contract drift: if any constant, SQL object, command branch, or report name changes, update the tests and reports in the same commit.


### `code/tests/hkg_t24/test_database_url_priority.py`

Tests DSN priority, fallback use, exact missing-DSN error text, CLI fail-closed behavior, and report creation when no DSN is configured. The file currently has 58 lines, 1764 bytes, and SHA-256 prefix `7d4ed3a86f0cbe86`. Its assertions are narrow by design: they guard the exact foundation rules that are cheap to break during later modeling work.


### `code/tests/hkg_t24/test_h24n_contract_policy.py`

Tests cutoff and freeze clocks, partition naming, source-registry final-patch compatibility, feature prefix mapping, schema version constants, and forbidden finalized lag 1 target-memory names. The file currently has 62 lines, 2379 bytes, and SHA-256 prefix `fd73f2d6ba172180`. Its assertions are narrow by design: they guard the exact foundation rules that are cheap to break during later modeling work.


### `code/tests/hkg_t24/test_real_db_contracts.py`

Runs only when at least one DSN variable exists; then it checks source table discovery, migrations, views, source registry population, and snapshot/safe-row DB surfaces. The file currently has 17 lines, 594 bytes, and SHA-256 prefix `c4ecae3c9de4ae02`. Its assertions are narrow by design: they guard the exact foundation rules that are cheap to break during later modeling work.


### `code/tests/hkg_t24/test_schema_sql_contract.py`

Asserts SQL contract strings contain the final physical feature matrix, compatibility views, safe-row filters, raw-response compatibility view, live scaffolds, and validation scaffolds. The file currently has 45 lines, 2338 bytes, and SHA-256 prefix `f059794caa4f7682`. Its assertions are narrow by design: they guard the exact foundation rules that are cheap to break during later modeling work.


### `code/tests/hkg_t24/test_snapshot_builder_synthetic.py`

Exercises synthetic target-memory construction on 120 labels and proves the lag 2 policy creates expected counts without lag 1 finalized feature names. The file currently has 29 lines, 1206 bytes, and SHA-256 prefix `3875a0c18f154538`. Its assertions are narrow by design: they guard the exact foundation rules that are cheap to break during later modeling work.


### `config/hkg_t24/hkg_t24_001_foundation.yaml`

This artifact records a durable contract surface for Jira 001. It currently has 12 lines, 427 bytes, and SHA-256 prefix `29e0e79f012f5774`. Review it with the Python source because the code remains the executable source while the artifact gives operators and reviewers a stable file to inspect.


### `schemas/hkg_t24/hkg_t24_001_schema_versions.json`

This artifact records a durable contract surface for Jira 001. It currently has 17 lines, 542 bytes, and SHA-256 prefix `3081daa00d0cfcfa`. Review it with the Python source because the code remains the executable source while the artifact gives operators and reviewers a stable file to inspect.


### `sql/hkg_t24/hkg_t24_001_foundation_schema.sql`

This artifact records a durable contract surface for Jira 001. It currently has 25 lines, 934 bytes, and SHA-256 prefix `e4785f05f6003b42`. Review it with the Python source because the code remains the executable source while the artifact gives operators and reviewers a stable file to inspect.


### `sql/hkg_t24/hkg_t24_001_gribstream_safe_rows.sql`

This artifact records a durable contract surface for Jira 001. It currently has 24 lines, 913 bytes, and SHA-256 prefix `b0d198e3579410b2`. Review it with the Python source because the code remains the executable source while the artifact gives operators and reviewers a stable file to inspect.


### `reports/phase0_preflight_report.md`

DB preflight, LightGBM import, registry constant validation, and warning-level source absence. The generated artifact currently has 20 lines, 366 bytes, and SHA-256 prefix `1567ba91d971a20c`. It is not imported by the package; it is evidence produced by the CLI and should be regenerated when the DB state changes.


### `reports/schema_conflict_report.md`

Managed-column type conflict result before DDL proceeds. The generated artifact currently has 10 lines, 163 bytes, and SHA-256 prefix `627e4579f591b749`. It is not imported by the package; it is evidence produced by the CLI and should be regenerated when the DB state changes.


### `reports/source_inventory_report.md`

Source table discovery and row-count evidence. The generated artifact currently has 26 lines, 1044 bytes, and SHA-256 prefix `6658114d4c02a82b`. It is not imported by the package; it is evidence produced by the CLI and should be regenerated when the DB state changes.


### `reports/source_registry.csv`

Machine-readable final source registry. The generated artifact currently has 24 lines, 6386 bytes, and SHA-256 prefix `80e69380f63397f4`. It is not imported by the package; it is evidence produced by the CLI and should be regenerated when the DB state changes.


### `reports/schema_contract_report.md`

Target-label fallback, official forecast table, tactical forecast table, raw response object, and full-run source checks. The generated artifact currently has 26 lines, 1219 bytes, and SHA-256 prefix `798f7b42ef399ef7`. It is not imported by the package; it is evidence produced by the CLI and should be regenerated when the DB state changes.


### `reports/schema_migration_source_registry.md`

Registry migration surface and final-patch primary key shape. The generated artifact currently has 10 lines, 298 bytes, and SHA-256 prefix `65aefc8672d0fb88`. It is not imported by the package; it is evidence produced by the CLI and should be regenerated when the DB state changes.


### `reports/schema_migration_feature_matrix.md`

Feature-matrix physical-table migration and compatibility-view result. The generated artifact currently has 15 lines, 433 bytes, and SHA-256 prefix `06db191a8f0bde8f`. It is not imported by the package; it is evidence produced by the CLI and should be regenerated when the DB state changes.


### `reports/gribstream_source_scope_audit.csv`

Dataset-level scoped/safe/excluded row counts from the safe-row ledger. The generated artifact currently has 15 lines, 453 bytes, and SHA-256 prefix `3dbdb2d34ab0cb7a`. It is not imported by the package; it is evidence produced by the CLI and should be regenerated when the DB state changes.


### `reports/gribstream_source_scope_audit.md`

Human-readable GribStream scope and buffer audit. The generated artifact currently has 30 lines, 1341 bytes, and SHA-256 prefix `a902cc80491068fb`. It is not imported by the package; it is evidence produced by the CLI and should be regenerated when the DB state changes.


### `reports/snapshot_coverage_report.csv`

Partition-level snapshot availability counts. The generated artifact currently has 6 lines, 232 bytes, and SHA-256 prefix `121896b747911119`. It is not imported by the package; it is evidence produced by the CLI and should be regenerated when the DB state changes.


### `reports/snapshot_coverage_report.md`

Human-readable H24N snapshot coverage summary. The generated artifact currently has 13 lines, 440 bytes, and SHA-256 prefix `eecc3b1a06c6dc12`. It is not imported by the package; it is evidence produced by the CLI and should be regenerated when the DB state changes.


### `reports/live_shadow_availability_report.csv`

ARWF and CWA WRF live-shadow availability export. The generated artifact currently has 4 lines, 155 bytes, and SHA-256 prefix `c7b817162e230889`. It is not imported by the package; it is evidence produced by the CLI and should be regenerated when the DB state changes.


### `reports/live_shadow_availability_report.md`

Live-shadow interpretation and warning-level source absence. The generated artifact currently has 14 lines, 344 bytes, and SHA-256 prefix `71b860c4d52097ab`. It is not imported by the package; it is evidence produced by the CLI and should be regenerated when the DB state changes.


### `reports/leakage_audit_report.md`

Leakage-event error count and scope boundary. The generated artifact currently has 14 lines, 244 bytes, and SHA-256 prefix `f5488a3e2c2beaf1`. It is not imported by the package; it is evidence produced by the CLI and should be regenerated when the DB state changes.


### `reports/jira_001_contract_coverage.md`

End-to-end Jira 001 coverage claim and required report presence. The generated artifact currently has 57 lines, 2170 bytes, and SHA-256 prefix `ffc8a82152573750`. It is not imported by the package; it is evidence produced by the CLI and should be regenerated when the DB state changes.


## Public Interfaces and Contracts

The public interface is the module invocation `python -m hkg_t24.cli`. The command parser accepts `phase0-preflight`, `build-source-registry`, and `build-h24n-snapshots`. The snapshot command also accepts `--start-date` and `--end-date`; without those flags it uses the final Jira range from `2000-01-02` through `2026-06-21`. Every DB-backed command writes a manifest row with command name, code version, git commit, redacted database identity hash, start time, finish time, status, and notes.

The environment contract is exact. `HKG_TMAX_DATABASE_URL` has priority over `HKG_TMAX_DB_DSN`. If both are present, the code sends the contract warning through the message sink. If neither is present, the command exits with the exact configured error and writes fail-closed reports without pretending that DB acceptance passed. The package also requires LightGBM import to pass during preflight because the Jira final patch removed fallback behavior to a different estimator family.

The data contract exposes `target_date_hkt` as the date key, `cutoff_id` as `H24N`, `formal_cutoff_utc` as T-1 15:00 HKT converted to UTC, `operational_freeze_utc` as T-1 14:45 HKT converted to UTC, and `snapshot_id` as `H24N:YYYY-MM-DD`. The source registry uses `source_code` as its primary key and no new dependency on a deprecated strict-status field.

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


The migration sequence first creates the required schemas: `model_core`, `model_features`, `model_oof`, `model_router`, `model_validation`, `model_live`, `model_audit`, and `model_eval`. It then checks managed columns against expected types before applying the foundation DDL. If a managed object already exists with an incompatible type, the command writes `reports/schema_conflict_report.md` and fails before silently changing semantics.

Legacy feature-matrix handling is intentionally conservative. The final physical relation is `model_features.feature_matrix`. If `snapshot_feature_matrix_strict` or `snapshot_feature_matrix_proxy` already exists as a physical table, compatible rows are copied into the final table using the corresponding schema version, the old table is dropped, and the name is recreated as a view. That keeps old readers working while preventing two physical feature-matrix truths from drifting apart.

## Error Handling/Edge Cases

The missing-DSN branch is handled before PostgreSQL import or connection creation. The schema-conflict branch records a report and exits before DDL can partially mutate a conflicting database. The migration code treats old feature-matrix relations differently based on relation kind, which prevents a view from being handled as a table. The safe-row builder logs excluded GribStream rows rather than deleting them from the ledger, so a reviewer can distinguish unsafe timing, blocked daily Tmax source, and non-H24N cutoff.

Target labels are loaded only from the selected canonical fallback discovered by source contracts. The implemented final state selected `label_core.hko_daily_tmax`, not sealed labels and not raw audit labels. That distinction matters because sealed 2024 and 2025 labels exist in PostgreSQL but are not development-visible for Jira 001 tuning.

## Security, Privacy, and Credentials

Database URLs are redacted in reports. `database_url_hash` records a short identity hash so the manifest can separate DB targets without leaking credentials. The CLI uses environment variables only; no DSN is written into config files. Generated reports show `postgresql://***:***@127.0.0.1:5432/hkg_tmax_research` rather than the password-bearing connection string.

The SQL helpers quote only identifiers that pass a strict identifier regex. The foundation commands use parametrized SQL where runtime values enter inserts and updates. DDL strings are fixed source-controlled assets, and dynamic legacy-table migration only permits the two known compatibility names.

## Performance Notes

The full snapshot command is DB-heavy because it scans tactical forecast rows, refreshes the safe-row ledger, and builds 9,668 target-date snapshots. A first run under a 120 second wrapper timed out, then the exact operation passed with a longer timeout and took roughly 143.7 seconds. Idempotent reruns should avoid surprise row duplication because key tables use primary keys or conflict updates.

The largest operation is GribStream safe-row refresh: it joins `nwp_tactical.forecast_wide` to `nwp_tactical.raw_response_object`, filters source scope, applies cutoff safety, and writes a ledger that recorded 1,964,157 scoped rows with 1,858,133 safe rows in the verified DB. Future optimization should focus on indexes around `source_response_object_id`, `target_date_hkt`, `cutoff_id`, `dataset_code`, and `run_time_utc` if the source tables grow.

## Configuration

`config/hkg_t24/hkg_t24_001_foundation.yaml` records the Jira, cutoff id, target-date range, formal cutoff, operational freeze, GribStream source scope, buffer hours, and schema versions. `schemas/hkg_t24/hkg_t24_001_schema_versions.json` records the final strict, proxy, and shadow schema versions plus the final physical feature-matrix relation and compatibility views. `pyproject.toml` adds the console entry point `hkg-t24 = "hkg_t24.cli:main"`, while direct module execution remains the tested command form.

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


The unit tests cover environment precedence, exact failure text, H24N clock values, snapshot naming, source-registry rows, feature-prefix mapping, schema version constants, safe-row SQL filters, raw-response compatibility, live/eval scaffolds, validation scaffolds, and synthetic lag 2 target-memory counts. The DB-backed test is skipped only when both DSN variables are absent; in the verified run the DSN was present and the DB-backed branch ran.

## Known Limitations and Follow-Up Work

Jira 001 does not fill the final feature matrix with trained candidate features. It creates the durable table, compatibility views, source registry, safe-row ledger, snapshots, and scaffolds that later modeling tasks will use. ARWF first-seen history is warning-level absent in this foundation run. CWA WRF is live-shadow and prospective rather than a strict feature. Official HKO forecast availability reported zero rows in the snapshot availability count even though the official forecast archive exists and passes table checks; later feature engineering must connect exact-vintage official rows into feature columns before training.

The final implementation does not claim that shadow sources are promotable. `ifsoper`, `ifsenfo`, `aifsoper`, `aifsenfo`, `aigfssfc`, `graphcast`, and `fourcastnetgfs` are recorded for audit and later challenger work. `nbmoc`, `aigfspres`, and `aigefssfc` are excluded from daily Tmax feature matrices by final source policy.

## Reviewer Checklist

- Confirm the three CLI commands still run with the intended DSN.
- Confirm `reports/jira_001_contract_coverage.md` remains `PASS` after rerun.
- Confirm `reports/schema_conflict_report.md` remains conflict-free before accepting migrations.
- Confirm `reports/source_registry.csv` still has 22 final-patch rows unless the Jira contract changed.
- Confirm `model_features.feature_matrix` is the only physical feature-matrix relation.
- Confirm `snapshot_feature_matrix_strict` and `snapshot_feature_matrix_proxy` are views.
- Confirm the safe-row SQL still joins raw response objects and filters `full_tactical_backfill_ok_tmax`.
- Confirm no finalized target-memory feature name uses lag 1.
- Confirm sealed labels are not loaded into development-visible target labels.
- Confirm any later model-training work adds its own tests rather than treating this foundation as model acceptance.

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

## Symbol Evidence Ledger

The symbol ledger below is generated from the Python AST and gives reviewers exact entry points without requiring a full source scan first.


| path | symbol | line | note |
| --- | --- | --- | --- |
| code/src/hkg_t24/artifacts/reports.py | class ReportPathSet | 14 |  |
| code/src/hkg_t24/artifacts/reports.py | class ReportWriter | 30 | Simple deterministic report writer. |
| code/src/hkg_t24/audit/leakage_events.py | def record_leakage_event | 9 |  |
| code/src/hkg_t24/audit/leakage_events.py | def leakage_error_count | 30 |  |
| code/src/hkg_t24/audit/schema_contracts.py | class TableRef | 10 |  |
| code/src/hkg_t24/audit/schema_contracts.py | class SourceCheck | 20 |  |
| code/src/hkg_t24/audit/schema_contracts.py | class DiscoveredTable | 28 |  |
| code/src/hkg_t24/audit/schema_contracts.py | def table_exists | 35 |  |
| code/src/hkg_t24/audit/schema_contracts.py | def table_columns | 42 |  |
| code/src/hkg_t24/audit/schema_contracts.py | def count_rows | 55 |  |
| code/src/hkg_t24/audit/schema_contracts.py | def choose_column | 64 |  |
| code/src/hkg_t24/audit/schema_contracts.py | def discover_table | 71 |  |
| code/src/hkg_t24/audit/source_registry.py | constant CSV_HEADERS | 12 | module constant |
| code/src/hkg_t24/audit/source_registry.py | def source_registry_rows | 35 |  |
| code/src/hkg_t24/audit/source_registry.py | def populate_source_registry | 39 |  |
| code/src/hkg_t24/audit/source_registry.py | def validate_source_registry_contract | 81 |  |
| code/src/hkg_t24/cli.py | constant REPO_ROOT | 37 | module constant |
| code/src/hkg_t24/cli.py | def _parse_date | 40 |  |
| code/src/hkg_t24/cli.py | def build_parser | 47 |  |
| code/src/hkg_t24/cli.py | def _connect_and_migrate | 59 |  |
| code/src/hkg_t24/cli.py | def _write_phase0_report | 71 |  |
| code/src/hkg_t24/cli.py | def _write_contract_coverage | 91 |  |
| code/src/hkg_t24/cli.py | def _run_with_manifest | 145 |  |
| code/src/hkg_t24/cli.py | def _phase0_operation | 221 |  |
| code/src/hkg_t24/cli.py | def _source_registry_operation | 240 |  |
| code/src/hkg_t24/cli.py | def _snapshot_operation_factory | 251 |  |
| code/src/hkg_t24/cli.py | def main | 280 |  |
| code/src/hkg_t24/constants.py | constant CUTOFF_ID | 8 | module constant |
| code/src/hkg_t24/constants.py | constant CUTOFF_RULE_VERSION | 9 | module constant |
| code/src/hkg_t24/constants.py | constant TARGET_DATE_COLUMN | 10 | module constant |
| code/src/hkg_t24/constants.py | constant SNAPSHOT_ID_PREFIX | 11 | module constant |
| code/src/hkg_t24/constants.py | constant START_TARGET_DATE | 12 | module constant |
| code/src/hkg_t24/constants.py | constant END_TARGET_DATE | 13 | module constant |
| code/src/hkg_t24/constants.py | constant STRICT_SCHEMA_VERSION | 15 | module constant |
| code/src/hkg_t24/constants.py | constant PROXY_SCHEMA_VERSION | 16 | module constant |
| code/src/hkg_t24/constants.py | constant SHADOW_SCHEMA_VERSION | 17 | module constant |
| code/src/hkg_t24/constants.py | constant CODE_VERSION | 18 | module constant |
| code/src/hkg_t24/constants.py | constant DATABASE_URL_ENV | 20 | module constant |
| code/src/hkg_t24/constants.py | constant DATABASE_DSN_ENV | 21 | module constant |
| code/src/hkg_t24/constants.py | constant MISSING_DSN_ERROR | 22 | module constant |
| code/src/hkg_t24/constants.py | constant DUAL_DSN_WARNING | 26 | module constant |
| code/src/hkg_t24/constants.py | constant LIGHTGBM_ERROR | 27 | module constant |
| code/src/hkg_t24/constants.py | constant MODEL_SCHEMAS | 32 | module constant |
| code/src/hkg_t24/constants.py | constant REPORT_NAMES | 43 | module constant |
| code/src/hkg_t24/constants.py | constant STRICT_NWP_DATASETS | 72 | module constant |
| code/src/hkg_t24/constants.py | constant SHADOW_NWP_DATASETS | 73 | module constant |
| code/src/hkg_t24/constants.py | constant BLOCKED_DAILY_TMAX_DATASETS | 83 | module constant |
| code/src/hkg_t24/constants.py | constant DATASET_FEATURE_PREFIX | 85 | module constant |
| code/src/hkg_t24/constants.py | constant GRIBSTREAM_EXPERT_ID | 102 | module constant |
| code/src/hkg_t24/constants.py | constant CALENDAR_MODEL_FEATURE_WHITELIST | 116 | module constant |
| code/src/hkg_t24/constants.py | constant TARGET_MEMORY_FEATURE_WHITELIST | 128 | module constant |
| code/src/hkg_t24/constants.py | constant FORBIDDEN_FINALIZED_TARGET_TERMS | 144 | module constant |
| code/src/hkg_t24/constants.py | constant ARWF_WARNING | 146 | module constant |
| code/src/hkg_t24/constants.py | constant CWA_WRF_WARNING | 150 | module constant |
| code/src/hkg_t24/constants.py | class SourceRegistryRow | 157 | Final-patch source-registry row. |
| code/src/hkg_t24/constants.py | def assert_no_forbidden_target_memory_names | 647 | Fail if finalized daily target-memory feature names use forbidden lag1 wording. |
| code/src/hkg_t24/db/connection.py | class DatabaseConfigError | 18 | Raised when required database configuration is missing. |
| code/src/hkg_t24/db/connection.py | class DatabaseUnavailable | 22 | Raised when the PostgreSQL runtime dependency is unavailable. |
| code/src/hkg_t24/db/connection.py | def get_database_url | 26 | Return the contract DSN using the final-patch priority order. |
| code/src/hkg_t24/db/connection.py | def import_psycopg | 44 |  |
| code/src/hkg_t24/db/connection.py | def connect | 52 |  |
| code/src/hkg_t24/db/connection.py | def redact_database_url | 57 |  |
| code/src/hkg_t24/db/connection.py | def database_url_hash | 67 |  |
| code/src/hkg_t24/db/ddl.py | class ExpectedColumn | 11 |  |
| code/src/hkg_t24/db/ddl.py | constant SCHEMA_SQL | 18 | module constant |
| code/src/hkg_t24/db/ddl.py | constant FOUNDATION_SQL | 20 | module constant |
| code/src/hkg_t24/db/ddl.py | constant SNAPSHOT_COMPAT_VIEW_SQL | 293 | module constant |
| code/src/hkg_t24/db/ddl.py | constant NWP_COMPAT_VIEW_SQL | 305 | module constant |
| code/src/hkg_t24/db/ddl.py | constant NWP_SAFE_VIEW_SQL | 323 | module constant |
| code/src/hkg_t24/db/ddl.py | constant EXPECTED_COLUMNS | 356 | module constant |
| code/src/hkg_t24/db/migrations.py | class ColumnConflict | 26 |  |
| code/src/hkg_t24/db/migrations.py | def git_commit | 33 |  |
| code/src/hkg_t24/db/migrations.py | def execute_sql | 47 |  |
| code/src/hkg_t24/db/migrations.py | def find_column_conflicts | 52 |  |
| code/src/hkg_t24/db/migrations.py | def write_schema_conflict_report | 77 |  |
| code/src/hkg_t24/db/migrations.py | def _relation_kind | 95 |  |
| code/src/hkg_t24/db/migrations.py | def _columns | 110 |  |
| code/src/hkg_t24/db/migrations.py | def migrate_legacy_feature_matrix_tables | 123 |  |
| code/src/hkg_t24/db/migrations.py | def write_source_registry_migration_report | 191 |  |
| code/src/hkg_t24/db/migrations.py | def apply_foundation_migrations | 207 |  |
| code/src/hkg_t24/db/migrations.py | def create_run_manifest | 230 |  |
| code/src/hkg_t24/db/migrations.py | def finish_run_manifest | 266 |  |
| code/src/hkg_t24/features/gribstream_safe_rows.py | def gribstream_tables_available | 12 |  |
| code/src/hkg_t24/features/gribstream_safe_rows.py | def refresh_nwp_safe_row_ledger | 18 |  |
| code/src/hkg_t24/features/gribstream_safe_rows.py | def write_gribstream_reports | 74 |  |
| code/src/hkg_t24/features/snapshot_builder.py | def build_target_memory_features | 23 | Build leakage-safe target-memory features ending at T-2 or older. |
| code/src/hkg_t24/features/snapshot_builder.py | def calendar_rows | 67 |  |
| code/src/hkg_t24/features/snapshot_builder.py | def populate_cutoff_calendar | 71 |  |
| code/src/hkg_t24/features/snapshot_builder.py | def _target_source_hash_expr | 106 |  |
| code/src/hkg_t24/features/snapshot_builder.py | def populate_target_labels | 110 |  |
| code/src/hkg_t24/features/snapshot_builder.py | def _safe_view_exists | 149 |  |
| code/src/hkg_t24/features/snapshot_builder.py | def populate_h24n_snapshots | 156 |  |
| code/src/hkg_t24/features/snapshot_builder.py | def write_snapshot_reports | 239 |  |
| code/src/hkg_t24/features/snapshot_builder.py | def build_snapshots | 314 |  |
| code/src/hkg_t24/features/source_contracts.py | class SourceContractResult | 22 |  |
| code/src/hkg_t24/features/source_contracts.py | def verify_lightgbm_required | 34 |  |
| code/src/hkg_t24/features/source_contracts.py | def _has_required_columns | 41 |  |
| code/src/hkg_t24/features/source_contracts.py | def _scalar_int | 60 |  |
| code/src/hkg_t24/features/source_contracts.py | def run_source_contract_checks | 67 |  |
| code/src/hkg_t24/features/source_contracts.py | def write_source_contract_reports | 211 |  |
| code/src/hkg_t24/timeutils.py | constant HONG_KONG_TZ | 11 | module constant |
| code/src/hkg_t24/timeutils.py | class CalendarRow | 15 | Deterministic cutoff-calendar row. |
| code/src/hkg_t24/timeutils.py | def formal_cutoff_utc | 34 | Return 15:00 HKT on T-1 as UTC. |
| code/src/hkg_t24/timeutils.py | def operational_freeze_utc | 40 | Return 14:45 HKT on T-1 as UTC. |
| code/src/hkg_t24/timeutils.py | def partition_for_target_date | 46 | Return the contract partition for a target date. |
| code/src/hkg_t24/timeutils.py | def season_for_month | 57 |  |
| code/src/hkg_t24/timeutils.py | def snapshot_id | 67 |  |
| code/src/hkg_t24/timeutils.py | def calendar_row | 71 |  |
| code/src/hkg_t24/timeutils.py | def iter_target_dates | 91 | Return inclusive target dates. |
| code/src/hkg_t24/timeutils.py | def assert_hong_kong_fixed_utc8 | 99 | Fail if Python timezone data no longer treats Hong Kong as fixed UTC+08. |
| code/src/hkg_t24/utils/hashing.py | def sha256_text | 11 |  |
| code/src/hkg_t24/utils/hashing.py | def sha256_json | 15 |  |
| code/src/hkg_t24/utils/hashing.py | def sha256_file | 20 |  |
| code/src/hkg_t24/utils/sql.py | constant IDENTIFIER_RE | 8 | module constant |
| code/src/hkg_t24/utils/sql.py | def quote_ident | 11 | Quote an already-validated PostgreSQL identifier. |
| code/src/hkg_t24/utils/sql.py | def qualified_name | 18 |  |
| code/src/hkg_t24/utils/sql.py | def csv_line | 22 | Render a small RFC4180-compatible CSV line. |
| code/tests/hkg_t24/test_database_url_priority.py | def test_database_url_wins_over_fallback_dsn | 11 |  |
| code/tests/hkg_t24/test_database_url_priority.py | def test_fallback_dsn_used_when_primary_absent | 26 |  |
| code/tests/hkg_t24/test_database_url_priority.py | def test_missing_database_dsn_error_is_exact | 33 |  |
| code/tests/hkg_t24/test_database_url_priority.py | def test_cli_fails_closed_without_dsn | 42 |  |
| code/tests/hkg_t24/test_h24n_contract_policy.py | def test_h24n_cutoff_and_freeze_are_final_patch_times | 24 |  |
| code/tests/hkg_t24/test_h24n_contract_policy.py | def test_cutoff_calendar_partitions_and_snapshot_id | 29 |  |
| code/tests/hkg_t24/test_h24n_contract_policy.py | def test_source_registry_contract_rows_are_final_patch_compatible | 38 |  |
| code/tests/hkg_t24/test_h24n_contract_policy.py | def test_dataset_prefix_mapping_and_schema_version_are_final_patch_values | 52 |  |
| code/tests/hkg_t24/test_h24n_contract_policy.py | def test_finalized_target_memory_lag1_names_are_forbidden | 58 |  |
| code/tests/hkg_t24/test_real_db_contracts.py | def test_real_db_source_contracts_when_dsn_is_available | 10 |  |
| code/tests/hkg_t24/test_schema_sql_contract.py | def test_feature_matrix_is_physical_and_snapshot_matrices_are_views_only | 11 |  |
| code/tests/hkg_t24/test_schema_sql_contract.py | def test_nwp_safe_view_contains_mandatory_filters | 19 |  |
| code/tests/hkg_t24/test_schema_sql_contract.py | def test_raw_response_object_compat_view_uses_final_column_names | 27 |  |
| code/tests/hkg_t24/test_schema_sql_contract.py | def test_live_prediction_scaffold_uses_final_patch_fields | 33 |  |
| code/tests/hkg_t24/test_schema_sql_contract.py | def test_validation_scaffold_tables_exist_without_model_outputs | 41 |  |
| code/tests/hkg_t24/test_snapshot_builder_synthetic.py | def _labels | 8 |  |
| code/tests/hkg_t24/test_snapshot_builder_synthetic.py | def test_synthetic_120_label_target_memory_counts_use_lag2_not_lag1 | 13 |  |
