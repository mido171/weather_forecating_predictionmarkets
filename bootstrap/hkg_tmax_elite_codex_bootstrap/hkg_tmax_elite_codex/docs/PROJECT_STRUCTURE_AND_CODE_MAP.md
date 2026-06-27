# Project Structure and Code Map

Last updated: 2026-06-27

This file is the canonical map for implementation code, tests, migrations, scripts, task records, and experiment evidence in this repository.

Agents must update this file whenever code, tests, scripts, migrations, A-to-Z tasks, or experiment-output mappings are added, moved, renamed, or retired.

## Top-Level Layout

```text
.
├── code/
│   ├── src/                         Python packages shipped by this repo
│   └── tests/                       Pytest suite
├── scripts/                         Manual and automated runner scripts
├── migrations/                      SQL migrations
├── config/                          YAML config and source registries
├── experiments/                     Immutable experiment and task evidence folders
├── tasks/                           A-to-Z task package and completed/not-completed records
├── docs/                            Human documentation and specifications
├── documentation/                   Strategy docs, runbooks, and project-root references
├── data/                            Local data inventory and audit snapshots
├── reports/                         Generated and curated reports
├── artifacts/                       Durable artifact indexes and small metadata
├── schemas/                         Machine-readable schemas
├── sql/                             Reviewed SQL/query assets outside durable migrations
├── analysis/                        Analysis inputs/outputs
├── metadata/                        Derived metadata artifacts
├── models/                          Model artifacts
├── predictions/                     Prediction outputs
└── run_logs/                        Runtime logs
```

## Executable Python Packages

All package source code now lives under `code/src`.

`code/src/hkg_tmax/` contains the main research and acquisition package:

```text
code/src/hkg_tmax/__main__.py              CLI module entry
code/src/hkg_tmax/cli.py                   Main hkg_tmax CLI
code/src/hkg_tmax/config.py                Repo-root discovery and YAML config loading
code/src/hkg_tmax/doctor.py                Environment and required-path checks
code/src/hkg_tmax/acquisition.py           Raw acquisition storage
code/src/hkg_tmax/acquisition_contracts.py Acquisition contract reporting
code/src/hkg_tmax/analysis_contracts.py    Analysis contract helpers
code/src/hkg_tmax/asof.py                  As-of and cutoff helpers
code/src/hkg_tmax/bronze.py                Bronze-layer construction
code/src/hkg_tmax/collector.py             Source collector runner
code/src/hkg_tmax/distribution.py          Distribution and bucket math
code/src/hkg_tmax/experiments.py           Experiment folder creation/indexing
code/src/hkg_tmax/fetch.py                 HTTP fetch/archive helpers
code/src/hkg_tmax/hashing.py               Hash helpers
code/src/hkg_tmax/hko.py                   HKO parsing
code/src/hkg_tmax/hko_backfill.py          HKO backfill helpers
code/src/hkg_tmax/manifest.py              Manifest builder
code/src/hkg_tmax/market.py                Market snapshot helpers
code/src/hkg_tmax/metrics.py               Forecast metrics
code/src/hkg_tmax/milestones.py            Milestone rendering
code/src/hkg_tmax/publication.py           Publication timing helpers
code/src/hkg_tmax/settlement.py            Settlement and bucket rules
code/src/hkg_tmax/source_normalization.py  Source normalization
code/src/hkg_tmax/sources.py               Source inventory reporting
code/src/hkg_tmax/static_context.py        Static station/context generation
code/src/hkg_tmax/statistics.py            Statistical helpers
code/src/hkg_tmax/storage.py               Local storage helpers
code/src/hkg_tmax/target.py                Target-label validation
code/src/hkg_tmax/timeutils.py             Time and horizon utilities
code/src/hkg_tmax/validation.py            Validation commands
```

`code/src/hkg_tmax/gribstream/` contains the reusable GribStream acquisition client introduced by T06:

```text
code/src/hkg_tmax/gribstream/client.py      Secret-safe one-thread HTTP client, retries, raw NDJSON gzip landing
code/src/hkg_tmax/gribstream/catalog.py     Live shared-parameter selector resolver
code/src/hkg_tmax/gribstream/planner.py     `/runs` request planner and quota/row estimates
code/src/hkg_tmax/gribstream/normalizer.py  `/runs` NDJSON normalization to run/valid/lead rows
code/src/hkg_tmax/gribstream/store.py       PostgreSQL raw_audit/nwp_core lineage persistence
```

`code/src/hkg_tmax/hkg_t24/` contains HKG T-24-specific policy code:

```text
code/src/hkg_tmax/hkg_t24/governance.py    Governance artifact builder
code/src/hkg_tmax/hkg_t24/guard.py         Locked-period and leakage guards
code/src/hkg_tmax/hkg_t24/moisture.py      Moisture feature helpers
code/src/hkg_tmax/hkg_t24/peak_anatomy.py  Peak-temperature analysis helpers
```

`code/src/hkg_t24/` is the dedicated package for the HKG T+24 / H24N full strategy implementation contract and the four Jira packets under `documentation/strategy_implementation_documentation/actual_strategy_implementation_contract/jira_breakdow/`.

The strategy contract path `src/hkg_t24` maps to this repository path:

```text
code/src/hkg_t24/
```

Initial ownership map:

```text
code/src/hkg_t24/audit/          Source registries, schema checks, leakage audit events, provenance
code/src/hkg_t24/db/             Database connection, migrations, DDL helpers, SQL execution boundaries
code/src/hkg_t24/features/       H24N snapshots, official/target/NWP/proxy feature builders, matrices
code/src/hkg_t24/models/         Expert models, OOF generation, routers, specialists, distribution
code/src/hkg_t24/validation/     Scoreboards, leakage tests, negative controls, sealed validation
code/src/hkg_t24/live/           Live prediction, replay prediction, post-settlement scoring
code/src/hkg_t24/orchestration/  End-to-end phase runners and full-pipeline coordination
code/src/hkg_t24/artifacts/      Candidate freeze and artifact manifest helpers
code/src/hkg_t24/utils/          Narrow shared helpers with no domain-policy ownership
```

`code/src/hkg_tmax_db/` contains the database-ingestion package:

```text
code/src/hkg_tmax_db/cli.py                hkg-tmax-db CLI
code/src/hkg_tmax_db/connection.py         DB connection and migration helpers
code/src/hkg_tmax_db/contracts.py          Audit-bundle validation
code/src/hkg_tmax_db/cutoff.py             Canonical T-24 cutoff and eligibility code
code/src/hkg_tmax_db/hashing.py            DB-side hash helpers
code/src/hkg_tmax_db/psql_loader.py        Direct PostgreSQL loader
code/src/hkg_tmax_db/reconciliation.py     Source reconciliation
code/src/hkg_tmax_db/reports.py            DB ingestion reports
```

## Tests

All pytest files now live under `code/tests`.

Important test groups:

```text
code/tests/test_config_and_sources.py                 Config/source registry tests
code/tests/test_validation.py                         Validation command tests
code/tests/test_acquisition*.py                       Acquisition and contract tests
code/tests/test_hko*.py                               HKO parsing and backfill tests
code/tests/test_target.py                             Target-label tests
code/tests/test_settlement.py                         Settlement/bucket tests
code/tests/test_distribution.py                       Distribution tests
code/tests/test_market.py                             Market snapshot tests
code/tests/test_hkg_t24_*.py                          HKG T-24 experiment, guard, and specialist tests
code/tests/test_hkg_tmax_db_audit_ingestion.py        Audit-driven DB ingestion tests
code/tests/test_hkg_t24_time_availability_contract.py T01 cutoff/availability contract tests
code/tests/test_t02_full_current_data_census_reconciliation.py T02 census registry tests
code/tests/test_t03_t05_foundation_tasks.py T03-T05 GribStream/NWP/location foundation tests
code/tests/test_t06_gribstream_resumable_runs_client.py T06 GribStream client/retry/normalization tests
code/tests/test_t07_t13_gribstream_backfill.py Legacy T07-T13 broad-runner planner/credential tests
code/tests/test_tactical_gribstream_h24n.py Tactical H24N GribStream schema and exact-timesList payload tests
code/tests/hkg_t24/                              Dedicated tests for the full HKG T+24 strategy package
```

Current count: 146 Python test files under `code/tests`.

The configured test root is `code/tests` in `pyproject.toml`.

## Scripts

Manual and automation entrypoints remain under `scripts/`.

Current script counts:

```text
scripts/*.py   161 Python entrypoints
scripts/*.ps1  6 Windows PowerShell helpers
scripts/*.sh   4 shell helpers
```

Script conventions:

- `scripts/run_hkg_t24_####_slug.py` usually writes to `experiments/####_slug/`.
- `scripts/run_hkg_tmax_*.py` writes audit, baseline, or dataset reports.
- Windows collector controls live in `scripts/install_windows_collectors.ps1`, `scripts/start_collectors.ps1`, `scripts/stop_collectors.ps1`, and `scripts/uninstall_windows_collectors.ps1`.
- Direct `python scripts/...` commands assume the package has been installed from this repo, normally by `python -m pip install -e ".[research,dev]"` or the repo bootstrap flow.

## Database Migrations

SQL migrations live under `migrations/postgres/`.

Current migrations:

```text
migrations/postgres/20260623_0001_audit_driven_ingestion.sql
migrations/postgres/20260624_0002_t24_time_availability_contract.sql
migrations/postgres/20260624_0003_t02_census_registry_compatibility.sql
migrations/postgres/20260624_0004_t03_gribstream_catalog_registry.sql
migrations/postgres/20260624_0005_t04_nwp_storage_lineage.sql
migrations/postgres/20260624_0006_t05_location_station_geospatial_registry.sql
migrations/postgres/20260625_0007_tactical_gribstream_h24n_schema.sql
```

Code/tests linked to those migrations:

```text
code/src/hkg_tmax_db/cli.py
code/src/hkg_tmax_db/psql_loader.py
code/tests/test_hkg_tmax_db_audit_ingestion.py
code/tests/test_hkg_t24_time_availability_contract.py
code/tests/test_t02_full_current_data_census_reconciliation.py
code/tests/test_t03_t05_foundation_tasks.py
```

## A-to-Z Task Package

The A-to-Z package lives here:

```text
tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/
```

Task state lives here:

```text
tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/tasks/completed/
tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/tasks/not-completed/
```

Status index:

```text
tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/TASK_STATUS_INDEX.csv
```

Completed implementation task records:

```text
tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/tasks/completed/T00_repository_database_preflight/
tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/tasks/completed/T01_canonical_t24_time_availability_contract/
tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/tasks/completed/T02_full_current_data_census_reconciliation/
tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/tasks/completed/T03_gribstream_catalog_coverage_licence_quota_audit/
tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/tasks/completed/T04_nwp_database_object_storage_migrations/
tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/tasks/completed/T05_canonical_location_station_geospatial_registry/
tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/tasks/completed/T06_gribstream_resumable_runs_client/
```

## Experiment Evidence

Experiment and task evidence folders live under `experiments/`.

Main convention:

```text
experiments/####_slug/
```

For script-driven experiments, the usual relationship is:

```text
scripts/run_hkg_t24_####_slug.py
  -> experiments/####_slug/
  -> code/tests/test_hkg_t24_####_slug.py when a dedicated test exists
```

Recent A-to-Z evidence mappings:

```text
T00 task record:
tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/tasks/completed/T00_repository_database_preflight/

T00 evidence:
experiments/0207_repository_database_preflight/

T01 task record:
tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/tasks/completed/T01_canonical_t24_time_availability_contract/

T01 evidence:
experiments/0208_canonical_t24_time_availability_contract/

T01 implementation:
code/src/hkg_tmax_db/cutoff.py
migrations/postgres/20260624_0002_t24_time_availability_contract.sql
code/tests/test_hkg_t24_time_availability_contract.py

T02 task record:
tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/tasks/completed/T02_full_current_data_census_reconciliation/

T02 evidence:
experiments/0209_full_current_data_census_reconciliation/

T02 implementation:
scripts/run_t02_full_current_data_census_reconciliation.py
migrations/postgres/20260624_0003_t02_census_registry_compatibility.sql
code/tests/test_t02_full_current_data_census_reconciliation.py

T03-T05 completed implementation:
scripts/run_t03_t05_foundation_tasks.py
scripts/check_t03_t05_status.py
migrations/postgres/20260624_0004_t03_gribstream_catalog_registry.sql
migrations/postgres/20260624_0005_t04_nwp_storage_lineage.sql
migrations/postgres/20260624_0006_t05_location_station_geospatial_registry.sql
code/tests/test_t03_t05_foundation_tasks.py

T03-T05 evidence:
experiments/0210_gribstream_catalog_coverage_licence_quota_audit/
experiments/0211_nwp_database_object_storage_migrations/
experiments/0212_canonical_location_station_geospatial_registry/

T03-T05 task records:
tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/tasks/completed/T03_gribstream_catalog_coverage_licence_quota_audit/
tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/tasks/completed/T04_nwp_database_object_storage_migrations/
tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/tasks/completed/T05_canonical_location_station_geospatial_registry/

T06 completed implementation:
code/src/hkg_tmax/gribstream/client.py
code/src/hkg_tmax/gribstream/catalog.py
code/src/hkg_tmax/gribstream/planner.py
code/src/hkg_tmax/gribstream/normalizer.py
code/src/hkg_tmax/gribstream/store.py
scripts/run_t06_gribstream_resumable_runs_client.py
scripts/check_t06_gribstream_status.py
config/acquisition_policy.yaml
code/tests/test_t06_gribstream_resumable_runs_client.py

T06 evidence:
experiments/0213_gribstream_resumable_runs_client/
data/_pipeline_internal/raw/gribstream/gfs/runs/run_time_utc=20260623_000000/ecfb27dcebbbfbf058049cf321478c6309cacc9dca381e797697d6a80b3715f4.ndjson.gz

T06 task record:
tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/tasks/completed/T06_gribstream_resumable_runs_client/

T07-T12 consolidated tactical GribStream implementation:
scripts/reset_tactical_gribstream_store.py
scripts/run_tactical_gribstream_h24n_smoke.py
scripts/run_tactical_gribstream_first_week.py
scripts/run_tactical_gribstream_batch_smoke.py
scripts/audit_tactical_gribstream_deep_sanity.py
migrations/postgres/20260625_0007_tactical_gribstream_h24n_schema.sql
code/tests/test_tactical_gribstream_h24n.py
documentation/T07_T12_CONSOLIDATED_TACTICAL_GRIBSTREAM_BACKFILL_RUNBOOK.md

T07-T12 tactical evidence and ledgers:
experiments/0214_tactical_h24n_gribstream_backfill/
experiments/0214_tactical_h24n_gribstream_backfill/first_week_pull/
experiments/0214_tactical_h24n_gribstream_backfill/batch_smoke_10w/
experiments/0214_tactical_h24n_gribstream_backfill/full_tactical_backfill_ok_tmax/
data/_pipeline_internal/raw/gribstream_tactical_smoke/
data/_pipeline_internal/raw/gribstream_tactical_first_week/
data/_pipeline_internal/raw/gribstream_tactical_batch_smoke_10w/
data/_pipeline_internal/raw/gribstream_tactical_full_tactical_backfill_ok_tmax/
documentation/T07_T12_FULL_TACTICAL_BACKFILL_20260625_RESULT.md
documentation/T07_T12_DEEP_SANITY_AUDIT_20260625.md
documentation/strategy_implementation_documentation/GRIBSTREAM_FETCHED_DATA_INVENTORY_20260626.md
documentation/strategy_implementation_documentation/GRIBSTREAM_LEAKAGE_SAFE_DB_RETRIEVAL_LEDGER_20260626.md

T07-T12 active consolidated task record:
tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/tasks/not-completed/T07_T12_tactical_h24n_gribstream_backfill/

T07-T12 superseded split task records:
tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/tasks/superseded/T07_T12_legacy_split_gribstream_fetch_tasks/

Retired broad GribStream runner:
scripts/run_t07_t13_gribstream_backfill.py

T13 remains separate and not-completed because it is not a GribStream task:
tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/tasks/not-completed/T13_hko_arwf_exact_vintage_collector/
```

## Strategy Implementation Contract Layout

The four dense implementation Jira packets live under:

```text
documentation/strategy_implementation_documentation/actual_strategy_implementation_contract/jira_breakdow/
```

Each packet folder contains the Jira markdown file, `IMPLEMENTATION_PACKET.md`, `CONTRACT_COVERAGE_TEMPLATE.md`, and copied binding contract docs under `binding_contract_docs/`.

All implementation code for these packets must live under `code/src/hkg_t24/`.

Supporting files for the full strategy implementation use these locations:

```text
code/tests/hkg_t24/          tests
config/hkg_t24/              configuration
sql/hkg_t24/                 reviewed SQL/query assets
migrations/postgres/         durable PostgreSQL migrations
schemas/hkg_t24/             machine-readable schemas
reports/hkg_t24/             report indexes and non-canonical reports
reports/hkg_t24/jira_coverage/
artifacts/hkg_t24/           artifact indexes and small durable metadata
```

The project-root secondary reference docs and metadata files that previously lived as loose root files now live under:

```text
documentation/project_root_reference/
```

Core repository entry files still intentionally remain at the root: `README.md`, `CODEX_START_HERE.md`, `AGENTS.md`, `FIRST_GOALS.md`, `MILESTONES.md`, `EXPERIMENT_INDEX.md`, `CHANGELOG.md`, `SECURITY.md`, `pyproject.toml`, requirements files, `Makefile`, `Dockerfile`, and `compose.yaml`.

## Build and Import Configuration

The package root is `code/src`.

Files that define this:

```text
pyproject.toml
Makefile
Dockerfile
```

Important details:

- `pyproject.toml` installs packages from `code/src`.
- `pyproject.toml` discovers tests under `code/tests`.
- `Makefile` exports `PYTHONPATH=$(CURDIR)/code/src:$(CURDIR)`.
- Existing virtual environments need `python -m pip install -e ".[research,dev]"` after this layout change so editable package metadata points at `code/src`.

## Update Rule for Future Agents

Update this file in the same work unit when:

- adding, moving, or deleting source code under `code/src`;
- adding, moving, or deleting tests under `code/tests`;
- adding, moving, or deleting scripts under `scripts`;
- adding migrations under `migrations`;
- completing A-to-Z tasks under `tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION`;
- creating experiment folders under `experiments`;
- changing package discovery, pytest discovery, import-path hooks, Docker, or Makefile paths.

Do not leave a code change without a matching map update when the change affects where future agents should look.
