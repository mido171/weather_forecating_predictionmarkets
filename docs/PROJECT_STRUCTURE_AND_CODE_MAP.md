# Project Structure and Code Map

Last updated: 2026-07-08

This file is the canonical map for implementation code, tests, migrations, scripts, task records, and experiment evidence in this repository.

Agents must update this file whenever code, tests, scripts, migrations, A-to-Z tasks, or experiment-output mappings are added, moved, renamed, or retired.

## Top-Level Layout

```text
.
├── code/
│   ├── src/                         Python packages shipped by this repo
│   └── tests/                       Pytest suite
├── apps/                            Local web frontends served by repo services
├── scripts/                         Manual and automated runner scripts
├── migrations/                      SQL migrations
├── config/                          YAML config and source registries
├── configs/                         Experiment runner configs
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

`code/src/hkg_tmax/data/`, `features/`, `modeling/`, and `evaluation/` contain the HKG Tmax residual-ML strategy pipeline:

```text
code/src/hkg_tmax/data/forecast_anchor.py          Info.gov local forecast anchor selection and revision features
code/src/hkg_tmax/data/anchor_provenance_audit.py  Early-cutoff forecast-anchor provenance and missing-anchor audit
code/src/hkg_tmax/data/hourly_readings_features.py Info.gov hourly reading, station-network, warning, and cyclone features
code/src/hkg_tmax/data/target_history_features.py  Lag2+ target history and climatology features
code/src/hkg_tmax/data/official_residual_memory_features.py Lag-safe same-cutoff official residual-memory feature builder and publication-safety audit
code/src/hkg_tmax/features/feature_registry.py     Feature family and lineage registry
code/src/hkg_tmax/features/leakage_guards.py       Cutoff, source, raw-payload, and target-lag leakage audits
code/src/hkg_tmax/features/pruned_feature_policy.py Pruned next-round residual-ML feature policy and router feature guards
code/src/hkg_tmax/features/residual_memory_policy.py Residual-memory feature allow/deny policy and target/evaluation leakage guards
code/src/hkg_tmax/features/station_groups.py       Station grouping definitions for network-gradient features
code/src/hkg_tmax/features/text_regime_flags.py    Forecast/hourly text and warning flag extraction
code/src/hkg_tmax/modeling/baselines.py            Raw official, climatology, and grouped-residual baselines
code/src/hkg_tmax/modeling/residual_models.py      LightGBM, CatBoost, Huber, and direct diagnostic model wrappers
code/src/hkg_tmax/modeling/ensemble.py             Nonnegative residual ensemble and shrinkage guard
code/src/hkg_tmax/modeling/selective_router.py     Selective correction / abstention router for next-round residual ML
code/src/hkg_tmax/modeling/tail_specialist.py      Tail-error specialist overlay for high-error rows
code/src/hkg_tmax/evaluation/ablation_runner.py    Rolling validation, holdout, sealed confirmation, and ablation runner
code/src/hkg_tmax/evaluation/official_residual_memory_runner.py Governed D0-D5 official residual-memory benchmark helpers
code/src/hkg_tmax/evaluation/metrics.py            MAE/RMSE/bias/tail scoring helpers
code/src/hkg_tmax/evaluation/no_harm_reporting.py  Help/worse, apply-rate, decile, and no-harm guardrail reporting
code/src/hkg_tmax/evaluation/reporting.py          Artifact writers, model card, audits, and diagnostics
```

`code/src/hkg_tmax/hkg_t24/` contains HKG T-24-specific policy code:

```text
code/src/hkg_tmax/hkg_t24/governance.py    Governance artifact builder
code/src/hkg_tmax/hkg_t24/guard.py         Locked-period and leakage guards
code/src/hkg_tmax/hkg_t24/moisture.py      Moisture feature helpers
code/src/hkg_tmax/hkg_t24/peak_anatomy.py  Peak-temperature analysis helpers
```

`code/src/hkg_tmax_probability/` contains the HKG Tmax probability bucket calibration V1 package and V2 distribution-method challenger benchmark:

```text
code/src/hkg_tmax_probability/bucket_rules.py              Decimal-safe bucket keys and boundaries
code/src/hkg_tmax_probability/forecast_selection.py        Latest eligible Info.gov local forecast selection by cutoff
code/src/hkg_tmax_probability/data_build.py                PostgreSQL modeling-table builder and row-count artifacts
code/src/hkg_tmax_probability/label_publication_audit.py   Raw Daily Extract first-publication label audit
code/src/hkg_tmax_probability/leakage_audit.py             Cutoff, row-identity, sealed, and no-trading leakage checks
code/src/hkg_tmax_probability/models.py                    B0-B6, P1-P2, C1-C2, K0-K2, and S1 probability methods
code/src/hkg_tmax_probability/distribution_methods_v2.py   E1/E2/E3 EMOS, G1 tree location-scale, Q1/Q2 CDF, T1 time-decay B4, and H1 hybrid challengers
code/src/hkg_tmax_probability/leaderboard_v2.py            V2 B4 promotion-gate and champion-selection contract
code/src/hkg_tmax_probability/scoring.py                   Normalized RPS, NLL, Brier, CRPS, ECE/MCE, entropy
code/src/hkg_tmax_probability/reporting.py                 Scoreboards, diagnostics, model card, and manifest writers
code/src/hkg_tmax_probability/live_inference.py            Weather-probability-only live inference example writer
code/src/hkg_tmax_probability/*_pmf.py / *_calibration.py  Thin wrappers for method-family imports
```

`code/src/hkg_tmax_demo_trading/` contains the local-only HKG Polymarket demo backtester backend:

```text
code/src/hkg_tmax_demo_trading/domain.py       Fictitious trade math, EV, PnL, date/window helpers
code/src/hkg_tmax_demo_trading/market_data.py  Read-only Polymarket Gamma/CLOB market metadata adapters
code/src/hkg_tmax_demo_trading/probability.py  HKO local/FND live parsing, validated local-cutoff forecast selection, and B4 profile-filtered probability snapshots
code/src/hkg_tmax_demo_trading/store.py        PostgreSQL demo_trading ledger persistence, snapshot profile lookup, open-contract guards, and manual win/loss settlement writes
code/src/hkg_tmax_demo_trading/service.py      Market, trade, settlement, account, executable-CLOB, strategy gates, stale snapshot source validation, and manual 98c-win / 97%-max-loss settlement policy
code/src/hkg_tmax_demo_trading/api.py          FastAPI endpoints for markets, profiles, trades, account, reset, automatic settle, manual settle-win, and manual settle-loss
code/src/hkg_tmax_demo_trading/server.py       Uvicorn entrypoint, defaulting to localhost port 6000
```

`apps/hkg-polymarket-backtester/` contains the React/Vite frontend served by `hkg_tmax_demo_trading.server`:

```text
apps/hkg-polymarket-backtester/package.json    Vite/React scripts and UI dependencies
apps/hkg-polymarket-backtester/index.html      Vite entry HTML
apps/hkg-polymarket-backtester/src/main.jsx    React mount
apps/hkg-polymarket-backtester/src/App.jsx     Backtester UI, profile selector, guarded market API state, ticket gates, account panels, and manual win/loss ledger settlement actions
apps/hkg-polymarket-backtester/src/styles.css  Responsive operational layout for profiles, edge stack, ticket gates, manual settlement states, and ledger metadata
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

HKG-T24-001 foundation implementation entry points:

```text
code/src/hkg_t24/cli.py                         Jira 001 CLI commands
code/src/hkg_t24/constants.py                   Binding constants, source rows, schema versions
code/src/hkg_t24/timeutils.py                   H24N cutoff/freeze/calendar helpers
code/src/hkg_t24/db/connection.py               Contract DSN policy and psycopg boundary
code/src/hkg_t24/db/ddl.py                      Foundation DDL and compatibility views
code/src/hkg_t24/db/migrations.py               Idempotent schema/run-manifest orchestration
code/src/hkg_t24/audit/source_registry.py       Final source registry rows and CSV export
code/src/hkg_t24/audit/schema_contracts.py      Source-table discovery primitives
code/src/hkg_t24/audit/leakage_events.py        Leakage audit event writes/counts
code/src/hkg_t24/features/source_contracts.py   Phase0 source-contract checks
code/src/hkg_t24/features/snapshot_builder.py   H24N calendar, target labels, snapshots, target memory
code/src/hkg_t24/features/gribstream_safe_rows.py GribStream safe-row ledger and reports
```

HKG-T24-003 router, specialist, final forecast, and distribution implementation entry points:

```text
code/src/hkg_t24/cli.py                         Jira 003 train-router, train-specialists, train-distribution, run-system-replay commands
code/src/hkg_t24/constants.py                   Router IDs, expert caps, specialist IDs, distribution thresholds, Jira003 reports
code/src/hkg_t24/db/ddl.py                      Router, specialist, system prediction, and scoreboard tables
code/src/hkg_t24/features/feature_dictionary.py Strict dictionary additions for router/specialist-derived features
code/src/hkg_t24/features/nwp_daily.py          Extra contract feature builders used by specialists
code/src/hkg_t24/models/static_weights.py       Static SLSQP weights, cap masking, dynamic blend helpers
code/src/hkg_t24/models/expected_error.py       Per-expert expected absolute error models and fallback
code/src/hkg_t24/models/router.py               R0/R1 training, promotion, reports, and persistence
code/src/hkg_t24/models/specialists.py          S1-S6 specialist scoring, correction, no-harm, reports, persistence
code/src/hkg_t24/models/final_formula.py        R1/R0/E0/E2 fallback, specialist cap, official clip, component provenance
code/src/hkg_t24/models/distribution.py         Quantile calibration, empirical fallback, probabilities, no-trade, persistence
code/src/hkg_t24/models/system_replay.py        End-to-end strict-pre2024 Jira003 replay orchestration
code/src/hkg_t24/validation/metrics.py          Shared MAE/RMSE/bias/tail metrics
code/src/hkg_t24/validation/slices.py           Monthly system slice scoreboards
code/src/hkg_t24/validation/ablation.py         Final-vs-pre-distribution ablation matrix
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
code/tests/test_hko_info_gov_hourly_readings_backfill.py Info.gov HKO hourly-reading parser tests
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
code/tests/test_hkg_t24_0215_gpt_pro_point_forecast_strategy.py GPT-Pro HKO lead-1 point forecast strategy tests
code/tests/test_hkg_tmax_residual_ml_strategy.py HKG Tmax residual-ML cutoff, anchor, lag, and leakage tests
code/tests/test_hkg_tmax_next_round_selective_router.py HKG Tmax next-round router, tail, no-harm, and provenance tests
code/tests/test_hkg_tmax_official_residual_memory.py HKG Tmax 0003 official residual-memory lag safety, row identity, leakage, and promotion-gate tests
code/tests/test_hkg_tmax_probability_bucket_v1.py HKG Tmax probability bucket V1 rules, selection, scoring, calibration, stack, and no-trading tests
code/tests/test_hkg_tmax_probability_distribution_methods_v2.py HKG Tmax probability V2 EMOS/CDF/time-decay, leakage, governance, B4-regression, and champion-gate tests
code/tests/test_demo_trading_probability.py HKG demo trading B4 profile filtering and probability snapshot tests
code/tests/test_demo_trading_service.py HKG demo trading service tests for CLOB entries, profile gates, fallback snapshots, win/loss settlement, and account views
code/tests/test_demo_trading_api.py HKG demo trading FastAPI surface tests for profiles, markets, trades, account, reset, manual settlement, and error mapping
code/tests/hkg_t24/                              Dedicated tests for the full HKG T+24 strategy package
```

Current count: 171 Python test files under `code/tests`.

The configured test root is `code/tests` in `pyproject.toml`.

## Scripts

Manual and automation entrypoints remain under `scripts/`.

Current script counts:

```text
scripts/*.py   169 Python entrypoints
scripts/*.ps1  7 Windows PowerShell helpers
scripts/*.sh   4 shell helpers
```

Script conventions:

- `scripts/run_hkg_t24_####_slug.py` usually writes to `experiments/####_slug/`.
- `scripts/run_hkg_tmax_*.py` writes audit, baseline, or dataset reports.
- Windows collector controls live in `scripts/install_windows_collectors.ps1`, `scripts/start_collectors.ps1`, `scripts/stop_collectors.ps1`, and `scripts/uninstall_windows_collectors.ps1`.
- Direct `python scripts/...` commands assume the package has been installed from this repo, normally by `python -m pip install -e ".[research,dev]"` or the repo bootstrap flow.

Hourly-reading acquisition:

```text
scripts/backfill_hko_info_gov_hourly_readings.py      Info.gov PRESS WEATHER HOURLY READINGS raw HTML backfill, normalization, reports, and one-table Postgres load
scripts/backfill_public_weather_to_postgres.py        Lean public GFS/GEFS/Himawari/radar source-issue backfill into weather_backfill Postgres tables with serial and optimized low-staging modes
scripts/benchmark_public_weather_speed_optimization.py Bounded speed benchmark for S3 byte-range GFS/GEFS fetches, Himawari fetch/decode workers, raw cleanup, and CPU/staging telemetry
scripts/run_public_weather_backfill_day_shards.py     Day-sharded launcher for multi-day public weather DB backfills, including optimized worker flag passthrough
scripts/run_hkg_tmax_residual_ml_strategy.py          HKG Tmax residual-ML matrix, ablation, ensemble, leakage audit, and report generator
scripts/run_hkg_tmax_residual_ml_next_round.py        HKG Tmax next-round pruned C1, selective C2, tail C3, no-harm, and anchor-provenance runner
scripts/run_hkg_tmax_residual_ml_official_memory.py   HKG Tmax 0003 official residual-memory D0-D5 point-forecast benchmark runner
scripts/run_hkg_tmax_probability_bucket_v1.py         HKG Tmax probability bucket calibration benchmark, scoreboards, audits, and diagnostics
scripts/run_hkg_tmax_probability_distribution_methods_v2.py HKG Tmax probability V2 EMOS/challenger benchmark, promotion gates, scoreboards, audits, and diagnostics
configs/hkg_tmax/residual_ml_strategy.yaml            Residual-ML date window, cutoff profile, source, and sealed-mode config
configs/hkg_tmax/residual_ml_next_round.yaml          Next-round pruned feature, router, tail, and early-cutoff audit config
configs/hkg_tmax/residual_ml_official_memory.yaml     Official residual-memory D0-D5 benchmark config, lag-2 contract, gates, and grids
configs/hkg_tmax/probability_bucket_v1.yaml           Probability bucket cutoffs, temporal governance, model grids, scoring, bootstrap, and acceptance gates
configs/hkg_tmax/probability_distribution_methods_v2.yaml Probability V2 EMOS/challenger grids, B4 promotion gates, cutoff sensitivity, and weather-only exclusions
```

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
migrations/postgres/20260704_0008_hko_info_gov_hourly_readings.sql
migrations/postgres/20260706_0009_demo_trading_backtester.sql
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

GPT-Pro HKO lead-1 point forecast strategy implementation:
scripts/run_hkg_t24_0215_gpt_pro_point_forecast_strategy.py
code/tests/test_hkg_t24_0215_gpt_pro_point_forecast_strategy.py

GPT-Pro HKO lead-1 point forecast strategy evidence:
experiments/0215_gpt_pro_point_forecast_strategy/
experiments/0215_gpt_pro_point_forecast_strategy/final_forecasting_strategy_report.md
experiments/0215_gpt_pro_point_forecast_strategy/results/baseline_scoreboard.csv
experiments/0215_gpt_pro_point_forecast_strategy/results/model_scoreboard.csv
experiments/0215_gpt_pro_point_forecast_strategy/results/cutoff_scoreboard.csv
experiments/0215_gpt_pro_point_forecast_strategy/results/selected_model_metadata.json
experiments/0215_gpt_pro_point_forecast_strategy/artifacts/all_cutoff_features.parquet
experiments/0215_gpt_pro_point_forecast_strategy/artifacts/all_oof_predictions.parquet

HKG Tmax residual-ML strategy implementation:
scripts/run_hkg_tmax_residual_ml_strategy.py
configs/hkg_tmax/residual_ml_strategy.yaml
code/tests/test_hkg_tmax_residual_ml_strategy.py

HKG Tmax residual-ML strategy evidence:
experiments/hkg_tmax_residual_ml_strategy/results/
experiments/hkg_tmax_residual_ml_strategy/results/final_model_card.md
experiments/hkg_tmax_residual_ml_strategy/results/scoreboard.csv
experiments/hkg_tmax_residual_ml_strategy/results/leakage_audit.json
experiments/hkg_tmax_residual_ml_strategy/results/model_selection_log.json
experiments/hkg_tmax_residual_ml_strategy/results/prediction_rows.parquet

HKG Tmax probability bucket calibration V1 implementation:
scripts/run_hkg_tmax_probability_bucket_v1.py
configs/hkg_tmax/probability_bucket_v1.yaml
code/src/hkg_tmax_probability/
code/tests/test_hkg_tmax_probability_bucket_v1.py

HKG Tmax probability bucket calibration V1 evidence:
experiments/hkg_tmax_probability_buckets_v1/results/
experiments/hkg_tmax_probability_buckets_v1/results/scoreboard.csv
experiments/hkg_tmax_probability_buckets_v1/results/scoreboard_by_split.csv
experiments/hkg_tmax_probability_buckets_v1/results/scoreboard_by_cutoff.csv
experiments/hkg_tmax_probability_buckets_v1/results/leakage_audit.json
experiments/hkg_tmax_probability_buckets_v1/results/label_publication_audit.json
experiments/hkg_tmax_probability_buckets_v1/results/final_probability_model_card.md
experiments/hkg_tmax_probability_buckets_v1/results/reproducibility_manifest.json

HKG Tmax probability distribution methods V2 implementation:
scripts/run_hkg_tmax_probability_distribution_methods_v2.py
configs/hkg_tmax/probability_distribution_methods_v2.yaml
code/src/hkg_tmax_probability/distribution_methods_v2.py
code/src/hkg_tmax_probability/leaderboard_v2.py
code/tests/test_hkg_tmax_probability_distribution_methods_v2.py

HKG Tmax probability distribution methods V2 evidence:
experiments/hkg_tmax_probability_distribution_methods_v2/
experiments/hkg_tmax_probability_distribution_methods_v2/README.md
experiments/hkg_tmax_probability_distribution_methods_v2/STATUS.yaml
experiments/hkg_tmax_probability_distribution_methods_v2/IMPLEMENTATION_DEEP_DIVE.md
experiments/hkg_tmax_probability_distribution_methods_v2/results/scoreboard.csv
experiments/hkg_tmax_probability_distribution_methods_v2/results/scoreboard_by_split.csv
experiments/hkg_tmax_probability_distribution_methods_v2/results/scoreboard_by_cutoff.csv
experiments/hkg_tmax_probability_distribution_methods_v2/results/scoreboard_by_high_bucket.csv
experiments/hkg_tmax_probability_distribution_methods_v2/results/proper_score_deltas_bootstrap.csv
experiments/hkg_tmax_probability_distribution_methods_v2/results/continuous_distribution_params.parquet
experiments/hkg_tmax_probability_distribution_methods_v2/results/leakage_audit.json
experiments/hkg_tmax_probability_distribution_methods_v2/results/row_identity_gate.json
experiments/hkg_tmax_probability_distribution_methods_v2/results/final_probability_model_card.md
experiments/hkg_tmax_probability_distribution_methods_v2/results/supreme_method_summary.md
experiments/hkg_tmax_probability_distribution_methods_v2/results/reproducibility_manifest.json

HKG Polymarket demo backtester implementation:
code/src/hkg_tmax_demo_trading/
apps/hkg-polymarket-backtester/
configs/hkg_tmax/probability_bucket_v1.yaml
migrations/postgres/20260706_0009_demo_trading_backtester.sql
code/tests/test_demo_trading_probability.py
code/tests/test_demo_trading_service.py
code/tests/test_demo_trading_api.py

Organized HKG Tmax experiment namespace:
experiments/hkg_tmax/README.md
experiments/hkg_tmax/EXPERIMENT_INDEX.md
experiments/hkg_tmax/0001_residual_ml_strategy_20260705/
experiments/hkg_tmax/0002_selective_no_harm_router_20260705/
experiments/hkg_tmax/0003_official_residual_memory_20260706/
experiments/hkg_tmax/0009_public_weather_backfill_jun25_jul7_lean_db_20260708/
experiments/hkg_tmax/0012_public_weather_backfill_optimized_pipeline_validation_20260709/
experiments/hkg_tmax/0012_public_weather_backfill_optimized_pipeline_validation_20260709/documentation/README.md
experiments/hkg_tmax/0012_public_weather_backfill_optimized_pipeline_validation_20260709/documentation/PUBLIC_WEATHER_BACKFILL_IMPLEMENTATION_AND_VALIDATION.md
experiments/hkg_tmax/0012_public_weather_backfill_optimized_pipeline_validation_20260709/documentation/LIVE_POSTGRES_MEASUREMENT_SNAPSHOT_20260710.md
experiments/hkg_tmax/0012_public_weather_backfill_optimized_pipeline_validation_20260709/documentation/POSTGRES_STORAGE_CAPACITY_ESTIMATE_2017_TO_2026.md

HKG Tmax residual-ML next-round implementation:
scripts/run_hkg_tmax_residual_ml_next_round.py
configs/hkg_tmax/residual_ml_next_round.yaml
code/src/hkg_tmax/features/pruned_feature_policy.py
code/src/hkg_tmax/modeling/selective_router.py
code/src/hkg_tmax/modeling/tail_specialist.py
code/src/hkg_tmax/evaluation/no_harm_reporting.py
code/src/hkg_tmax/data/anchor_provenance_audit.py
code/tests/test_hkg_tmax_next_round_selective_router.py

HKG Tmax residual-ML next-round evidence:
experiments/hkg_tmax/0002_selective_no_harm_router_20260705/results/
experiments/hkg_tmax_residual_ml_next_round/results/
experiments/hkg_tmax/0002_selective_no_harm_router_20260705/results/next_round_model_card.md
experiments/hkg_tmax/0002_selective_no_harm_router_20260705/results/scoreboard.csv
experiments/hkg_tmax/0002_selective_no_harm_router_20260705/results/no_harm_audit.json
experiments/hkg_tmax/0002_selective_no_harm_router_20260705/results/leakage_audit.json
experiments/hkg_tmax/0002_selective_no_harm_router_20260705/results/anchor_provenance_summary.json
experiments/hkg_tmax/0002_selective_no_harm_router_20260705/results/prediction_rows.parquet

HKG Tmax official residual-memory point-forecast implementation:
scripts/run_hkg_tmax_residual_ml_official_memory.py
configs/hkg_tmax/residual_ml_official_memory.yaml
code/src/hkg_tmax/data/official_residual_memory_features.py
code/src/hkg_tmax/features/residual_memory_policy.py
code/src/hkg_tmax/evaluation/official_residual_memory_runner.py
code/tests/test_hkg_tmax_official_residual_memory.py

HKG Tmax official residual-memory point-forecast evidence:
experiments/hkg_tmax/0003_official_residual_memory_20260706/results/
experiments/hkg_tmax_residual_ml_official_memory/results/
experiments/hkg_tmax/0003_official_residual_memory_20260706/inputs/gpt_pro_point_forecast_ml_strategy_deep_analysis_next_round_spec_20260706.txt
experiments/hkg_tmax/0003_official_residual_memory_20260706/results/scoreboard.csv
experiments/hkg_tmax/0003_official_residual_memory_20260706/results/scoreboard_by_split.csv
experiments/hkg_tmax/0003_official_residual_memory_20260706/results/scoreboard_by_residual_memory_bin.csv
experiments/hkg_tmax/0003_official_residual_memory_20260706/results/model_card.md
experiments/hkg_tmax/0003_official_residual_memory_20260706/results/leakage_audit.json
experiments/hkg_tmax/0003_official_residual_memory_20260706/results/residual_memory_publication_safety_audit.json
experiments/hkg_tmax/0003_official_residual_memory_20260706/results/row_identity_gate.json
experiments/hkg_tmax/0003_official_residual_memory_20260706/results/model_selection_log.json
experiments/hkg_tmax/0003_official_residual_memory_20260706/results/prediction_rows.parquet

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

HKG-T24-001 supporting files:

```text
config/hkg_t24/hkg_t24_001_foundation.yaml
sql/hkg_t24/hkg_t24_001_foundation_schema.sql
sql/hkg_t24/hkg_t24_001_gribstream_safe_rows.sql
schemas/hkg_t24/hkg_t24_001_schema_versions.json
code/tests/hkg_t24/test_database_url_priority.py
code/tests/hkg_t24/test_h24n_contract_policy.py
code/tests/hkg_t24/test_snapshot_builder_synthetic.py
code/tests/hkg_t24/test_schema_sql_contract.py
code/tests/hkg_t24/test_real_db_contracts.py
```

HKG-T24-003 supporting files:

```text
code/tests/hkg_t24/test_jira003_router_specialists_distribution.py
reports/jira_003_contract_coverage.md
reports/router_scoreboard_strict.csv
reports/router_weight_diagnostics.csv
reports/router_promotion_decisions.csv
reports/specialist_scoreboard_strict.csv
reports/specialist_activation_report.csv
reports/specialist_no_harm_report.csv
reports/specialist_promotion_decisions.csv
reports/distribution_scoreboard.csv
reports/distribution_calibration_report.csv
reports/distribution_calibration_report.md
reports/calibration_report.md
reports/threshold_probability_scoreboard.csv
reports/prediction_interval_coverage_report.csv
reports/system_scoreboard_strict.csv
reports/system_scoreboard_proxy.csv
reports/system_ablation_matrix.csv
reports/system_slice_scoreboard.csv
reports/system_replay_report.md
documentation/strategy_implementation_documentation/context/HKG_T24_003_00_READ_STATE_AND_CONTRACT_TRACE.md
documentation/strategy_implementation_documentation/context/HKG_T24_003_01_SCHEMA_AND_CLI_IMPLEMENTATION_LOG.md
documentation/strategy_implementation_documentation/context/HKG_T24_003_02_ROUTER_IMPLEMENTATION_LOG.md
documentation/strategy_implementation_documentation/context/HKG_T24_003_03_SPECIALIST_IMPLEMENTATION_LOG.md
documentation/strategy_implementation_documentation/context/HKG_T24_003_04_DISTRIBUTION_AND_FINAL_FORMULA_LOG.md
documentation/strategy_implementation_documentation/context/HKG_T24_003_05_TEST_AND_SMOKE_VERIFICATION_LOG.md
documentation/strategy_implementation_documentation/context/HKG_T24_003_06_FINAL_HANDOFF.md
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
