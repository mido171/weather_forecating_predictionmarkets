# KLGA Tmax Implementation Agent Instructions

These instructions apply to every file under `bootstrap/klga_tmax`.

## Mission

Implement the KLGA Tmax Polymarket strategy as a leakage-safe, reproducible, task-by-task Python/Postgres project. The project is intentionally split into numbered acquisition and foundation tasks so each Codex conversation can take one task, implement it fully, verify it, document it, and leave the next task easy to assign.

## Project Layout

- Strategy source documents live in `strategy_spec`.
- Data acquisition task specs live in `strategy_spec/data_aquisition/<numbered_task_folder>`.
- Cross-task context and implementation handoff documents live in `strategy_spec/context`.
- All executable code, tests, Alembic migrations, and CLI code live in `implementation`.
- Do not add source code or tests outside `implementation`.
- Do not put provider credentials in this tree.

## Canonical Implementation Root

Run implementation commands from:

```powershell
Set-Location C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\klga_tmax\implementation
```

The package is a Python 3.11 project with:

- `src/klga_tmax` for implementation code.
- `tests` for tests.
- `alembic/versions` for schema migrations.
- `python -m klga_tmax.cli` as the local module CLI.
- `klga-tmax` as the installed console entry point when the package is installed.

## Database Contract

Use `KLGA_DB_URL` as the only canonical application DSN environment variable.

Local verified DSN:

```powershell
$env:KLGA_DB_URL = "postgresql+psycopg://postgres:root@127.0.0.1:5432/klga_tmax_research"
```

Expected local DB:

```text
host: 127.0.0.1
port: 5432
database: klga_tmax_research
user: postgres
password: root
```

Do not introduce `KLGA_TMAX_DATABASE_URL` or a second DB variable.

## Required Read Order For Each New Task

Before implementing any numbered task, read these files in order:

1. `AGENTS.md`
2. `strategy_spec/KLGA_strategy_spec/KLGA_TMAX_TRADING_STRATEGY_SPEC.md`
3. `strategy_spec/KLGA_strategy_spec/supplemental_doc_1.md`
4. `strategy_spec/KLGA_strategy_spec/supplemental_doc_1_patch_1.md`
5. `strategy_spec/context/KLGA_TMAX_POSTGRES_PERSISTENCE_CONTEXT.md`
6. `strategy_spec/context/KLGA_TMAX_00_FOUNDATION_IMPLEMENTATION_DEEP_DIVE.md`
7. `strategy_spec/context/KLGA_TMAX_01_STATION_UNIVERSE_IMPLEMENTATION_DEEP_DIVE.md`
8. `strategy_spec/context/KLGA_TMAX_TASK_IMPLEMENTATION_QUEUE.md`
9. The exact task spec under `strategy_spec/data_aquisition/<numbered_task_folder>`.

If a task has already been implemented, also read its deep-dive document before modifying it.

## Current Completed Foundation

Task 00 foundation is implemented. It provides the shared Postgres schemas, Alembic machinery, source request/record contracts, availability ledger, target instances, feature values, feature matrix, cutoff logic, leakage checks, audit runs, and foundation validation.

Task 01 station universe is implemented. `registry.station_registry` is canonical, and `registry.stations` is a compatibility projection for existing Task 00 foreign keys.

Provider fetchers must use:

- `src/klga_tmax/registry/station_universe.py`
- `registry.station_registry`
- `STATION_REGISTRY_VERSION = "v2026_06_27_klga_core"`

Do not copy station lists, pseudo-points, provider station IDs, or coordinate tiers into provider modules.

## Task Implementation Workflow

For each numbered task:

1. Identify the task folder and source spec from `KLGA_TMAX_TASK_IMPLEMENTATION_QUEUE.md`.
2. Read all required governing docs and the task spec in full.
3. Inspect existing code, migrations, CLI commands, and tests before editing.
4. Implement only the task's requested surface.
5. Add an Alembic migration when persistence changes.
6. Add or update source modules under `src/klga_tmax`.
7. Add or update tests under `tests`.
8. Add a CLI command for user-facing task operations when the task creates an operation the user will run.
9. Add a validation command when the task creates persistent data or a contract that must be checked.
10. Update `db inspect-contract` if the task adds required schema objects.
11. Create a task deep-dive document in `strategy_spec/context`.
12. Update the task queue status if the task is completed.
13. Run the verification commands listed below.

## Migration Rules

- Alembic revision strings must be 32 characters or fewer because `alembic_version.version_num` is length-limited.
- Migration filenames may be descriptive, but the internal `revision` value must be short.
- Prefer additive migrations.
- Use explicit constraints and indexes for identity, availability, and lookup contracts.
- Use PostgreSQL JSONB for structured provider payload metadata.
- Use parameterized SQL for all runtime values.
- Never remove existing user data unless the user explicitly asks for a destructive cleanup.

## Leakage And Availability Rules

Every provider task must preserve Task 00's central rule:

```text
A feature is eligible for a forecast cutoff only if the system could have known it by that cutoff.
```

Provider ingestion must store:

- provider issue time when available
- provider valid time when applicable
- provider availability time when observed or conservatively inferred
- local ingestion time
- source request identity
- source record identity
- raw payload hash
- revision number when source data can change
- effective availability timestamp used by features

Never use run time, valid time, archive timestamp, or ingestion timestamp as a substitute for availability unless the provider task spec explicitly defines that rule.

## Provider Fetching Rules

- Implement clients with bounded retry and clear permanent vs temporary failures.
- Persist raw provider responses before normalized rows.
- Redact secrets from logs and audit rows.
- Do not hardcode credentials.
- Local GribStream credentials are kept outside this KLGA tree at:

```powershell
C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\.secrets\gribstream_api_token.txt
```

Load it for live GribStream commands with:

```powershell
$env:GRIBSTREAM_API_TOKEN = (Get-Content "C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\.secrets\gribstream_api_token.txt" -Raw).Trim()
```

The `.secrets/` directory is git-ignored. Do not print this token, copy it into docs, commit it, or include it in generated artifacts.
- Do not run large backfills unless the user explicitly asks for the live pull and confirms the date range or quota-sensitive shape.
- For GribStream tasks, use the local `gribstream-api` skill and live catalog/API discovery for exact selectors. Do not invent selectors.
- For current external API docs, use official provider documentation as source of truth.

## CLI And Exit Codes

Keep the existing CLI shape:

```text
python -m klga_tmax.cli db migrate
python -m klga_tmax.cli db inspect-contract
python -m klga_tmax.cli registry seed
python -m klga_tmax.cli registry materialize-targets
python -m klga_tmax.cli validate foundation
python -m klga_tmax.cli validate station-universe
```

Existing exit code contract:

- Missing required DB config: 10
- Migration failure: 20
- Validation failure: 30

New task commands should use the existing audit wrapper pattern where appropriate so `audit.pipeline_runs` records command name, args, status, exit code, row counts, and error text.

## Required Verification For Each Task

Run these non-DB checks from `implementation`:

```powershell
python -m compileall -q src tests
python -m pytest -q
python -m klga_tmax.cli --help
python -m klga_tmax.cli validate --help
```

Run these DB checks when the task touches schema, persistence, seed data, or validation:

```powershell
$env:KLGA_DB_URL = "postgresql+psycopg://postgres:root@127.0.0.1:5432/klga_tmax_research"
python -m klga_tmax.cli db migrate
python -m klga_tmax.cli db inspect-contract
python -m klga_tmax.cli validate foundation
```

Also run the task-specific validation command that the task adds.

If a command cannot be run because a tool is missing or a provider key is unavailable, document the exact reason, impact, and replacement verification used.

## Documentation Requirement

Every implemented task must create a deep-dive document:

```text
strategy_spec/context/KLGA_TMAX_<NN>_<TASK_SLUG>_IMPLEMENTATION_DEEP_DIVE.md
```

The document must cover:

- exact files changed
- schema/table/index definitions
- CLI commands and exit codes
- provider API assumptions
- availability and leakage rules
- migration and rollback notes
- verification commands and outputs
- known limitations
- next task handoff

Run the documentation quality gate when the `exceptional-code-document-writer` skill scripts are available.

## Assignment Files

Use these context files to make future task assignment repeatable:

- `strategy_spec/context/KLGA_TMAX_TASK_IMPLEMENTATION_QUEUE.md`
- `strategy_spec/context/KLGA_TMAX_TASK_ASSIGNMENT_TEMPLATE.md`
- `strategy_spec/context/KLGA_TMAX_TASK_HANDOFF_CHECKLIST.md`

## Git And Workspace Safety

The outer Git repo is `weather_data_extraction`, currently used from the `extraction-cleanup` branch. `bootstrap/klga_tmax` may appear as an untracked tree in that outer repo. Verify `git rev-parse --show-toplevel`, branch, and remote before making staging, commit, or push claims.

Do not revert unrelated dirty files in the HKG bootstrap tree or elsewhere in `weather_data_extraction`.
