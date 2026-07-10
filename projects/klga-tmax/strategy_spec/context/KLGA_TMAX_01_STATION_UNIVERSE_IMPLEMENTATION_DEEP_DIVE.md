# KLGA Task 01 Station Universe Implementation Deep Dive

## Executive Summary

Task 01 is implemented as the canonical station and coordinate registry for the KLGA Tmax strategy. The developer-facing result is a versioned `registry.station_registry` table, a central Python module with the exact station, pseudo-point, group, and coordinate-tier contracts from the Task 01 spec, seed code that writes both the canonical table and the existing Task 00 compatibility table, a new `klga-tmax validate station-universe` command, and unit/schema tests that enforce the contract.

The highest-impact implementation file is `bootstrap/klga_tmax/implementation/src/klga_tmax/registry/station_universe.py`, because future IEM, Wunderground, MOS, GribStream, and Open-Meteo fetchers should import its constants and lookup functions instead of creating local station lists. The highest-impact persistence file is `bootstrap/klga_tmax/implementation/alembic/versions/0002_klga_station_universe_registry.py`, because it creates `registry.station_registry` and adjusts `registry.stations.station_role` to accept the Task 01 role names.

The primary architectural decision is to keep `registry.stations` alive for existing Task 00 foreign keys while treating `registry.station_registry` as the canonical Task 01 source of truth. `registry.stations` is now a compatibility projection that only keeps the 29 Task 01 rows active. Rows from the earlier supplemental pseudo-point defaults are not deleted; they are marked inactive so existing references are not erased.

Verification status is passing. The implementation root compiled, `pytest` reported `28 passed`, the CLI exposes the new validation command, Alembic migrated to `0002_station_universe`, contract inspection passed against Postgres, `validate foundation` still passed, `validate station-universe` passed, and the documentation quality gate passed with the explicit Task 01 changed-file list.

## Reader Orientation and Document Map

Read this document if you are implementing the next KLGA provider fetcher or reviewing whether Task 01 is ready for downstream data acquisition. The first files to inspect are:

- `bootstrap/klga_tmax/implementation/src/klga_tmax/registry/station_universe.py` for station IDs, provider IDs, groups, and coordinate tiers.
- `bootstrap/klga_tmax/implementation/src/klga_tmax/registry/seed_stations.py` for how canonical rows are persisted.
- `bootstrap/klga_tmax/implementation/src/klga_tmax/validation/station_universe.py` for the acceptance checks that define a compliant database.
- `bootstrap/klga_tmax/implementation/alembic/versions/0002_klga_station_universe_registry.py` for the table, role constraint, indexes, and rollback path.

Document sections:

- Scope Boundaries names what this task includes and what it deliberately excludes.
- Source-of-Truth Inputs lists the spec, code, commands, and database evidence used here.
- Requirements-to-Implementation Traceability maps every plan requirement to code and verification.
- Change Inventory gives one row per Task 01 changed file.
- Architecture and Control Flow explains how CLI calls, seeding, validation, and schema inspection fit together.
- File-by-File Deep Dive documents exact symbols, side effects, and maintenance rules for each changed file.
- Public Interfaces and Contracts records the CLI, Python module, and database contracts created by this task.
- Testing and Verification Evidence records exact commands, directories, results, and proof boundaries.

## Scope Boundaries

Included in Task 01:

- A versioned canonical station table named `registry.station_registry`.
- A central in-code station universe with 19 airport/station rows and 10 gridded pseudo-point rows from `10_station_universe_and_coordinates.md`.
- Exact station group constants named `TARGET_STATION`, `NYC_CORE_STATIONS`, `COASTAL_MARINE_STATIONS`, `INLAND_HOT_REFERENCE_STATIONS`, `UPSTREAM_SOUTHWEST_STATIONS`, `BACKDOOR_FRONT_STATIONS`, and `LONG_ISLAND_SOUND_STATIONS`.
- Exact Tier A and Tier B point lists, plus deterministic Tier C generation for the 25-point KLGA research grid.
- Seed behavior that writes `registry.station_registry` and refreshes `registry.stations` as a compatibility projection.
- Contract inspection and validation coverage for the new table and canonical row counts.
- Unit and schema tests for the Task 01 constants, migration declaration, CLI wiring, and negative validation path.

Excluded from Task 01:

- No GribStream, IEM, Wunderground, Open-Meteo, MOS, or Polymarket data is fetched.
- No provider API credentials are added or read.
- No feature builder consumes the station groups yet.
- No Tier C backfill is launched. Tier C is generated and tested, but it is not a default fetch target.
- No station-elevation source is introduced. `elevation_m` is supported by schema and dataclass but remains null for the seeded Task 01 rows because the spec did not provide exact elevation values.

## Source-of-Truth Inputs

The implementation and this document used these inputs:

- User implementation plan for `01_station_universe_and_coordinates`.
- Task spec file `bootstrap/klga_tmax/strategy_spec/data_aquisition/01_station_universe_and_coordinates/10_station_universe_and_coordinates.md`.
- Existing Task 00 project under `bootstrap/klga_tmax/implementation`.
- Final contents of the changed implementation files listed in the Change Inventory.
- Current Git state from `git status --short --branch` in `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction`.
- Command evidence from compile, pytest, CLI help, Alembic migration, registry seeding, contract inspection, station-universe validation, and direct SQL readback.
- Documentation quality gate script at `C:\Users\ahmad\.codex\skills\exceptional-code-document-writer\scripts\documentation_quality_gate.py`.

Important evidence caveat: `bootstrap/klga_tmax/` is an untracked project tree in the outer `weather_data_extraction` Git repository. Because of that, normal `git diff --stat` and `git diff --name-status` do not isolate Task 01 hunks. The documentation quality gate was run with an explicit changed-file list for Task 01 instead of relying on outer-repo automatic discovery.

## Requirements-to-Implementation Traceability

| Requirement | Implementation location | Behavior delivered | Verification evidence | Caveat |
|---|---|---|---|---|
| Add Alembic migration `0002_klga_station_universe_registry.py`. | `bootstrap/klga_tmax/implementation/alembic/versions/0002_klga_station_universe_registry.py` | Creates `registry.station_registry`, role constraints, and indexes. | `python -m klga_tmax.cli db migrate` passed; DB readback shows `alembic_version=0002_station_universe`. | File name uses requested `0002_klga_station_universe_registry.py`; internal Alembic revision is shortened to fit Alembic's 32-character version column. |
| Use `station_registry_version = "v2026_06_27_klga_core"`. | `station_universe.py`, `seed_stations.py`, `migrations_check.py`, `validation/station_universe.py` | Version is the Python constant and the seed/validation filter. | `validate station-universe` details include `station_registry_version`. SQL readback shows 29 rows for that version. | No second registry version exists yet. |
| Store exact fields from the task spec. | Migration and `StationRegistry` ORM class. | Columns include provider IDs, `grid_point_id`, role, lat/lon, elevation, native metadata JSON, active dates, and notes. | Schema test reads migration text; `db inspect-contract` checks required columns. | `created_at` is an additional audit column. |
| Use `grid_point_id TEXT NOT NULL DEFAULT ''` and a real primary key. | Migration and ORM class. | Primary key is `(station_registry_version, station_id, grid_point_id)`. Airport rows use blank `grid_point_id`; pseudo-points use their grid ID. | `test_station_registry_migration_declares_versioned_registry_table` asserts the default and primary key text. | This avoids an expression primary key in PostgreSQL. |
| Update compatibility station roles. | Migration and `Station` ORM constraint. | `registry.stations.station_role` accepts `target`, `nearby_core`, `regional_context`, and `gridded_pseudo_point`. Old role names are converted during upgrade. | Compile, migration, schema test, and DB validation passed. | Downgrade maps `regional_context` and `nearby_core` back to `nearby`, which loses role granularity. |
| Add central station-universe module. | `station_universe.py` | Defines dataclass, 19 station rows, 10 pseudo-point rows, groups, tiers, and lookup functions. | `test_station_universe.py` validates counts, KLGA IDs, group maps, Tier A/B order, and Tier C. | Module is pure Python and does not query the DB. |
| Seed canonical table and compatibility projection. | `seed_stations.py`, `seed.py` | `seed_all` writes cutoffs, canonical station registry, compatibility stations, and feature version. | `db migrate` and `registry seed` row counts show 29 station registry rows and 29 active station rows. | Five older compatibility rows remain inactive in local DB. |
| Include `registry.station_registry` in contract inspection. | `migrations_check.py` | Required table, columns, index, and row count are checked. | `db inspect-contract` passed with `tables_checked=18`, `indexes_checked=5`, `station_registry_rows=29`. | Inspection requires seeded rows, not only DDL. |
| Add `klga-tmax validate station-universe`. | `cli.py`, `validation/station_universe.py` | New Typer command runs constant checks and DB row checks and exits through the shared validation error path. | `python -m klga_tmax.cli validate --help` lists `station-universe`; CLI wiring test verifies exit code 30 path. | DB-touching command still requires `KLGA_DB_URL`. |
| Keep `validate foundation` working. | `migrations_check.py`, `seed_stations.py`, existing foundation validation. | Foundation inspection now counts active compatibility station rows and includes station registry count. | `python -m klga_tmax.cli validate foundation` passed. | Existing `gold.target_instances` count came from the prior Task 00 local DB state. |
| Document implementation. | `bootstrap/klga_tmax/strategy_spec/context/KLGA_TMAX_01_STATION_UNIVERSE_IMPLEMENTATION_DEEP_DIVE.md` | This document records files, schema, seed behavior, contracts, tests, limitations, and handoff guidance. | Quality gate passed after this file was written. | Documentation uses explicit changed-file scope because outer Git has `bootstrap/klga_tmax` untracked. |

## Change Inventory

| File path | Change type | Why it changed | Main symbols or objects | Effect | Verification coverage |
|---|---|---|---|---|---|
| `bootstrap/klga_tmax/implementation/alembic/versions/0002_klga_station_universe_registry.py` | Added migration, schema | Adds Task 01 persistence and role compatibility migration. | `upgrade`, `downgrade`, `registry.station_registry`, role check constraints, indexes. | DB can store versioned station registry rows and keep Task 00 FKs. | Migration command, schema text test, contract inspection. |
| `bootstrap/klga_tmax/implementation/src/klga_tmax/registry/station_universe.py` | Added domain module | Centralizes the station universe and coordinate tier contract. | `StationRegistryEntry`, `CANONICAL_STATION_REGISTRY`, `STATION_GROUPS`, `coordinate_tier`, `tier_c_points`. | Fetchers can import one source for station and coordinate lists. | Unit tests for counts, groups, tiers, provider IDs. |
| `bootstrap/klga_tmax/implementation/src/klga_tmax/registry/seed_stations.py` | Modified seed module | Writes canonical Task 01 rows and compatibility projection. | `seed_station_registry`, `seed_stations`, `_compatibility_groups`, `ALL_STATION_SEEDS`. | DB seed produces 29 canonical rows and 29 active compatibility rows. | DB migrate, registry seed, contract inspection, validation command. |
| `bootstrap/klga_tmax/implementation/src/klga_tmax/registry/seed.py` | Modified seed orchestrator | Adds canonical station registry seeding to `seed_all`. | `seed_all`, import of `seed_station_registry`. | `db migrate` and `registry seed` include `registry.station_registry` row counts. | CLI DB commands passed. |
| `bootstrap/klga_tmax/implementation/src/klga_tmax/db/models.py` | Modified ORM model file | Mirrors Task 01 role constraint and new table. | `Station` role check, `StationRegistry` class. | ORM metadata matches the migration for future DB use. | Compile and migration/schema tests. |
| `bootstrap/klga_tmax/implementation/src/klga_tmax/db/migrations_check.py` | Modified contract inspector | Makes station registry part of the foundation contract. | `REQUIRED_TABLES`, `REQUIRED_COLUMNS`, `REQUIRED_INDEXES`, `inspect_contract`. | `db inspect-contract` fails if the new table, columns, index, or row count are missing. | Contract command and foundation validation passed. |
| `bootstrap/klga_tmax/implementation/src/klga_tmax/validation/station_universe.py` | Added validation module | Implements Task 01 positive and negative acceptance checks. | `validate_station_universe`, `_validate_constant_contract`, `_validate_database_rows`. | A seeded DB can be checked for exact constants and persisted rows. | Unit negative test and live DB validation command. |
| `bootstrap/klga_tmax/implementation/src/klga_tmax/cli.py` | Modified CLI entry point | Exposes station-universe validation command. | `validate_station_universe_command`, import of `validate_station_universe`. | Users can run `python -m klga_tmax.cli validate station-universe`. | CLI help and CLI wiring test. |
| `bootstrap/klga_tmax/implementation/tests/test_station_universe.py` | Added tests | Covers the Task 01 domain contract and validation failure path. | Seven tests covering constants, KLGA IDs, tiers, groups, DB failure, CLI exit code. | Regressions in station constants or CLI validation wiring fail fast. | `pytest` passed. |
| `bootstrap/klga_tmax/implementation/tests/test_schema_contract.py` | Modified tests | Extends schema contract coverage for station registry. | `test_required_task00_tables_are_in_contract_list`, `test_station_registry_migration_declares_versioned_registry_table`. | Missing table declaration or bad `grid_point_id` primary-key design fails tests. | `pytest` passed. |
| `bootstrap/klga_tmax/strategy_spec/context/KLGA_TMAX_01_STATION_UNIVERSE_IMPLEMENTATION_DEEP_DIVE.md` | Added documentation | Records implementation, contracts, evidence, and handoff guidance. | This document. | Future provider fetchers have a Task 01 engineering handoff. | Documentation quality gate passed. |

## Architecture and Control Flow

Task 01 places the station universe in one domain module, then projects that module into the database through seed code and validates both the constants and the DB state through CLI commands.

```mermaid
flowchart TD
    A["Task 01 spec: 10_station_universe_and_coordinates.md"] --> B["station_universe.py constants"]
    B --> C["seed_stations.py"]
    C --> D["registry.station_registry canonical rows"]
    C --> E["registry.stations active compatibility projection"]
    D --> F["validation/station_universe.py"]
    E --> F
    F --> G["klga-tmax validate station-universe"]
    D --> H["db inspect-contract"]
    E --> H
```

The data flow is deterministic:

1. `station_universe.py` defines immutable `StationRegistryEntry` objects.
2. `seed_all` calls `seed_station_registry` before `seed_stations`.
3. `seed_station_registry` inserts or updates all 29 canonical rows using `ON CONFLICT`.
4. `seed_stations` inserts or updates the same 29 rows into `registry.stations`, then marks non-canonical active rows inactive.
5. `inspect_contract` checks that the canonical version has exactly 29 rows and that `registry.stations` has exactly 29 active rows.
6. `validate_station_universe` checks the in-code constants and compares each canonical DB row to its expected role, provider IDs, lat/lon, and active compatibility row.

The failure path is also deterministic. Missing `KLGA_DB_URL` exits through the existing configuration path with exit code 10. Migration exceptions exit 20. Validation failures, including missing station registry rows, exit 30 through `_run_audited`.

## File-by-File Deep Dive

### `bootstrap/klga_tmax/implementation/alembic/versions/0002_klga_station_universe_registry.py`

This migration is the persisted schema boundary for Task 01. `upgrade()` first drops the previous `ck_stations_role`, converts older role values to Task 01 role names, and recreates the compatibility role check for `target`, `nearby_core`, `regional_context`, and `gridded_pseudo_point`.

The migration then creates `registry.station_registry` with the Task 01 fields. The primary key is `(station_registry_version, station_id, grid_point_id)`. `grid_point_id` is `text NOT NULL DEFAULT ''`, which lets airport stations use a blank grid point while PostgreSQL still enforces a normal primary key. This follows the plan's instruction to avoid an expression primary key.

Indexes:

- `ix_station_registry_role` supports version-and-role reads used by inspectors and future feature builders.
- `ix_station_registry_grid_point` supports pseudo-point lookups and ignores blank airport grid IDs.
- `ix_station_registry_iem_asos_id` supports IEM station mapping.
- `ix_station_registry_mos_station_id` supports MOS station mapping.

The internal Alembic revision is `0002_station_universe`. The requested file name is preserved, but the revision string had to be shortened because Alembic stores version identifiers in a `varchar(32)` column. Using the longer file stem as the revision produced a PostgreSQL length error during the first migration attempt.

`downgrade()` drops `registry.station_registry`, restores the older compatibility role check, and maps Task 01 roles back to the Task 00 values. That rollback preserves table validity, but it cannot preserve the distinction between `nearby_core` and `regional_context` because both map back to `nearby`.

### `bootstrap/klga_tmax/implementation/src/klga_tmax/registry/station_universe.py`

This is the canonical Task 01 domain module. It has no DB dependency and no network dependency. That makes it safe for provider clients, feature builders, tests, and validation code to import.

`StationRegistryEntry` is a frozen dataclass with the fields needed by the table seed: `station_id`, `role`, `lat`, `lon`, optional provider IDs, `grid_point_id`, optional `elevation_m`, `source_native_metadata_json`, active dates, and notes. Its `is_pseudo_point` property checks `role == "gridded_pseudo_point"`.

`MANDATORY_STATION_REGISTRY` contains the 19 station rows from the spec. `MANDATORY_PSEUDO_POINT_REGISTRY` contains the 10 GribStream/Open-Meteo pseudo-points. `CANONICAL_STATION_REGISTRY` is the 29-row concatenation used by seeds and validators.

The group constants are exact tuples, not mutable lists. That makes accidental in-process mutation harder and gives tests stable expected values. The module also exposes `STATION_GROUPS` so callers can look up a named group without importing every constant.

The coordinate tiers are split by cost and usage:

- `TIER_A_POINT_IDS` is the 4-point minimum viable pull.
- `TIER_B_POINT_IDS` is the 10-point recommended production pull.
- `tier_c_points()` generates the 25-point research grid with offsets `-0.10`, `-0.05`, `0.00`, `+0.05`, and `+0.10` around KLGA.

Lookup functions raise `KeyError` on unknown station IDs, grid point IDs, providers, groups, or tiers. That fail-fast behavior is intentional: a fetcher using an unsupported provider name should stop at configuration time instead of silently using a wrong ID.

### `bootstrap/klga_tmax/implementation/src/klga_tmax/registry/seed_stations.py`

This file is the persistence adapter from the pure station universe module to Postgres. It imports `CANONICAL_STATION_REGISTRY`, `STATION_GROUPS`, and `STATION_REGISTRY_VERSION`.

`seed_station_registry(connection)` writes the canonical Task 01 table. It iterates over all 29 canonical entries and inserts each row into `registry.station_registry`. The conflict target is the table primary key, so repeated seeding updates changed metadata without duplicating rows. `source_native_metadata_json` is serialized with `json.dumps(..., sort_keys=True)` and cast as JSONB in SQL. This was required because passing a raw Python dictionary into this text query caused psycopg adaptation failure.

`seed_stations(connection)` writes the compatibility projection to `registry.stations`. It derives a display name for airport rows from `_STATION_NAMES`; pseudo-points use a generated name based on `grid_point_id`. The compatibility `provider_primary_id` chooses Wunderground, then IEM, then MOS, then grid point, then station ID. The compatibility `station_group` array contains the role and every named station group that includes the row's `station_id`.

After the 29 upserts, `seed_stations` marks stale active rows inactive with:

```sql
UPDATE registry.stations
SET active = false
WHERE NOT (station_id = ANY(:active_station_ids))
  AND active = true
```

This is why the local DB now has 29 active rows and 5 inactive compatibility rows. The inactive rows are earlier supplemental pseudo-point defaults and are retained instead of deleted.

### `bootstrap/klga_tmax/implementation/src/klga_tmax/registry/seed.py`

`seed.py` orchestrates seed order. It now imports `seed_station_registry` and returns a row-count dictionary with four keys: `registry.cutoffs`, `registry.station_registry`, `registry.stations`, and `registry.feature_versions`.

The order matters. The canonical station registry is seeded before the compatibility projection so both tables come from the same `CANONICAL_STATION_REGISTRY` constant set during one command execution. `db migrate` and `registry seed` print these row counts, which makes a wrong seed count visible in the CLI output.

### `bootstrap/klga_tmax/implementation/src/klga_tmax/db/models.py`

The existing `Station` ORM model now mirrors the Task 01 role names in `ck_stations_role`. This keeps ORM metadata aligned with the migration and prevents future SQLAlchemy-based code from thinking the old `nearby`, `pseudo_point`, or `external_context` values remain valid.

The new `StationRegistry` ORM class maps the canonical table. It uses a three-column primary key, JSONB metadata, date columns for active windows, lat/lon check constraints, and the same role check as the migration. It is not currently used for ORM persistence by the seed path, which uses SQL text for explicit upsert control, but the class gives future code a typed metadata representation of the table.

### `bootstrap/klga_tmax/implementation/src/klga_tmax/db/migrations_check.py`

The contract inspector now treats `registry.station_registry` as a required registry table. `REQUIRED_TABLES` includes it under `registry`, `REQUIRED_COLUMNS` names the canonical columns that must exist, and `REQUIRED_INDEXES` includes `ix_station_registry_role`.

`inspect_contract(connection)` now computes:

- `station_count` from `registry.stations WHERE active = true`.
- `station_registry_count` from `registry.station_registry WHERE station_registry_version = :station_registry_version`.

Both counts must equal `len(CANONICAL_STATION_REGISTRY)`, which is 29. This keeps Task 00 compatibility active rows in lockstep with the Task 01 source of truth without requiring deletion of inactive historical compatibility rows.

### `bootstrap/klga_tmax/implementation/src/klga_tmax/validation/station_universe.py`

This module is the formal Task 01 validator. It returns the existing `ContractInspection` type so the CLI can print the same JSON structure as `db inspect-contract` and `validate foundation`.

`_validate_constant_contract(result)` checks the in-code registry before touching the DB. It verifies row counts, duplicate `station_id` and `grid_point_id` values, KLGA provider IDs, ordered Tier A and Tier B point IDs, Tier C count and base coordinate, station group values, and duplicate provider IDs among the 19 airport station rows.

`_validate_database_rows(connection, result)` selects all rows for `STATION_REGISTRY_VERSION` from `registry.station_registry`, indexes them by `(version, station_id, grid_point_id)`, and compares every canonical row's role, provider IDs, latitude, and longitude. It then reads `registry.stations` for the canonical station IDs and verifies those compatibility rows are active with matching role and coordinates.

The validator appends string failures instead of raising immediately. This lets the CLI print all mismatches in one JSON payload before exiting 30.

### `bootstrap/klga_tmax/implementation/src/klga_tmax/cli.py`

The CLI imports `validate_station_universe` and exposes it as `validate station-universe`. The command uses `_run_audited` with:

- `command_name="validate station-universe"`
- `command_args={}`
- `failure_exit_code=EXIT_VALIDATION_ERROR`

That gives the new command the same audit behavior as the existing validation commands: it starts an `audit.pipeline_runs` row, executes validation inside a DB connection, records success or failure, prints JSON, and exits 30 on validation failure.

### `bootstrap/klga_tmax/implementation/tests/test_station_universe.py`

This test file covers the Task 01 domain and command boundary. It asserts the exact 19 station and 10 pseudo-point counts, KLGA provider IDs, Tier A/B ordered IDs, deterministic Tier C count and base coordinate, and exact station group mapping.

The negative DB test uses `_EmptyStationConnection`, whose `execute` method returns no rows. That forces `validate_station_universe` to report missing canonical DB rows without requiring a real Postgres fixture. The CLI wiring test monkeypatches `_run_audited` and asserts the command name and exit code for `validate station-universe`.

### `bootstrap/klga_tmax/implementation/tests/test_schema_contract.py`

This file extends the existing schema contract tests. `test_required_task00_tables_are_in_contract_list` now requires `registry.station_registry` in the contract list. `test_station_registry_migration_declares_versioned_registry_table` reads the migration file and asserts the table creation, version column, `grid_point_id text NOT NULL DEFAULT ''`, primary key, and Task 01 role names.

The test reads migration text rather than requiring a live DB, which keeps the default `pytest` suite fast and deterministic while still catching accidental migration edits.

### `bootstrap/klga_tmax/strategy_spec/context/KLGA_TMAX_01_STATION_UNIVERSE_IMPLEMENTATION_DEEP_DIVE.md`

This file is the Task 01 implementation handoff. It records the final behavior, file inventory, schema, seed behavior, public contracts, verification evidence, rollback notes, and known limitations. The documentation quality gate checks that this file names the explicit Task 01 changed-file list, includes required sections, contains verification commands, and avoids repeated paragraphs.

## Public Interfaces and Contracts

CLI contracts:

- `python -m klga_tmax.cli validate station-universe`
- Requires `KLGA_DB_URL` for DB access.
- Prints a JSON payload with `ok`, `details`, `failures`, and `warnings`.
- Exits 0 when the station universe constants and DB rows match.
- Exits 30 when validation fails.

Python contracts:

- `STATION_REGISTRY_VERSION = "v2026_06_27_klga_core"`.
- `CANONICAL_STATION_REGISTRY` contains exactly 29 rows.
- `MANDATORY_STATION_REGISTRY` contains exactly 19 airport/station rows.
- `MANDATORY_PSEUDO_POINT_REGISTRY` contains exactly 10 pseudo-point rows.
- `coordinate_tier("A")` returns 4 points in spec order.
- `coordinate_tier("B")` returns 10 points in spec order.
- `coordinate_tier("C")` returns 25 generated points.
- `provider_station_id("KLGA", "iem_asos") == "LGA"`.
- `provider_station_id("KLGA", "wunderground") == "KLGA"`.
- `provider_station_id("KLGA", "mos") == "LGA"`.

Database contracts:

- Canonical table: `registry.station_registry`.
- Canonical version: `v2026_06_27_klga_core`.
- Active compatibility table: `registry.stations`.
- Compatibility row count rule: exactly 29 active rows.
- Canonical row count rule: exactly 29 rows for the current station registry version.
- Stale compatibility rows may exist with `active=false`.

## Data Model, Persistence, and Migration Notes

`registry.station_registry` columns:

- `station_registry_version text NOT NULL`
- `station_id text NOT NULL`
- `iem_asos_id text`
- `wunderground_station_id text`
- `mos_station_id text`
- `grid_point_id text NOT NULL DEFAULT ''`
- `role text NOT NULL`
- `lat double precision NOT NULL`
- `lon double precision NOT NULL`
- `elevation_m double precision`
- `source_native_metadata_json jsonb NOT NULL DEFAULT '{}'::jsonb`
- `active_from_date date NOT NULL DEFAULT '1900-01-01'`
- `active_until_date date`
- `notes text`
- `created_at timestamptz NOT NULL DEFAULT now()`

Constraints:

- Primary key on `(station_registry_version, station_id, grid_point_id)`.
- Role check on `target`, `nearby_core`, `regional_context`, and `gridded_pseudo_point`.
- Latitude range check between -90 and 90.
- Longitude range check between -180 and 180.
- Active-until date must be null or not earlier than active-from date.

Migration order:

1. Apply Task 00 base schema if not already present.
2. Apply Task 01 migration `0002_station_universe`.
3. Seed cutoffs, station registry, compatibility stations, and feature version through `seed_all`.
4. Run `db inspect-contract`.
5. Run `validate station-universe`.

Rollback:

- Alembic downgrade from Task 01 drops `registry.station_registry`.
- The downgrade restores old role values in `registry.stations`.
- Role downgrade loses the difference between `nearby_core` and `regional_context`.
- Do not downgrade a DB that already has future fetcher rows depending on `registry.station_registry` without first archiving or migrating those dependencies.

## Error Handling, Edge Cases, and Failure Modes

Missing DB configuration still follows the existing Task 00 behavior. Any DB-touching command calls `load_settings(require_db=True)` and exits 10 when `KLGA_DB_URL` is missing or invalid.

Migration failures exit 20 from `db migrate`. During implementation, two migration-path defects were found and fixed: the initial revision ID exceeded Alembic's 32-character storage column, and JSONB seeding failed when a raw Python dictionary was passed to a text query. The final code uses `revision = "0002_station_universe"` and JSON text cast to `jsonb`.

Validation failures exit 30. The station-universe validator accumulates mismatches so a bad DB returns a full list of missing or incorrect rows instead of stopping after the first row.

Unknown providers, station IDs, grid point IDs, groups, or coordinate tiers raise `KeyError` in the domain module. That behavior is appropriate for fetcher setup and test code because the station universe should be explicit. Future CLI wrappers can translate those exceptions if they expose user-entered station IDs.

Existing inactive compatibility rows are allowed. `inspect_contract` counts only active `registry.stations` rows, and `validate_station_universe` checks only canonical IDs for compatibility. This prevents local historical rows from breaking Task 01 while still requiring active rows to match the canonical registry.

## Security, Privacy, and Safety Review

No secrets are introduced. The implementation reads the existing `KLGA_DB_URL` environment variable for DB commands and does not add provider API keys.

All SQL statements use SQLAlchemy `text()` parameters for runtime values such as station IDs, version strings, arrays, JSON text, and feature metadata. The migration DDL is static SQL authored in the repository.

The seed path writes provider station IDs and coordinates only. No personal data, API token, market position, or trade decision is stored by Task 01.

The command audit path records command names, arguments, row counts, and errors in `audit.pipeline_runs`. The new validation command passes an empty `command_args` dictionary, so it does not log secrets.

## Performance, Scalability, and Concurrency

Task 01 is small by design. The canonical registry contains 29 persisted rows and Tier C generation creates 25 in-memory rows on demand. The seed path performs one upsert per canonical row plus one stale-row update. That is acceptable for a registry command run during setup and deployment.

The indexes added to `registry.station_registry` cover expected lookup patterns: version-plus-role, version-plus-grid-point, version-plus-IEM ID, and version-plus-MOS ID. Wunderground lookups can use the primary station ID for the current airport rows because the Wunderground ID matches the four-character station ID for the mandatory stations.

No concurrency mechanism was added. Repeated `db migrate` and `registry seed` calls are idempotent because upserts use primary keys and stale compatibility rows are updated by active status. If two seed commands run at the exact same time, PostgreSQL primary-key enforcement protects duplicates, but Task 01 does not add an advisory lock.

## Configuration and Environment

The existing canonical DB environment variable remains `KLGA_DB_URL`. The verified local value was:

```powershell
$env:KLGA_DB_URL='postgresql+psycopg://postgres:root@127.0.0.1:5432/klga_tmax_research'
```

No new package dependency, config file, feature flag, or credential file was added.

The implementation root for commands is:

```text
C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\klga_tmax\implementation
```

## Testing and Verification Evidence

### Verification: `python -m compileall -q src tests`

- Directory: `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\klga_tmax\implementation`
- Result: Passed.
- Relevant output: no output, exit code 0.
- Proves: all source and test files parse and compile.
- Does not prove: runtime DB schema validity.

### Verification: `python -m pytest -q`

- Directory: `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\klga_tmax\implementation`
- Result: Passed.
- Relevant output: `28 passed in 1.89s`.
- Proves: Task 00 tests still pass and Task 01 tests cover station counts, pseudo-point counts, KLGA provider IDs, Tier A/B order, Tier C count/base point, station groups, schema contract text, and validation failure wiring.
- Does not prove: live provider data fetching.

### Verification: `python -m klga_tmax.cli --help`

- Directory: implementation root.
- Result: Passed.
- Relevant output: commands include `db`, `registry`, and `validate`.
- Proves: CLI module imports after Task 01 changes.
- Does not prove: DB command success.

### Verification: `python -m klga_tmax.cli validate --help`

- Directory: implementation root.
- Result: Passed.
- Relevant output: validation commands include `foundation` and `station-universe`.
- Proves: the new validation command is exposed through Typer.
- Does not prove: DB row correctness.

### Verification: `python -m klga_tmax.cli db migrate`

- Directory: implementation root.
- Environment: `KLGA_DB_URL=postgresql+psycopg://postgres:root@127.0.0.1:5432/klga_tmax_research`
- Result: Passed.
- Relevant output: `{"ok": true, "row_counts": {"registry.cutoffs": 4, "registry.feature_versions": 1, "registry.station_registry": 29, "registry.stations": 29}}`.
- Proves: Alembic can apply head and seeding is idempotent.
- Does not prove: future migrations will be conflict-free.

### Verification: `python -m klga_tmax.cli db inspect-contract`

- Directory: implementation root.
- Environment: same `KLGA_DB_URL`.
- Result: Passed.
- Relevant output: `ok=true`, `station_registry_rows=29`, `station_rows=29`, `tables_checked=18`, `indexes_checked=5`.
- Proves: required schemas, tables, columns, indexes, extension, and seed counts exist in the local Postgres DB.
- Does not prove: provider source data coverage.

### Verification: `python -m klga_tmax.cli validate foundation`

- Directory: implementation root.
- Environment: same `KLGA_DB_URL`.
- Result: Passed.
- Relevant output: `ok=true`, `station_registry_rows=29`, `station_rows=29`, `late_feature_rows=0`, `target_instance_rows=24`.
- Proves: Task 00 foundation validation still passes after the role and station count changes.
- Does not prove: Task 01 station group exactness by itself.

### Verification: `python -m klga_tmax.cli validate station-universe`

- Directory: implementation root.
- Environment: same `KLGA_DB_URL`.
- Result: Passed.
- Relevant output: `ok=true`, `mandatory_station_rows=19`, `mandatory_pseudo_point_rows=10`, `station_registry_rows_for_version=29`, `registry_stations_rows_for_canonical_ids=29`, `tier_a_points=4`, `tier_b_points=10`, `tier_c_points=25`.
- Proves: in-code constants and DB rows match the Task 01 station universe.
- Does not prove: downstream fetchers import this module yet.

### Verification: direct SQL snapshot through SQLAlchemy

- Directory: implementation root.
- Result: Passed.
- Relevant output:

```text
active_registry_stations=29
alembic_version=0002_station_universe
inactive_registry_stations=5
station_registry_v2026_06_27_klga_core=29
```

- Proves: the DB is migrated to the Task 01 internal revision and has 29 canonical rows plus 29 active compatibility rows.
- Does not prove: `psql` is available on PATH. A direct `psql` command was attempted and failed because PowerShell could not find `psql`.

### Verification: documentation quality gate

- Directory: implementation root.
- Command: `python C:\Users\ahmad\.codex\skills\exceptional-code-document-writer\scripts\documentation_quality_gate.py --doc C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\klga_tmax\strategy_spec\context\KLGA_TMAX_01_STATION_UNIVERSE_IMPLEMENTATION_DEEP_DIVE.md --repo-root C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction --changed-file ... --fail-on-warnings`
- Result: Passed.
- Relevant output: `Findings: 0 error(s), 0 warning(s)`.
- Proves: this document names the explicit changed-file scope, includes required sections, includes verification evidence, and passes the skill audit.
- Does not prove: a human reviewer agrees with every engineering choice.

## Operational Runbook

Run the station universe setup from:

```powershell
Set-Location C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\klga_tmax\implementation
$env:KLGA_DB_URL='postgresql+psycopg://postgres:root@127.0.0.1:5432/klga_tmax_research'
python -m klga_tmax.cli db migrate
python -m klga_tmax.cli registry seed
python -m klga_tmax.cli db inspect-contract
python -m klga_tmax.cli validate foundation
python -m klga_tmax.cli validate station-universe
```

Expected healthy state:

- `db migrate` prints `ok=true`.
- `registry seed` prints 29 rows for `registry.station_registry` and 29 active rows for `registry.stations`.
- `db inspect-contract` prints no failures.
- `validate station-universe` prints no failures and details for 19 mandatory stations, 10 pseudo-points, and 25 Tier C generated points.

Debug order for failures:

1. If `KLGA_DB_URL` is missing, set it and rerun the command.
2. If migration fails, inspect `alembic_version` and the latest migration error in the console output.
3. If station row counts are wrong, run `python -m klga_tmax.cli registry seed` and rerun validation.
4. If a coordinate or provider ID mismatch appears, compare the failing row to `station_universe.py`, not to a local fetcher list.
5. If a future fetcher needs a station outside Task 01, add a new registry version instead of mutating historical rows in place.

## Compatibility, Rollback, and Upgrade Notes

Compatibility with Task 00 is preserved by keeping `registry.stations` and its primary key. Existing foreign keys from silver/gold tables still target `registry.stations.station_id`.

The new canonical table is additive. Code that does not know about `registry.station_registry` can still use the active compatibility rows in `registry.stations`, but new provider fetchers should import `station_universe.py` or read `registry.station_registry`.

Upgrade from Task 00 to Task 01:

- Old role values in `registry.stations` are migrated to Task 01 role values.
- Canonical rows are inserted into `registry.station_registry`.
- Compatibility rows are refreshed in `registry.stations`.
- Older supplemental compatibility rows are left inactive.

Rollback:

- Downgrade drops the canonical table.
- Downgrade maps Task 01 roles back to Task 00 role categories.
- Do not rely on downgrade if downstream provider tables already store `station_registry_version` as a foreign contract.

## Known Limitations and Follow-Up Work

Limitation: provider fetchers are not yet implemented.

- Impact: GribStream, IEM, Wunderground, MOS, and Open-Meteo code must still be written.
- Reason: Task 01 only defines the station universe prerequisite.
- Revisit trigger: start the first provider ingestion task.
- Blocks release: no, for Task 01 foundation; yes, for actual data acquisition.

Limitation: no elevation source is seeded.

- Impact: elevation-aware features cannot consume `elevation_m` yet.
- Reason: the Task 01 station spec did not provide exact elevation values.
- Revisit trigger: a feature spec requires elevation correction.
- Blocks release: no.

Limitation: inactive compatibility rows remain in the local DB.

- Impact: raw `SELECT count(*) FROM registry.stations` returns 34 in this local DB, while active rows return 29.
- Reason: stale rows are preserved to avoid deleting historical compatibility records.
- Revisit trigger: a future cleanup task defines retention rules for inactive registry rows.
- Blocks release: no, because all inspectors and validators count active rows.

Limitation: Tier C is generated but not persisted.

- Impact: Tier C cannot be queried from `registry.station_registry` unless a future task persists a new registry version or a separate generated-grid table.
- Reason: the plan required deterministic Tier C helpers and tests, not default Tier C persistence.
- Revisit trigger: GribStream confirms acceptable bulk pricing and the research plan chooses Tier C.
- Blocks release: no.

Limitation: `psql` was not available on the current PowerShell PATH.

- Impact: the requested `psql` query was replaced by an equivalent SQLAlchemy readback.
- Reason: local PATH did not include a `psql` executable.
- Revisit trigger: install PostgreSQL client tools or add them to PATH.
- Blocks release: no, because DB commands and SQLAlchemy readback succeeded.

## Next Task Handoff

Provider fetchers should treat `station_universe.py` and `registry.station_registry` as authoritative. Do not copy station lists into GribStream, IEM, Wunderground, MOS, or Open-Meteo fetcher modules.

For airport observation providers:

- Use `provider_station_id(station_id, "iem_asos")` for IEM ASOS station IDs.
- Use `provider_station_id(station_id, "wunderground")` for Wunderground station IDs.
- Use `provider_station_id(station_id, "mos")` for MOS station suffixes.

For gridded providers:

- Use `coordinate_tier("A")` for the minimum viable pull.
- Use `coordinate_tier("B")` for normal production pulls.
- Use `coordinate_tier("C")` only when a research task explicitly enables it.

Every future fetcher should persist source rows with canonical `station_id` or `grid_point_id` values that trace back to `STATION_REGISTRY_VERSION`.

## Reviewer Checklist

- [x] Every Task 01 changed file is listed in Change Inventory.
- [x] Every implementation file has a dedicated File-by-File Deep Dive section.
- [x] The canonical registry version is documented.
- [x] The 19 station row requirement is implemented and tested.
- [x] The 10 pseudo-point row requirement is implemented and tested.
- [x] KLGA provider IDs are implemented and tested.
- [x] Tier A, Tier B, and Tier C contracts are implemented and tested.
- [x] Station groups are implemented and tested.
- [x] `registry.station_registry` schema and indexes are documented.
- [x] `registry.stations` compatibility behavior is documented.
- [x] Migration, seed, inspect, and validation commands were run against local Postgres.
- [x] No provider fetching or credentials were added.
- [x] Documentation quality gate passed with the explicit Task 01 changed-file scope.
