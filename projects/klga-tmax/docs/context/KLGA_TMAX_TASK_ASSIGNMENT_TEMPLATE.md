# KLGA Tmax Task Assignment Template

Use this template when assigning one numbered KLGA task to a Codex implementation conversation.

## Copy-Paste Assignment

```text
Implement KLGA Task TASK_NUMBER: TASK_FOLDER_NAME.

Use engineering-excellence. If the task requires implementation documentation, also use exceptional-code-document-writer.

You must read these files in full before editing:

1. AGENTS.md
2. START_HERE.md
3. docs/specifications/strategy/KLGA_TMAX_TRADING_STRATEGY_SPEC.md
4. docs/specifications/strategy/supplemental_doc_1.md
5. docs/specifications/strategy/supplemental_doc_1_patch_1.md
6. docs/context/KLGA_TMAX_POSTGRES_PERSISTENCE_CONTEXT.md
7. docs/context/KLGA_TMAX_00_FOUNDATION_IMPLEMENTATION_DEEP_DIVE.md
8. docs/context/KLGA_TMAX_01_STATION_UNIVERSE_IMPLEMENTATION_DEEP_DIVE.md
9. docs/context/KLGA_TMAX_TASK_IMPLEMENTATION_QUEUE.md
10. docs/specifications/data-acquisition/TASK_FOLDER_NAME/TASK_SOURCE_SPEC_FILE

Project root, relative to the monorepo:
projects/klga-tmax

Scope:
- Implement the task fully under the canonical KLGA project.
- Keep source code under src/klga_tmax.
- Keep tests under tests.
- Add Alembic migration(s) for schema changes.
- Add CLI command(s) for task operations where relevant.
- Add validation command(s) for the task contract.
- Update db inspect-contract if new objects become required.
- Do not fetch live provider data unless this assignment explicitly says live fetching is authorized.
- Do not write provider credentials into the repo.
- Preserve the Task 00 availability/leakage contract.
- Use registry.station_registry and station_universe.py for station IDs, provider IDs, and coordinate tiers.

Required documentation output:
docs/context/TASK_DEEP_DIVE_DOC_NAME

Required offline verification from the project root:
python -m compileall -q src tests
python -m pytest -q
python -m klga_tmax.cli --help
python -m klga_tmax.cli validate --help
Run the task-specific validation command added by this task.

Database migration, contract inspection, and DB-backed validation require separate explicit
authorization plus a prepared local KLGA_DB_URL. Never embed a real password in this task.

At completion, update KLGA_TMAX_TASK_IMPLEMENTATION_QUEUE.md status for this task and state any live-fetch, credential, quota, or provider-data limitation.
```

## Fill-In Values

Use the queue file for exact values. Common next assignments:

### Task 02

```text
TASK_NUMBER = 02
TASK_FOLDER_NAME = 02_wunderground_settlement_actuals
TASK_SOURCE_SPEC_FILE = 01_wunderground_settlement_actuals.md
TASK_DEEP_DIVE_DOC_NAME = KLGA_TMAX_02_WUNDERGROUND_SETTLEMENT_ACTUALS_IMPLEMENTATION_DEEP_DIVE.md
```

### Task 03

```text
TASK_NUMBER = 03
TASK_FOLDER_NAME = 03_iem_mos_station_guidance
TASK_SOURCE_SPEC_FILE = 02_iem_mos_station_guidance.md
TASK_DEEP_DIVE_DOC_NAME = KLGA_TMAX_03_IEM_MOS_STATION_GUIDANCE_IMPLEMENTATION_DEEP_DIVE.md
```

### Task 04

```text
TASK_NUMBER = 04
TASK_FOLDER_NAME = 04_asos_metar_hf_and_minute_observations
TASK_SOURCE_SPEC_FILE = 04_asos_metar_hf_and_minute_observations.md
TASK_DEEP_DIVE_DOC_NAME = KLGA_TMAX_04_ASOS_METAR_HF_AND_MINUTE_OBSERVATIONS_IMPLEMENTATION_DEEP_DIVE.md
```

### Task 05

```text
TASK_NUMBER = 05
TASK_FOLDER_NAME = 05_open_meteo_auxiliary_forecast_runs
TASK_SOURCE_SPEC_FILE = 06_open_meteo_auxiliary_forecast_runs.md
TASK_DEEP_DIVE_DOC_NAME = KLGA_TMAX_05_OPEN_METEO_AUXILIARY_FORECAST_RUNS_IMPLEMENTATION_DEEP_DIVE.md
```

### Task 06

```text
TASK_NUMBER = 06
TASK_FOLDER_NAME = 06_polymarket_market_data
TASK_SOURCE_SPEC_FILE = 08_polymarket_market_data.md
TASK_DEEP_DIVE_DOC_NAME = KLGA_TMAX_06_POLYMARKET_MARKET_DATA_IMPLEMENTATION_DEEP_DIVE.md
```

### Task 07

```text
TASK_NUMBER = 07
TASK_FOLDER_NAME = 07_ncep_availability_cutoff_audit
TASK_SOURCE_SPEC_FILE = 09_ncep_availability_cutoff_audit.md
TASK_DEEP_DIVE_DOC_NAME = KLGA_TMAX_07_NCEP_AVAILABILITY_CUTOFF_AUDIT_IMPLEMENTATION_DEEP_DIVE.md
```

### Task 08

```text
TASK_NUMBER = 08
TASK_FOLDER_NAME = 08_gribstream_nwp_forecast_runs
TASK_SOURCE_SPEC_FILE = 03_gribstream_nwp_forecast_runs.md
TASK_DEEP_DIVE_DOC_NAME = KLGA_TMAX_08_GRIBSTREAM_NWP_FORECAST_RUNS_IMPLEMENTATION_DEEP_DIVE.md
```

### Task 09

```text
TASK_NUMBER = 09
TASK_FOLDER_NAME = 09_rtma_urma_analysis_fields
TASK_SOURCE_SPEC_FILE = 05_rtma_urma_analysis_fields.md
TASK_DEEP_DIVE_DOC_NAME = KLGA_TMAX_09_RTMA_URMA_ANALYSIS_FIELDS_IMPLEMENTATION_DEEP_DIVE.md
```

### Task 10

```text
TASK_NUMBER = 10
TASK_FOLDER_NAME = 10_noaa_raw_archives_optional_bulk_fallback
TASK_SOURCE_SPEC_FILE = 07_noaa_raw_archives_optional_bulk_fallback.md
TASK_DEEP_DIVE_DOC_NAME = KLGA_TMAX_10_NOAA_RAW_ARCHIVES_OPTIONAL_BULK_FALLBACK_IMPLEMENTATION_DEEP_DIVE.md
```

## Live Fetch Authorization Add-On

Add this block only when the user wants actual provider data pulled during the task:

```text
Live fetching is authorized for this assignment.

Provider:
Date range:
Stations or coordinate tier:
Maximum request/credit budget:
Credential source:
Rate-limit or pacing rule:
Stop condition:
```

If that block is absent, implement code, schema, fixtures, tests, and smoke checks only. Do not run a broad live backfill.
