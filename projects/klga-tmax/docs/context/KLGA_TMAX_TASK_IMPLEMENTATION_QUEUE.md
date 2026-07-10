# KLGA Tmax Task Implementation Queue

## Purpose

This file is the task assignment index for the remaining KLGA Tmax implementation work. Use it to select the next task, find the exact source spec, understand dependencies, and know what completion must leave behind.

The folder number is the execution order. The inner markdown filename is the original strategy-spec number and may not match the folder number.

## Current Baseline

Canonical project root, relative to the monorepo:

```text
projects/klga-tmax
```

Canonical DB variable:

```powershell
$env:KLGA_DB_URL = "postgresql+psycopg://<user>:<password>@127.0.0.1:5432/klga_tmax_research"
```

Completed foundation:

- Task 00 implemented and documented in `KLGA_TMAX_00_FOUNDATION_IMPLEMENTATION_DEEP_DIVE.md`.
- Task 01 implemented and documented in `KLGA_TMAX_01_STATION_UNIVERSE_IMPLEMENTATION_DEEP_DIVE.md`.
- Task 02 implemented and documented in `KLGA_TMAX_02_WUNDERGROUND_SETTLEMENT_ACTUALS_IMPLEMENTATION_DEEP_DIVE.md`.
- `registry.station_registry` is canonical for station and pseudo-point identity.
- `registry.stations` remains a compatibility projection for existing foreign keys.

## Status Values

- `DONE`: code, tests, migration if needed, validation, and deep-dive document exist.
- `READY`: can be assigned to a Codex implementation conversation now.
- `READY_WITH_LIVE_BOUNDARY`: code/schema/client work can be implemented now, but full live backfill needs credentials, quota confirmation, or explicit user approval.
- `DEFERRED`: should wait until an upstream task or provider condition is satisfied.
- `OPTIONAL`: not required for the first complete data acquisition pass.

## Ordered Task Queue

| Order | Status | Task folder | Source spec | Primary implementation output | Key dependencies | Required handoff document |
|---:|---|---|---|---|---|---|
| 00 | DONE | `00_foundation_universal_ingestion_contract_and_availability_ledger` | `00_universal_ingestion_contract_and_availability_ledger.md` | Shared schemas, availability ledger, audit tables, bronze/silver/gold foundation, CLI, validation. | PostgreSQL database. | `KLGA_TMAX_00_FOUNDATION_IMPLEMENTATION_DEEP_DIVE.md` |
| 01 | DONE | `01_station_universe_and_coordinates` | `10_station_universe_and_coordinates.md` | Canonical station registry, pseudo-points, coordinate tiers, groups, validation. | Task 00. | `KLGA_TMAX_01_STATION_UNIVERSE_IMPLEMENTATION_DEEP_DIVE.md` |
| 02 | DONE | `02_wunderground_settlement_actuals` | `01_wunderground_settlement_actuals.md` | Wunderground/Weather.com provider client, raw/normalized settlement actual tables, label availability rules, bounded/resumable backfill command, coverage tracking, validation, and post-backfill data sanity correction. | Tasks 00 and 01. Provider key is read from `WUNDERGROUND_API_KEY` or `WEATHERCOM_API_KEY`; full all-station backfill completed for `1973-01-01` through `2026-06-27` with `354,501` usable saved Tmax station-days, `16,683` no-data station-days, `0` failed, and `0` not-fetched. | `KLGA_TMAX_02_WUNDERGROUND_SETTLEMENT_ACTUALS_IMPLEMENTATION_DEEP_DIVE.md` |
| 03 | READY | `03_iem_mos_station_guidance` | `02_iem_mos_station_guidance.md` | IEM MOS product fetch/parser, product identity tables, run/availability metadata, station mapping through Task 01 registry, validation. | Tasks 00 and 01. | `KLGA_TMAX_03_IEM_MOS_STATION_GUIDANCE_IMPLEMENTATION_DEEP_DIVE.md` |
| 04 | READY | `04_asos_metar_hf_and_minute_observations` | `04_asos_metar_hf_and_minute_observations.md` | IEM ASOS/METAR and minute observation ingestion contracts, observation normalization, station/time availability validation. | Tasks 00 and 01. | `KLGA_TMAX_04_ASOS_METAR_HF_AND_MINUTE_OBSERVATIONS_IMPLEMENTATION_DEEP_DIVE.md` |
| 05 | READY | `05_open_meteo_auxiliary_forecast_runs` | `06_open_meteo_auxiliary_forecast_runs.md` | Open-Meteo exact-run client, model update metadata, availability rules, auxiliary forecast persistence, validation. | Tasks 00 and 01. Official Open-Meteo docs should be checked live before final selectors. | `KLGA_TMAX_05_OPEN_METEO_AUXILIARY_FORECAST_RUNS_IMPLEMENTATION_DEEP_DIVE.md` |
| 06 | READY | `06_polymarket_market_data` | `08_polymarket_market_data.md` | Market discovery, bucket parsing, metadata/orderbook/price-history persistence, trading schema validation. | Task 00. Official Polymarket docs should be checked live before endpoint contracts. | `KLGA_TMAX_06_POLYMARKET_MARKET_DATA_IMPLEMENTATION_DEEP_DIVE.md` |
| 07 | READY | `07_ncep_availability_cutoff_audit` | `09_ncep_availability_cutoff_audit.md` | NCEP production-status collector, raw poll retention, parsed job rows, availability audit validation. | Task 00. Live collector scheduling decision required before enabling recurring runs. | `KLGA_TMAX_07_NCEP_AVAILABILITY_CUTOFF_AUDIT_IMPLEMENTATION_DEEP_DIVE.md` |
| 08 | READY_WITH_LIVE_BOUNDARY | `08_gribstream_nwp_forecast_runs` | `03_gribstream_nwp_forecast_runs.md` | GribStream client, dataset catalog discovery, exact run selection, request batching, quota accounting, NWP persistence, validation. | Tasks 00 and 01. GribStream API key and quota/amnesty confirmation for full backfill. | `KLGA_TMAX_08_GRIBSTREAM_NWP_FORECAST_RUNS_IMPLEMENTATION_DEEP_DIVE.md` |
| 09 | DEFERRED | `09_rtma_urma_analysis_fields` | `05_rtma_urma_analysis_fields.md` | RTMA/URMA analysis ingestion via GribStream, live vs retrospective availability separation, validation. | Tasks 00, 01, and GribStream client foundation from Task 08. | `KLGA_TMAX_09_RTMA_URMA_ANALYSIS_FIELDS_IMPLEMENTATION_DEEP_DIVE.md` |
| 10 | OPTIONAL | `10_noaa_raw_archives_optional_bulk_fallback` | `07_noaa_raw_archives_optional_bulk_fallback.md` | Optional raw archive fallback modules for NOAA/ECMWF sources, provenance and extraction contracts. | Use only when primary provider path cannot meet coverage, cost, or auditability needs. | `KLGA_TMAX_10_NOAA_RAW_ARCHIVES_OPTIONAL_BULK_FALLBACK_IMPLEMENTATION_DEEP_DIVE.md` |

## Recommended Next Assignment

Next normal assignment:

```text
03_iem_mos_station_guidance
```

Reason:

- IEM MOS is public, long-history, and station-specific.
- It is a forecast backbone source that can progress without GribStream quota.
- Settlement labels from task 02 are now in place for the KLGA WU path.

If the user wants to advance high-frequency observed weather instead, assign:

```text
04_asos_metar_hf_and_minute_observations
```

Reason:

- It is public, station-based, and complementary to Wunderground actuals.
- It provides intraday observational features separate from the settlement-label source.

## Per-Task Completion Contract

A task is not complete until it has:

1. A migration when schema changes.
2. Source code under `src/klga_tmax`.
3. Tests under `tests`.
4. A CLI command for user-run operations when relevant.
5. A validation command when persistent rows or contracts are introduced.
6. `db inspect-contract` updates when required objects become part of the core contract.
7. A deep-dive implementation document under `docs/context`.
8. Verification evidence for compile, tests, CLI help, migration, contract inspection, and task validation.
9. A clear statement of any live-fetch boundary, credential requirement, rate-limit risk, or unavailable provider data.

## Standard Verification Command Set

From `projects/klga-tmax` in a prepared local environment:

```powershell
python -m compileall -q src tests
python -m pytest -q
python -m klga_tmax.cli --help
python -m klga_tmax.cli validate --help
```

Database migration, contract inspection, and persistent-data validation are a separate,
explicitly authorized step against a prepared local `KLGA_DB_URL`; they are not part of
offline startup verification. Then run the task-specific dry-run/fixture checks and any
authorized local-database validation added by the task.

## Assignment Artifacts

Use these files when starting a new Codex conversation:

- `KLGA_TMAX_TASK_ASSIGNMENT_TEMPLATE.md`
- `KLGA_TMAX_TASK_HANDOFF_CHECKLIST.md`
- This queue file.

The template is designed to be pasted into a new Codex task with the task folder and source spec filled in from the table above.
