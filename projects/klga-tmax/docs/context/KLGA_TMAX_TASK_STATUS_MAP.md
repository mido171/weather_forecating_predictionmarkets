# KLGA Tmax Task Status Map

Last updated: 2026-07-10

## Purpose

This file is the quick status map for the numbered KLGA implementation tasks. It answers one question directly:

```text
For each task, is it fully completed or not?
```

Use `KLGA_TMAX_TASK_IMPLEMENTATION_QUEUE.md` for assignment details, dependencies, and required handoff documents. Use this file when you only need a compact completion tracker.

## Completion Rule

A task is marked completed only when all required implementation deliverables exist:

- code under `src/klga_tmax`
- tests under `tests`
- Alembic migration when schema changes
- CLI command when the task creates a user-run operation
- validation command when the task creates persistent data or a contract
- `db inspect-contract` update when new schema objects become part of the core contract
- deep-dive implementation document under `docs/context`
- verification evidence in that deep-dive document

Planning notes, prerequisite research, scraper audits, quota planning, or support emails do not make a numbered task completed.

## Status Values

| Value | Meaning |
| --- | --- |
| `COMPLETED` | Full implementation, tests, validation, and deep-dive handoff exist. |
| `NOT_COMPLETED` | The task has not been fully implemented yet. It may still be ready to assign or have prerequisite research done. |

## Task Map

| Order | Task folder | Source spec | Completed? | Queue status | Completion evidence | Notes |
|---:|---|---|---|---|---|---|
| 00 | `00_foundation_universal_ingestion_contract_and_availability_ledger` | `00_universal_ingestion_contract_and_availability_ledger.md` | `COMPLETED` | `DONE` | `KLGA_TMAX_00_FOUNDATION_IMPLEMENTATION_DEEP_DIVE.md` | Foundation schema, Alembic, CLI, audit, cutoffs, availability ledger, target instances, and validation are implemented. |
| 01 | `01_station_universe_and_coordinates` | `10_station_universe_and_coordinates.md` | `COMPLETED` | `DONE` | `KLGA_TMAX_01_STATION_UNIVERSE_IMPLEMENTATION_DEEP_DIVE.md` | Canonical station registry, provider IDs, pseudo-points, coordinate tiers, station groups, seed path, and validation are implemented. |
| 02 | `02_wunderground_settlement_actuals` | `01_wunderground_settlement_actuals.md` | `COMPLETED` | `DONE` | `KLGA_TMAX_02_WUNDERGROUND_SETTLEMENT_ACTUALS_IMPLEMENTATION_DEEP_DIVE.md` | Wunderground/Weather.com schema, provider client, parser, persistence, bounded backfill CLI, coverage tracking, validation, KLGA August 2021 smoke evidence, the full `1973-01-01` through `2026-06-27` all-station backfill, and the post-backfill sanity correction are complete. Final strict usable-Tmax coverage has `354,501` saved station-days, `16,683` no-data station-days, `0` failed, and `0` not-fetched. |
| 03 | `03_iem_mos_station_guidance` | `02_iem_mos_station_guidance.md` | `NOT_COMPLETED` | `READY` | Not yet available. Expected: `KLGA_TMAX_03_IEM_MOS_STATION_GUIDANCE_IMPLEMENTATION_DEEP_DIVE.md` | Ready to implement after tasks 00 and 01. |
| 04 | `04_asos_metar_hf_and_minute_observations` | `04_asos_metar_hf_and_minute_observations.md` | `NOT_COMPLETED` | `READY` | Not yet available. Expected: `KLGA_TMAX_04_ASOS_METAR_HF_AND_MINUTE_OBSERVATIONS_IMPLEMENTATION_DEEP_DIVE.md` | Ready to implement after tasks 00 and 01. |
| 05 | `05_open_meteo_auxiliary_forecast_runs` | `06_open_meteo_auxiliary_forecast_runs.md` | `NOT_COMPLETED` | `READY` | Not yet available. Expected: `KLGA_TMAX_05_OPEN_METEO_AUXILIARY_FORECAST_RUNS_IMPLEMENTATION_DEEP_DIVE.md` | Ready to implement, with official Open-Meteo endpoint/selector verification before final code. |
| 06 | `06_polymarket_market_data` | `08_polymarket_market_data.md` | `NOT_COMPLETED` | `READY` | Not yet available. Expected: `KLGA_TMAX_06_POLYMARKET_MARKET_DATA_IMPLEMENTATION_DEEP_DIVE.md` | Ready to implement, with current Polymarket endpoint verification before final code. |
| 07 | `07_ncep_availability_cutoff_audit` | `09_ncep_availability_cutoff_audit.md` | `NOT_COMPLETED` | `READY` | Not yet available. Expected: `KLGA_TMAX_07_NCEP_AVAILABILITY_CUTOFF_AUDIT_IMPLEMENTATION_DEEP_DIVE.md` | Ready to implement; recurring live polling should remain disabled until explicitly configured. |
| 08 | `08_gribstream_nwp_forecast_runs` | `03_gribstream_nwp_forecast_runs.md` | `NOT_COMPLETED` | `READY_WITH_LIVE_BOUNDARY` | Not yet available. Expected: `KLGA_TMAX_08_GRIBSTREAM_NWP_FORECAST_RUNS_IMPLEMENTATION_DEEP_DIVE.md` | GribStream planning and quota discussion are not a completed task implementation. |
| 09 | `09_rtma_urma_analysis_fields` | `05_rtma_urma_analysis_fields.md` | `NOT_COMPLETED` | `DEFERRED` | Not yet available. Expected: `KLGA_TMAX_09_RTMA_URMA_ANALYSIS_FIELDS_IMPLEMENTATION_DEEP_DIVE.md` | Deferred until the GribStream client foundation from task 08 exists. |
| 10 | `10_noaa_raw_archives_optional_bulk_fallback` | `07_noaa_raw_archives_optional_bulk_fallback.md` | `NOT_COMPLETED` | `OPTIONAL` | Not yet available. Expected: `KLGA_TMAX_10_NOAA_RAW_ARCHIVES_OPTIONAL_BULK_FALLBACK_IMPLEMENTATION_DEEP_DIVE.md` | Optional fallback only if primary provider paths cannot satisfy coverage, cost, or auditability needs. |

## Current Completion Snapshot

```text
Completed tasks: 3 / 11
Completed task numbers: 00, 01, 02
Not completed task numbers: 03, 04, 05, 06, 07, 08, 09, 10
Next normal task: 03_iem_mos_station_guidance
```

## Update Procedure

When a task is fully completed:

1. Change `Completed?` from `NOT_COMPLETED` to `COMPLETED`.
2. Change `Queue status` to `DONE`.
3. Replace the expected deep-dive filename with the actual completed deep-dive filename.
4. Update `Current Completion Snapshot`.
5. Update `KLGA_TMAX_TASK_IMPLEMENTATION_QUEUE.md` with the same completion status.
6. Do not mark live backfill completion unless the task explicitly required live data pull and the pull was actually run.
