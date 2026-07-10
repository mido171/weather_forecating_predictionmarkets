# T07-T12 Consolidated Tactical H24N GribStream Backfill

Status: not-completed

This task replaces the old split GribStream fetching tasks T07 through T12. Those split tasks encouraged too much coordination overhead and allowed the previous broad 0-84h runner to fetch far more data than the HKG Tmax use case needs.

## Scope

Acquire only the tactical H24N GribStream data needed for HKO daily Tmax research:

- exact `/runs` requests only;
- `timesList` model-run selection only;
- one selected run cycle per model and target date;
- exact lead windows from the tactical plan;
- 12 HKO stencil points for deterministic and ensemble-mean models;
- HKO center only for full-member ensembles;
- compact wide rows, not one row per variable;
- raw response hash and request hash recorded for every chunk.

## Included Models

Core:

- `gfs`
- `gefsatmosmean`
- `gefsatmos`
- `ifsoper`
- `ifsenfo`
- `cwawrf15` prospective collection

Useful-if-cheap / shadow:

- `aifsoper`
- `aifsenfo`
- `aigfssfc`
- `aigfspres`
- `aigefssfc`
- `graphcast`
- `fourcastnetgfs`
- `nbmoc` probe only

## Explicitly Retired

The old broad runner is retired:

```text
scripts/run_t07_t13_gribstream_backfill.py
```

It now blocks by default. Do not use it unless deliberately investigating the old failed approach.

## Active Implementation Files

```text
migrations/postgres/20260625_0007_tactical_gribstream_h24n_schema.sql
scripts/reset_tactical_gribstream_store.py
scripts/run_tactical_gribstream_h24n_smoke.py
scripts/run_tactical_gribstream_first_week.py
scripts/run_tactical_gribstream_batch_smoke.py
code/tests/test_tactical_gribstream_h24n.py
documentation/T07_T12_CONSOLIDATED_TACTICAL_GRIBSTREAM_BACKFILL_RUNBOOK.md
```

## Progress Notes

2026-06-25:

- Consolidated task created and old split T07-T12 folders superseded.
- Legacy broad GribStream storage was purged from the active tactical store.
- Tactical schema, exact `/runs` payloads, first-week pull, and model-specific batch smoke runner are implemented.
- 10-week batch sanity check completed with 96 HTTP 200 API events, 0 4xx/5xx/429 events, 95 completed chunks, 1 completed-empty `nbmoc` chunk, and 123,652 rows written to `nwp_tactical.forecast_wide`.
- Evidence lives in `experiments/0214_tactical_h24n_gribstream_backfill/batch_smoke_10w/` and `data/_pipeline_internal/raw/gribstream_tactical_batch_smoke_10w/`.
- Tmax/leakage sanity found that most models can produce leakage-safe daily Tmax features only after an H24N cutoff filter is applied; raw table queries are not automatically feature-safe.
- `aigfspres`, `aigefssfc`, and `nbmoc` must not be treated as daily Tmax-producing sources unless later selector/provider probes prove usable non-null 2m/Tmax coverage.
- Full tactical backfill `full_tactical_backfill_ok_tmax` ran after provider allowance was increased. It processed all 1,163 planned chunks, wrote 1,964,157 rows to `nwp_tactical.forecast_wide`, consumed an estimated 1,889,276 credits, and recorded no 429/4xx/5xx API responses.
- The full tactical backfill is still not a clean-pass completion because of data-quality flags: recent `ifsenfo` chunks are missing member `0`, the `fourcastnetgfs` tail returned empty, and `nbmoc` returned empty.
- Full-run evidence lives in `experiments/0214_tactical_h24n_gribstream_backfill/full_tactical_backfill_ok_tmax/`, `data/_pipeline_internal/raw/gribstream_tactical_full_tactical_backfill_ok_tmax/`, and `documentation/T07_T12_FULL_TACTICAL_BACKFILL_20260625_RESULT.md`.
- Deep sanity audit completed with 1,964,157 full-run rows, 1,965,090 total `forecast_wide` rows, 933 older `batch_smoke_10w` `gefsatmos` rows still mixed into `forecast_wide`, 0 missing full-run raw files, 0 raw byte-size mismatches, and 0 API HTTP errors.
- Deep sanity audit evidence lives in `documentation/T07_T12_DEEP_SANITY_AUDIT_20260625.md`, `experiments/0214_tactical_h24n_gribstream_backfill/full_tactical_backfill_ok_tmax/DEEP_SANITY_AUDIT_20260625.md`, and `experiments/0214_tactical_h24n_gribstream_backfill/full_tactical_backfill_ok_tmax/deep_sanity_audit_20260625.json`.
- Remaining work: resolve or explicitly accept the full-run data-quality flags, enforce the H24N leakage-safe feature filter downstream, then move this task to completed with a `COMPLETION_RECORD.md`.

## Acceptance Criteria

1. T07-T12 old folders are archived under `tasks/superseded/T07_T12_legacy_split_gribstream_fetch_tasks/`.
2. This folder is the only active not-completed GribStream fetching task.
3. Legacy broad GribStream rows/raw objects are purged from active storage.
4. Tactical schema exists and blocks broad request ranges at table level.
5. Smoke payloads use exact `timesList` and exact lead windows.
6. No more than three authenticated data calls are used for the initial smoke.
7. The final backfill runner must use one worker, few larger requests, bounded retries, and `Retry-After` handling.

## Completion Record Requirement

When this task is eventually completed, move this folder to:

```text
tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/tasks/completed/T07_T12_tactical_h24n_gribstream_backfill/
```

and add `COMPLETION_RECORD.md` explaining:

- exact models/date ranges fetched;
- request chunking policy;
- final row counts and credit usage;
- acceptance criteria evidence;
- remaining exclusions or provider blockers.
