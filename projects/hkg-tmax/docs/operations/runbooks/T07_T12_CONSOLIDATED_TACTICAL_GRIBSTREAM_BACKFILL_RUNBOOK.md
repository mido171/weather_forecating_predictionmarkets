# T07-T12 Consolidated Tactical GribStream Backfill Runbook

Last updated: 2026-06-25

> **Relocation note (2026-07-10):** This document preserves historical task,
> code, migration, evidence, and data paths. Current code/tests use `src` and
> `tests`, migrations use `db/migrations/postgres`, planning packages use
> `planning/tasks`, and mutable data/run outputs use the configured external
> roots. Treat every old path below as provenance until mapped to those homes.

## Current Decision

T07-T12 are now one task:

```text
tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/tasks/not-completed/T07_T12_tactical_h24n_gribstream_backfill/
```

The previous split folders are preserved only for traceability:

```text
tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/tasks/superseded/T07_T12_legacy_split_gribstream_fetch_tasks/
```

T13 is not included because it is an HKO ARWF exact-vintage collector, not a GribStream API task.

## Hard Rules

- Use one worker.
- Use `/api/v2/{dataset}/runs`.
- Use `timesList` for exact model-run timestamps.
- Do not use `forecastedFrom` or `forecastedUntil` for historical primary backfill.
- Batch by model/month/run-time list once provider allowance is agreed.
- Use the 12-point HKO stencil for deterministic and ensemble-mean models.
- Use HKO center only for full-member ensembles.
- Store compact wide rows in `nwp_tactical.forecast_wide`.
- Record every request in `nwp_tactical.acquisition_chunk`.
- Record raw object hash and row count in `nwp_tactical.raw_response_object`.
- Honor `Retry-After`; on `429`, pause and stop the active loop rather than sending more requests.

## Active Files

```text
migrations/postgres/20260625_0007_tactical_gribstream_h24n_schema.sql
scripts/reset_tactical_gribstream_store.py
scripts/run_tactical_gribstream_h24n_smoke.py
scripts/run_tactical_gribstream_first_week.py
scripts/run_tactical_gribstream_batch_smoke.py
code/tests/test_tactical_gribstream_h24n.py
```

The old broad runner is retired and blocks by default:

```text
scripts/run_t07_t13_gribstream_backfill.py
```

## Clean-Slate Reset

Dry run:

```powershell
.\.venv\Scripts\python.exe scripts\reset_tactical_gribstream_store.py --apply-schema --purge-raw
```

Execute:

```powershell
.\.venv\Scripts\python.exe scripts\reset_tactical_gribstream_store.py --apply-schema --purge-raw --execute
```

Summary output:

```text
experiments/0214_tactical_h24n_gribstream_backfill/legacy_purge_summary.json
```

## Smoke Test

The GribStream API path is per dataset, so two or three data calls cannot fetch every model. The smoke runner therefore:

1. builds the tactical payloads for all core and optional models;
2. runs public catalog preflight for all of them;
3. executes only three authenticated `/runs` data calls by default:
   - `gfs`
   - `ifsenfo`
   - `aifsoper`

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_tactical_gribstream_h24n_smoke.py --max-data-calls 3
```

All-model command used for the 2026-06-25 smoke:

```powershell
.\.venv\Scripts\python.exe scripts\run_tactical_gribstream_h24n_smoke.py --max-data-calls 14 --smoke-datasets "gfs,gefsatmosmean,gefsatmos,ifsoper,ifsenfo,cwawrf15,aifsoper,aifsenfo,aigfssfc,aigfspres,aigefssfc,graphcast,fourcastnetgfs,nbmoc" --api-min-interval-seconds 12
```

Outputs:

```text
experiments/0214_tactical_h24n_gribstream_backfill/catalog_preflight.json
experiments/0214_tactical_h24n_gribstream_backfill/smoke_api_results.csv
experiments/0214_tactical_h24n_gribstream_backfill/smoke_status.json
experiments/0214_tactical_h24n_gribstream_backfill/request_payloads/
data/_pipeline_internal/raw/gribstream_tactical_smoke/
```

Latest all-model smoke result, 2026-06-25:

- `gfs`: HTTP 200, 300 rows.
- `gefsatmosmean`: HTTP 200, 96 rows.
- `gefsatmos`: HTTP 200, 248 rows.
- `ifsoper`: HTTP 200, 108 rows.
- `ifsenfo`: HTTP 200, 408 rows.
- `cwawrf15`: HTTP 200, 60 rows.
- `aifsoper`: HTTP 200, 60 rows.
- `aifsenfo`: HTTP 200, 204 rows.
- `aigfssfc`: HTTP 200, 0 rows.
- `aigfspres`: HTTP 200, 0 rows.
- `aigefssfc`: HTTP 200, 124 rows.
- `graphcast`: HTTP 200, 60 rows.
- `fourcastnetgfs`: HTTP 200, 60 rows.
- `nbmoc`: HTTP 200, 0 rows.
- No 4xx, 5xx, or 429 responses were recorded in the all-model smoke result.

Interpretation:

- 11 of 14 planned models returned usable HKO tactical rows.
- `aigfssfc`, `aigfspres`, and `nbmoc` did not error, but returned empty result sets for the tested HKO `/runs` slices.
- Follow-up probes using one HKO point and corrected selectors also returned HTTP 200 with 0 rows for `aigfssfc`, `aigfspres`, and `nbmoc`.

Current selector warnings from catalog preflight:

- `aigfssfc`: current parameter summary does not expose `DPT`; do not require dewpoint for this dataset unless a later catalog detail query proves a valid exact selector.
- `nbmoc`: current parameter detail exposes `TMP`, `DPT`, `WIND`, and `WDIR`, but not `PRMSL`, `UGRD`, or `VGRD`; keep `nbmoc` as probe-only unless HKO-domain rows can be proven.

## First-Week Tactical Pull

Measured command used on 2026-06-25:

```powershell
.\.venv\Scripts\python.exe scripts\run_tactical_gribstream_first_week.py --apply-schema --api-min-interval-seconds 12 --api-max-attempts 2 --pause-on-429-seconds 300
```

Outputs:

```text
experiments/0214_tactical_h24n_gribstream_backfill/first_week_pull/first_week_results.csv
experiments/0214_tactical_h24n_gribstream_backfill/first_week_pull/first_week_summary.json
experiments/0214_tactical_h24n_gribstream_backfill/logs/gribstream_first_week_api_events.jsonl
data/_pipeline_internal/raw/gribstream_tactical_first_week/
```

Database writes:

```text
nwp_tactical.acquisition_chunk
nwp_tactical.raw_response_object
nwp_tactical.forecast_wide
```

Measured result:

- Status: passed.
- Dataset requests: 14.
- API errors: no 4xx, 5xx, or 429 responses.
- Raw response objects: 14.
- `acquisition_chunk` statuses: 13 `completed`, 1 `completed_empty`.
- Rows returned/upserted to `forecast_wide`: 11,796.
- Estimated credits consumed: 10,952.
- Wall time: 193.758 seconds.
- Effective speed including one-thread waits: 3,391 credits/minute and 60.88 rows/second.

Per-model returned rows:

```text
gfs: 1,800 rows, 1,950 estimated credits
gefsatmosmean: 672 rows, 448 estimated credits
gefsatmos: 1,736 rows, 1,736 estimated credits
ifsoper: 756 rows, 756 estimated credits
ifsenfo: 2,856 rows, 2,856 estimated credits
cwawrf15: 180 rows, 150 estimated credits
aifsoper: 420 rows, 280 estimated credits
aifsenfo: 1,428 rows, 1,428 estimated credits
aigfssfc: 120 rows, 40 estimated credits
aigfspres: 120 rows, 20 estimated credits
aigefssfc: 868 rows, 868 estimated credits
graphcast: 420 rows, 210 estimated credits
fourcastnetgfs: 420 rows, 210 estimated credits
nbmoc: 0 rows, 0 estimated credits
```

Projection from the first-week measurement:

- Current tactical scope projected credits: 1,895,123.
- Current tactical scope projected rows: 1,971,993.
- Provider temporary allowance: 768,000 credits/day for 3 days, or 2,304,000 credits total.
- Headroom against temporary allowance: 408,877 credits.
- Estimated completion time at measured speed: 558.79 minutes, or 9.31 hours.

Important caveats:

- The credit figure is estimated from returned run-valid-time/member/parameter dimensions because the API response did not expose a per-request billing header in the client.
- `cwawrf15` is a rolling/prospective sample, not a full historical first week.
- `nbmoc` remains probe-only and returned zero rows.
- The projection is intentionally conservative for quota planning; some first archive-start dates returned fewer rows than the plan maximum.

## Model-Specific Batch Sizes

Use these batch sizes for the tactical backfill unless a provider or DB sanity check says otherwise:

```text
gfs: 14 days
gefsatmosmean: 31 days
gefsatmos: 5 days
ifsoper: 14 days
ifsenfo: 5 days
cwawrf15: 3 rolling days only
aifsoper: 10 days
aifsenfo: 5 days
aigfssfc: 31 days
aigfspres: 14 days
aigefssfc: 5 days
graphcast: 14 days
fourcastnetgfs: 14 days
nbmoc: 7-day probe only
```

Cold full-backfill estimate with these batch sizes:

- Total chunks/requests: 1,163.
- Estimated credits: 1,895,063.
- Estimated wide rows: 1,971,873.

## 10-Week Batch Smoke

Measured command used on 2026-06-25:

```powershell
.\.venv\Scripts\python.exe scripts\run_tactical_gribstream_batch_smoke.py --apply-schema --days 70 --output-name batch_smoke_10w --api-min-interval-seconds 12 --api-max-attempts 2 --pause-on-429-seconds 300
```

Outputs:

```text
experiments/0214_tactical_h24n_gribstream_backfill/batch_smoke_10w/batch_results.csv
experiments/0214_tactical_h24n_gribstream_backfill/batch_smoke_10w/batch_summary.json
experiments/0214_tactical_h24n_gribstream_backfill/batch_smoke_10w/progress.json
experiments/0214_tactical_h24n_gribstream_backfill/logs/gribstream_batch_smoke_10w_api_events.jsonl
data/_pipeline_internal/raw/gribstream_tactical_batch_smoke_10w/
```

Measured result:

- Status: passed.
- Chunks processed: 96.
- API event statuses: 96 HTTP 200 events; no 4xx, 5xx, or 429 responses.
- `acquisition_chunk` statuses: 95 `completed`, 1 `completed_empty`.
- Rows returned/upserted to `nwp_tactical.forecast_wide`: 123,652.
- Estimated credits consumed: 112,047.
- Wall time of final completed run: 1,435.779 seconds, with the first 5 GFS chunks reused from raw cache after an earlier Windows path-length restart.
- DB tables written: `nwp_tactical.acquisition_chunk`, `nwp_tactical.raw_response_object`, `nwp_tactical.forecast_wide`.

Per-model returned rows:

```text
gfs: 20,700 rows, 22,425 estimated credits
gefsatmosmean: 6,720 rows, 4,480 estimated credits
gefsatmos: 17,360 rows, 17,360 estimated credits
ifsoper: 7,560 rows, 7,560 estimated credits
ifsenfo: 28,560 rows, 28,560 estimated credits
cwawrf15: 180 rows, 150 estimated credits
aifsoper: 4,200 rows, 2,800 estimated credits
aifsenfo: 14,280 rows, 14,280 estimated credits
aigfssfc: 3,660 rows, 1,220 estimated credits
aigfspres: 3,660 rows, 610 estimated credits
aigefssfc: 8,432 rows, 8,432 estimated credits
graphcast: 4,200 rows, 2,100 estimated credits
fourcastnetgfs: 4,140 rows, 2,070 estimated credits
nbmoc: 0 rows, 0 estimated credits
```

Interpretation:

- The tactical request shape works for all core models and all optional models except `nbmoc`, which remains empty/probe-only.
- The stored rows include 2m temperature, interval Tmax where available, dewpoint/RH, wind, pressure, precipitation, radiation, 850/925/700/500-level fields where available, and raw JSON provenance.
- Some archive-start or AI-model slices returned fewer rows than the maximum expected count. These were structural passes, not API failures: returned run times, valid times, locations, and members were in the requested sets.
- `cwawrf15` reused the same request identity as the earlier first-week probe, so its acquisition chunk id may show a first-week prefix even though the rows are included in this smoke's DB evidence.

## Tmax and Leakage Sanity Check

Measured against the 10-week batch smoke on 2026-06-25.

Critical result:

- Do not treat the raw `nwp_tactical.forecast_wide` table as directly feature-safe.
- Feature extraction must filter rows with `run_time_utc + publication_buffer <= target_date_hkt - 1 day at 15:00 HKT`.
- The 2026-06-25 sanity check used a 6-hour conservative publication/indexing buffer and found usable leakage-safe daily Tmax signals for most, but not all, models.

Tmax derivation status:

```text
gfs: OK, native interval_tmax_2m_k plus temperature_2m_k; 69 leakage-safe target days in smoke.
gefsatmosmean: OK, native interval_tmax_2m_k plus temperature_2m_k; 70 leakage-safe target days in smoke.
gefsatmos: OK, native member interval_tmax_2m_k at HKO center; 70 leakage-safe target days in smoke.
ifsoper: OK, derived from target-day 2m temperature snapshots; 70 leakage-safe target days in smoke.
ifsenfo: OK, derived per ensemble member from target-day 2m temperature snapshots; 70 leakage-safe target days in smoke.
cwawrf15: OK for rolling/prospective use only; 3 leakage-safe target days in smoke.
aifsoper: OK, derived from target-day 2m temperature snapshots; 70 leakage-safe target days in smoke.
aifsenfo: OK, derived per ensemble member from target-day 2m temperature snapshots; 70 leakage-safe target days in smoke.
aigfssfc: OK after archive-start edge filtering; 61 leakage-safe target days in smoke.
graphcast: OK, derived from target-day 2m temperature snapshots; 70 leakage-safe target days in smoke.
fourcastnetgfs: OK, derived from target-day 2m temperature snapshots; 69 leakage-safe target days in smoke.
aigfspres: NOT a Tmax source; no surface 2m temperature or interval Tmax, upper-air support only.
aigefssfc: BLOCKED as Tmax source; 8,432 rows returned but 8,308 had JSON null for member_temperature_2m_k, leaving only one non-null target day.
nbmoc: BLOCKED/probe-only; returned zero rows.
```

Daily Tmax examples after the leakage filter:

```text
gfs target 2021-03-24: 26.68 C from 288 rows = 24 valid hours x 12 locations.
gefsatmos target 2020-10-03: 28.43 C from 248 rows = 8 valid times x 31 members at HKO center.
ifsoper target 2024-03-01: 18.07 C from 96 rows = 8 valid times x 12 locations.
ifsenfo target 2024-03-03: 20.42 C from 408 rows = 8 valid times x 51 members at HKO center.
```

Backfill gate:

- Full tactical backfill is allowed for the OK models only if the downstream extractor enforces the H24N cutoff filter with the configured publication buffer.
- Do not include `aigfspres`, `aigefssfc`, or `nbmoc` as daily Tmax-producing sources unless a later selector/provider probe proves usable non-null 2m/Tmax coverage.

## Full Tactical Backfill Result

The 2026-06-25 full tactical run is documented here:

```text
documentation/T07_T12_FULL_TACTICAL_BACKFILL_20260625_RESULT.md
```

Summary:

- API-clean: no 429, 4xx, or 5xx responses in the API event log.
- Processed all 1,163 planned chunks.
- Wrote 1,964,157 rows to `nwp_tactical.forecast_wide`.
- Estimated credits consumed: 1,889,276.
- Final runner status: `failed` because of data-quality flags, not provider blockage.
- Main flags: recent `ifsenfo` chunks missing member `0`, empty `fourcastnetgfs` tail, and empty `nbmoc` probe.

## Deep Sanity Audit

Latest post-run audit:

```powershell
.\.venv\Scripts\python.exe scripts\audit_tactical_gribstream_deep_sanity.py --skip-file-hash
```

Outputs:

```text
documentation/T07_T12_DEEP_SANITY_AUDIT_20260625.md
experiments/0214_tactical_h24n_gribstream_backfill/full_tactical_backfill_ok_tmax/DEEP_SANITY_AUDIT_20260625.md
experiments/0214_tactical_h24n_gribstream_backfill/full_tactical_backfill_ok_tmax/deep_sanity_audit_20260625.json
```

Audit result:

- Full-run rows: 1,964,157.
- Total `forecast_wide` rows: 1,965,090.
- Non-full rows still mixed in `forecast_wide`: 933 old `batch_smoke_10w` `gefsatmos` rows.
- Full-run raw objects checked: 1,163.
- Missing raw files: 0.
- Raw byte-size mismatches: 0.
- HTTP errors in API log: 0.
- Downstream features must filter to `raw_response_object.object_uri LIKE '%full_tactical_backfill_ok_tmax%'` or purge/move the old smoke rows first.

Do not move the consolidated T07-T12 task to completed until those flags are resolved or explicitly accepted in the task record.

## Backfill Scope to Request From GribStream Support

Core:

- `gfs`: 2021-03-22T00:00:00Z through 2026-06-22T00:00:00Z
- `gefsatmosmean`: 2020-10-01T18:00:00Z through 2026-06-21T18:00:00Z
- `gefsatmos`: 2020-10-01T18:00:00Z through 2026-06-21T18:00:00Z
- `ifsoper`: 2024-02-28T18:00:00Z through 2026-06-21T18:00:00Z
- `ifsenfo`: 2024-03-01T18:00:00Z through 2026-06-21T18:00:00Z
- `cwawrf15`: rolling last three days plus prospective collection

Optional/shadow:

- `aifsoper`: 2025-02-25T18:00:00Z through 2026-06-21T18:00:00Z
- `aifsenfo`: 2025-07-02T18:00:00Z through 2026-06-21T18:00:00Z
- `aigfssfc`: 2026-04-16T18:00:00Z through 2026-06-21T18:00:00Z
- `aigfspres`: 2026-04-16T18:00:00Z through 2026-06-21T18:00:00Z
- `aigefssfc`: 2025-06-01T18:00:00Z through 2026-06-21T18:00:00Z
- `graphcast`: 2024-04-25T18:00:00Z through 2026-05-05T00:00:00Z
- `fourcastnetgfs`: 2024-05-02T18:00:00Z through 2026-03-01T12:00:00Z
- `nbmoc`: tiny HKO/marine probe only

## Provider Mail Draft

Tracked mail draft:

```text
documentation/GRIBSTREAM_SUPPORT_BACKFILL_ALLOWANCE_EMAIL_DRAFT.md
```
