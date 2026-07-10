# T07-T12 Full Tactical GribStream Backfill Result - 2026-06-25

## Bottom Line

The full tactical GribStream backfill run finished without provider-side API blockage.

It is not a clean-pass completion yet. The runner ended with `status: failed` because of data-quality flags, not because of HTTP errors, rate limits, or process failure.

## Command Shape

The final/resumed full run used one worker and conservative spacing:

```powershell
.\.venv\Scripts\python.exe scripts\run_tactical_gribstream_batch_smoke.py `
  --apply-schema `
  --days 10000 `
  --output-name full_tactical_backfill_ok_tmax `
  --datasets gfs,gefsatmosmean,gefsatmos,ifsoper,ifsenfo,cwawrf15,aifsoper,aifsenfo,aigfssfc,aigfspres,aigefssfc,graphcast,fourcastnetgfs,nbmoc `
  --api-min-interval-seconds 12 `
  --api-max-attempts 2 `
  --pause-on-429-seconds 300
```

Runtime discipline:

- One thread.
- `/api/v2/{dataset}/runs`.
- Exact model-run `timesList`.
- 12-second request spacing.
- Bounded retries.
- Raw-object resume enabled; existing raw objects were reused instead of being fetched again.

## Output Locations

```text
experiments/0214_tactical_h24n_gribstream_backfill/full_tactical_backfill_ok_tmax/progress.json
experiments/0214_tactical_h24n_gribstream_backfill/full_tactical_backfill_ok_tmax/batch_results.csv
experiments/0214_tactical_h24n_gribstream_backfill/full_tactical_backfill_ok_tmax/process_stdout.log
experiments/0214_tactical_h24n_gribstream_backfill/full_tactical_backfill_ok_tmax/process_stderr.log
experiments/0214_tactical_h24n_gribstream_backfill/logs/gribstream_full_tactical_backfill_ok_tmax_api_events.jsonl
data/_pipeline_internal/raw/gribstream_tactical_full_tactical_backfill_ok_tmax/
```

Database:

```text
postgresql://***:***@127.0.0.1:5432/hkg_tmax_research
nwp_tactical.acquisition_chunk
nwp_tactical.raw_response_object
nwp_tactical.forecast_wide
```

## API Result

API event log:

```text
api_fetched chunks: 946
raw_reused chunks: 217
HTTP error matches: 0
429 responses: 0
4xx responses: 0
5xx responses: 0
```

Every recorded live API response in the event log was HTTP 200.

## Run Totals

```text
planned_chunks: 1163
completed_chunks: 1163
status: failed
completed chunk statuses: 1153
completed_empty chunk statuses: 2
failed chunk statuses: 8
rows_returned: 1,964,157
forecast_wide_rows_upserted: 1,964,157
raw_response_objects: 1,258
estimated_credits_consumed: 1,889,276
expected_credits: 1,895,063
total_wall_seconds: 15,388.761
```

`status: failed` means "content/data-quality flags exist". It does not mean the GribStream API blocked, rate-limited, or rejected the run.

## Per-Dataset Result

Rows and credits below come from `batch_results.csv`. Run-time ranges are the chunk ranges in UTC.

| Dataset | Chunks | Rows | Estimated credits | First stored/covered run time | Last run time | Last status |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `gfs` | 138 | 575,004 | 622,921 | 2021-03-22T00:00:00Z requested; DB rows start at 2021-03-23T00:00:00Z | 2026-06-22T00:00:00Z | completed |
| `gefsatmosmean` | 68 | 200,436 | 133,624 | 2020-10-01T18:00:00Z | 2026-06-21T18:00:00Z | completed |
| `gefsatmos` | 418 | 516,891 | 516,925 | 2020-10-01T18:00:00Z | 2026-06-21T18:00:00Z | completed |
| `ifsoper` | 61 | 91,260 | 91,260 | 2024-02-28T18:00:00Z | 2026-06-21T18:00:00Z | completed |
| `ifsenfo` | 169 | 343,616 | 343,616 | 2024-03-01T18:00:00Z | 2026-06-21T18:00:00Z | failed |
| `cwawrf15` | 1 | 180 | 150 | 2026-06-22T18:00:00Z | 2026-06-24T18:00:00Z | completed |
| `aifsoper` | 49 | 28,884 | 19,256 | 2025-02-25T18:00:00Z | 2026-06-21T18:00:00Z | completed |
| `aifsenfo` | 71 | 72,270 | 72,420 | 2025-07-02T18:00:00Z | 2026-06-21T18:00:00Z | completed |
| `aigfssfc` | 3 | 3,660 | 1,220 | 2026-04-21T18:00:00Z | 2026-06-21T18:00:00Z | completed |
| `aigfspres` | 5 | 3,660 | 610 | 2026-04-21T18:00:00Z | 2026-06-21T18:00:00Z | completed |
| `aigefssfc` | 78 | 46,252 | 46,252 | 2025-06-01T18:00:00Z | 2026-06-21T18:00:00Z | completed |
| `graphcast` | 53 | 44,220 | 22,110 | 2024-04-25T18:00:00Z | 2026-05-04T18:00:00Z | completed |
| `fourcastnetgfs` | 48 | 37,824 | 18,912 | 2024-05-02T18:00:00Z | 2026-02-28T18:00:00Z requested; DB rows end at 2026-02-18T18:00:00Z | completed_empty |
| `nbmoc` | 1 | 0 | 0 | 2026-06-17T18:00:00Z requested | 2026-06-23T18:00:00Z requested | completed_empty |

## Data-Quality Flags

### `ifsenfo`

Eight recent `ifsenfo` chunks returned HTTP 200 and persisted data, but failed the sanity gate because member `0` was missing.

| Chunk | Run-time window UTC | Rows | Expected rows | Issue |
| ---: | --- | ---: | ---: | --- |
| 847 | 2026-05-15T18:00:00Z to 2026-05-19T18:00:00Z | 2,000 | 2,040 | `missing_members=[0]` |
| 848 | 2026-05-20T18:00:00Z to 2026-05-24T18:00:00Z | 2,000 | 2,040 | `missing_members=[0]` |
| 849 | 2026-05-25T18:00:00Z to 2026-05-29T18:00:00Z | 2,000 | 2,040 | `missing_members=[0]` |
| 850 | 2026-05-30T18:00:00Z to 2026-06-03T18:00:00Z | 2,000 | 2,040 | `missing_members=[0]` |
| 851 | 2026-06-04T18:00:00Z to 2026-06-08T18:00:00Z | 2,000 | 2,040 | `missing_members=[0]` |
| 852 | 2026-06-09T18:00:00Z to 2026-06-13T18:00:00Z | 2,000 | 2,040 | `missing_members=[0]` |
| 853 | 2026-06-14T18:00:00Z to 2026-06-18T18:00:00Z | 2,000 | 2,040 | `missing_members=[0]` |
| 854 | 2026-06-19T18:00:00Z to 2026-06-21T18:00:00Z | 1,200 | 1,224 | `missing_members=[0]` |

This is a provider-content/availability mismatch for the expected ensemble member set, not an HTTP failure.

### `fourcastnetgfs`

The tail request for `2026-02-19T18:00:00Z` through `2026-02-28T18:00:00Z` returned HTTP 200 with zero rows.

Persisted DB rows for `fourcastnetgfs` end at `2026-02-18T18:00:00Z`.

### `nbmoc`

The probe request for `2026-06-17T18:00:00Z` through `2026-06-23T18:00:00Z` returned HTTP 200 with zero rows.

`nbmoc` remains probe-only and should not be treated as a usable HKO Tmax source unless a later provider/selector probe proves non-empty HKO-domain coverage.

## Downstream Use Rules

Do not query `nwp_tactical.forecast_wide` naively for modeling.

Feature extraction must enforce the H24N leakage-safe cutoff:

```text
run_time_utc + publication_buffer <= target_date_hkt - 1 day at 15:00 HKT
```

The current documented conservative buffer is 6 hours. The detailed leakage rule lives in:

```text
documentation/GRIBSTREAM_TMAX_LEAKAGE_SAFETY.md
```

The post-run deep sanity audit lives in:

```text
documentation/T07_T12_DEEP_SANITY_AUDIT_20260625.md
experiments/0214_tactical_h24n_gribstream_backfill/full_tactical_backfill_ok_tmax/DEEP_SANITY_AUDIT_20260625.md
experiments/0214_tactical_h24n_gribstream_backfill/full_tactical_backfill_ok_tmax/deep_sanity_audit_20260625.json
```

Current audit result: the full-run data is API-clean and structurally consistent, but `forecast_wide` still contains 933 older `batch_smoke_10w` `gefsatmos` rows, so modeling queries must filter to `raw_response_object.object_uri LIKE '%full_tactical_backfill_ok_tmax%'` or purge/move those old rows first.

Do not treat these as daily Tmax-producing sources without a later positive probe:

```text
aigfspres
aigefssfc
nbmoc
```

`aigfspres` is upper-air support only in the current probe. `aigefssfc` returned mostly null member 2m temperature in prior sanity checks. `nbmoc` returned zero rows.

## Completion Decision

Do not move `T07_T12_tactical_h24n_gribstream_backfill` to completed from row counts alone.

It can be completed only after one of these happens:

1. The `ifsenfo` member-0 gap is resolved or explicitly accepted as a known provider-content gap.
2. `fourcastnetgfs` tail emptiness is resolved or accepted as an archive-end limitation.
3. `nbmoc` is either removed from the tactical scope or documented as probe-only/blocked.
4. The downstream feature extractor enforces the H24N cutoff filter before modeling.
