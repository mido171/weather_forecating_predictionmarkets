# GribStream Acquisition Specification

## Objective

Create a resumable, exact-run historical and prospective acquisition system that stores every relevant model run, selector, location, member, valid time, and provenance field required for HKG T+24 forecasting. Bulk data acquisition may begin only after catalog discovery, quota sizing, coverage testing, and database migrations pass.

## API endpoint policy

- Use `/api/v2/<model>/runs` for research backfills and run-to-run history.
- Do not use an unrestricted `/timeseries` response as the canonical historical source because it selects the shortest eligible lead for each valid time.
- `asOf` is permitted as an additional run-time guard, never as sole availability proof.
- Discover exact `name`, `level`, and `info` selectors from the current public catalog. Never guess selector casing or levels.
- Prefer NDJSON streaming and gzip for large responses.

## Required raw fields

Every normalized value must retain:

```text
provider
model_code
model_version_or_selector_snapshot_id
forecasted_at_utc       # model run time returned by GribStream
forecasted_time_utc     # valid time returned by GribStream
lead_minutes
member_number
latitude
longitude
location_id
variable_id
native_name
native_level
native_info
native_unit
value
request_id
response_object_id
retrieved_at_utc
first_seen_at_utc       # prospective collector only
availability_contract_id
eligibility_grade
```

## Historical and prospective availability

Historical backfill records begin at `C_RUN_TIME_ONLY`. They may become `B_PROVIDER_SCHEDULE_PROVEN` only after T16 stores authoritative dissemination evidence and an approved conservative buffer. Prospective polling records can reach `A_EXACT_FIRST_SEEN` when the collector records the first successful response.

## Acquisition tiers

### Core deterministic point-and-patch tier

Models: `gfs`, `ifsoper`, later `aifsoper`, `aigfssfc`, `aigfspres`, `graphcast`, `fourcastnetgfs`, `cwawrf15`.

Collect:

- all target/station/reference coordinates;
- local Hong Kong patch;
- target-day hourly trajectory;
- 0–84 hour leads for operational flexibility;
- all available cycles, not only the cycle expected to be selected at 15:00 HKT.

### Core ensemble tier

Models: `gefsatmos`, `ifsenfo`, `aigefssfc`, `aigefspres`, `aifsenfo`.

Collect all members at selected coordinates for the P0/P1 variable subset. Use ensemble mean products or smaller spatial grids for broader patches.

### Synoptic context tier

Collect a compressed set of fields over approximately 18–28°N and 108–122°E. Store raw grid responses in immutable Parquet/object storage, not millions of unindexed JSON blobs. Persist derived spatial summaries to Postgres.

## Chunking and idempotency

Chunk requests by:

- model;
- run-time interval;
- variable group;
- coordinate group or grid;
- ensemble member batch;
- lead-time band.

Derive a deterministic request hash from canonical JSON. A completed request with a matching hash and valid response checksum must not be downloaded twice. Partial/failed requests must be resumable.

## Required failure behavior

- 429: honor `Retry-After`, record quota state, retry with bounded exponential backoff.
- 400: quarantine request definition and stop that selector group.
- 401: fail without logging token content.
- 5xx/network: retry; preserve attempt log.
- Empty response: distinguish no coverage, no matching run, selector error, and temporary source failure.

## Backfill date ranges

Use `DATA_SOURCE_AND_DATE_RANGE_PLAN.csv` and refresh dates from the live catalog. Never request dates before a variable/model’s catalog introduction date. CWA WRF and other short-window models must be polled immediately because old runs disappear.
