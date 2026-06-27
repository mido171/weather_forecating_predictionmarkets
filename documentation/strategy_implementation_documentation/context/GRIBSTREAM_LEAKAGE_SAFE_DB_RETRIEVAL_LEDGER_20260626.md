# GribStream Leakage-Safe DB Retrieval Ledger - 2026-06-26

This is the implementation ledger for retrieving GribStream forecast data from PostgreSQL without forward-looking leakage.

It is intentionally detailed. Any future feature extractor, experiment, training script, scoring script, dashboard, or manual SQL query that uses `nwp_tactical.forecast_wide` for HKG Tmax research must follow this document.

## Bottom Line

Do not query `nwp_tactical.forecast_wide` directly and group by `target_date_hkt`.

That table is a raw normalized forecast table. It contains rows that are structurally valid but not automatically safe for an H24N decision.

The required safe retrieval sequence is:

1. Scope rows to the real full tactical backfill source.
2. Apply the H24N decision cutoff with a conservative publication buffer.
3. Exclude sources that are not usable daily Tmax sources.
4. Build daily features only after those filters.
5. Preserve location and ensemble-member semantics when aggregating.

The core H24N leakage rule is:

```text
run_time_utc + publication_buffer <= target_date_hkt - 1 day at 15:00 HKT
```

For the current audited backfill, the publication buffer is:

```text
6 hours
```

Important: this 6-hour buffer is a conservative safety assumption, not a confirmed GribStream-provided availability SLA. It is the current guardrail we are using so the historical backtest does not accidentally treat model-run timestamps as instant availability timestamps.

Future work should test and confirm this assumption through the GribStream API/provider semantics before treating it as final production policy. In particular, a later task should compare `/runs` retrieval behavior, `/timeseries` `asOf` behavior where applicable, and provider guidance for model availability/indexing delay.

Equivalent UTC rule for Hong Kong:

```text
run_time_utc + 6 hours <= target_date_hkt - 1 day at 07:00 UTC
```

Hong Kong is UTC+8 and does not use daylight saving time.

## Source Of Truth

Database:

```text
postgresql://***:***@127.0.0.1:5432/hkg_tmax_research
```

Main schema:

```text
nwp_tactical
```

Main tables:

| Table | Purpose |
| --- | --- |
| `nwp_tactical.forecast_wide` | Normalized model forecast rows used for feature extraction after filtering. |
| `nwp_tactical.raw_response_object` | Raw GribStream response ledger. This is required for source-scope filtering. |
| `nwp_tactical.acquisition_chunk` | Acquisition planning/progress chunks. |
| `nwp_tactical.validation_issue` | Validation issues emitted during acquisition/audit. |

Important docs:

```text
documentation/strategy_implementation_documentation/GRIBSTREAM_FETCHED_DATA_INVENTORY_20260626.md
documentation/T07_T12_DEEP_SANITY_AUDIT_20260625.md
documentation/T07_T12_FULL_TACTICAL_BACKFILL_20260625_RESULT.md
documentation/GRIBSTREAM_TMAX_LEAKAGE_SAFETY.md
```

Machine-readable audit outputs:

```text
experiments/0214_tactical_h24n_gribstream_backfill/full_tactical_backfill_ok_tmax/deep_sanity_audit_20260625.json
experiments/0214_tactical_h24n_gribstream_backfill/full_tactical_backfill_ok_tmax/DEEP_SANITY_AUDIT_20260625.md
```

## What One Row Means

Each row in `nwp_tactical.forecast_wide` is one forecast point:

```text
dataset/model
+ acquisition version
+ target Hong Kong date
+ model run time
+ forecast valid time
+ forecast lead
+ location
+ ensemble member
+ weather values
+ source raw object pointer
```

This is forecast data, not observed actual weather.

The realized HKO daily maximum temperature label must come from the HKO target/settlement tables, not from GribStream.

## Key Time Fields

| Column | Meaning | Leakage implication |
| --- | --- | --- |
| `run_time_utc` | The model cycle/run timestamp. GribStream calls this the forecast run time / `forecasted_at` in `/runs` style data. | This is the key timestamp for deciding whether a row could have been known before the decision cutoff. |
| `valid_time_utc` | The time the forecast value applies to. GribStream calls this the forecasted/valid time. | This determines which Hong Kong target date the row belongs to. It does not prove the row was available before the decision cutoff. |
| `lead_hours` | `valid_time_utc - run_time_utc`, in hours. | Useful for sanity checks and lead-specific features. |
| `target_date_hkt` | Hong Kong local calendar date derived from `valid_time_utc`. | This is the market target date. It is not an availability timestamp. |
| `cutoff_id` | Current tactical cutoff family. Current full tactical backfill uses `H24N`. | Must be `H24N` for this ledger. |
| `source_response_object_id` | Pointer to `raw_response_object`. | Required to filter out old smoke rows and keep only the intended full-run pull. |

## `run_time_utc` Versus `valid_time_utc`

These two fields answer different questions.

Example:

```text
run_time_utc   = 2021-03-23 00:00Z
valid_time_utc = 2021-03-24 06:00Z
```

Meaning:

- The model was initialized at `2021-03-23 00:00Z`.
- One forecasted value from that model run applies to `2021-03-24 06:00Z`.
- The lead is 30 hours.
- The value belongs to target date `2021-03-24` in Hong Kong.

Leakage decision:

- Use `run_time_utc`, not `valid_time_utc`, to decide whether the forecast row was allowed at the decision cutoff.

## Why A Publication Buffer Exists

The model run timestamp is not the same as guaranteed user availability.

A model can have:

```text
model run time
processing time
provider indexing time
API availability time
```

For historical `/runs` data, GribStream gives us the run and valid timestamps, but it does not give a stable per-row historical `available_at` timestamp that proves the row was available at the exact moment in the past.

Because of that, this project uses a conservative publication/indexing buffer.

Current audited setting:

```text
publication_buffer = 6 hours
```

This is a deliberately cautious project assumption. It was chosen as a safe zone for the current audit and feature-gating work because no stable per-row historical `available_at` timestamp is present in the stored `/runs` rows.

It has not yet been proven as the exact GribStream availability delay for every model family. The correct future hardening step is to run a GribStream API confirmation pass, preferably with provider guidance, to test whether each model's practical availability is safely covered by 6 hours or whether model-specific buffers are needed.

Interpretation:

```text
Even if a model run says 00Z, we only allow it into H24N features as if it became usable at 06Z.
```

This is intentionally conservative. It protects the backtest from pretending that a model run was usable immediately at its model timestamp.

## H24N Decision Cutoff

The H24N decision policy is:

```text
15:00 HKT on T-1
```

For a Hong Kong target date `T`, this means:

```text
decision cutoff = T - 1 day at 15:00 HKT
```

In UTC:

```text
decision cutoff = T - 1 day at 07:00 UTC
```

With the current 6-hour publication buffer:

```text
latest allowed run_time_utc = T - 1 day at 01:00 UTC
```

Any run later than that is unsafe for H24N features.

## Concrete UTC Examples

### Example 1: Safe GFS Row

Target date:

```text
2021-03-24 HKT
```

H24N decision cutoff:

```text
2021-03-23 07:00Z
```

Publication buffer:

```text
6 hours
```

Latest allowed model run time:

```text
2021-03-23 01:00Z
```

GFS model run:

```text
run_time_utc = 2021-03-23 00:00Z
```

Safety check:

```text
2021-03-23 00:00Z + 6 hours = 2021-03-23 06:00Z
2021-03-23 06:00Z <= 2021-03-23 07:00Z
```

Result:

```text
Safe.
```

### Example 2: Unsafe GFS Row

Target date:

```text
2021-03-24 HKT
```

H24N decision cutoff:

```text
2021-03-23 07:00Z
```

Publication buffer:

```text
6 hours
```

GFS model run:

```text
run_time_utc = 2021-03-23 06:00Z
```

Safety check:

```text
2021-03-23 06:00Z + 6 hours = 2021-03-23 12:00Z
2021-03-23 12:00Z > 2021-03-23 07:00Z
```

Result:

```text
Unsafe. Do not use for H24N features for 2021-03-24.
```

### Example 3: Safe GEFS Ensemble Row

Target date:

```text
2026-06-23 HKT
```

H24N decision cutoff:

```text
2026-06-22 07:00Z
```

Publication buffer:

```text
6 hours
```

Latest allowed model run time:

```text
2026-06-22 01:00Z
```

GEFS model run:

```text
run_time_utc = 2026-06-21 18:00Z
```

Safety check:

```text
2026-06-21 18:00Z + 6 hours = 2026-06-22 00:00Z
2026-06-22 00:00Z <= 2026-06-22 07:00Z
```

Result:

```text
Safe.
```

## Current Full-Run DB State

Verified against local PostgreSQL on 2026-06-26.

Raw object ledger:

| Source scope | Raw objects | Raw row-count sum | Bytes |
| --- | ---: | ---: | ---: |
| `batch_smoke_10w` | 95 | 123,652 | 3,129,036 |
| `first_week` | 13 | 11,796 | 298,866 |
| `full_tactical_backfill_ok_tmax` | 1,163 | 1,964,157 | 56,488,866 |
| `smoke` | 14 | 1,728 | 50,853 |

Rows currently visible in `nwp_tactical.forecast_wide`:

| Source scope | Dataset | Rows |
| --- | --- | ---: |
| `batch_smoke_10w` | `gefsatmos` | 933 |
| `full_tactical_backfill_ok_tmax` | all full-run datasets | 1,964,157 |

Important:

```text
forecast_wide contains 933 old batch-smoke rows.
```

Those rows are not part of the completed full tactical backfill. Therefore, every production/modeling query must join to `raw_response_object` and filter to:

```text
full_tactical_backfill_ok_tmax
```

until those old smoke rows are purged or moved away from the modeling surface.

## Mandatory Source-Scope Filter

Use this first:

```sql
SELECT fw.*
FROM nwp_tactical.forecast_wide fw
JOIN nwp_tactical.raw_response_object r
  ON r.response_object_id = fw.source_response_object_id
WHERE r.object_uri LIKE '%full_tactical_backfill_ok_tmax%';
```

Do not replace this with a date filter.

Reason:

- source scope tells us which acquisition run the row came from;
- date range does not distinguish full-run rows from old smoke/test rows.

## Mandatory H24N Leakage Filter

PostgreSQL expression for the decision cutoff:

```sql
((fw.target_date_hkt - 1) + time '15:00') AT TIME ZONE 'Asia/Hong_Kong'
```

This returns the UTC instant corresponding to 15:00 HKT on the prior Hong Kong day.

Current audited safety filter:

```sql
fw.run_time_utc + interval '6 hours'
  <= ((fw.target_date_hkt - 1) + time '15:00') AT TIME ZONE 'Asia/Hong_Kong'
```

Do not write this as a loose date comparison.

Do not use `valid_time_utc` as the availability test.

Do not omit the publication buffer.

## Mandatory Usable-Tmax Source Filter

Usable daily Tmax sources for the current full tactical pull:

```text
gfs
gefsatmosmean
gefsatmos
ifsoper
ifsenfo
aifsoper
aifsenfo
aigfssfc
graphcast
fourcastnetgfs
cwawrf15
```

Do not use these as daily Tmax sources right now:

```text
nbmoc
aigfspres
aigefssfc
```

Reasons:

| Dataset | Current status |
| --- | --- |
| `nbmoc` | Probe returned HTTP 200 but zero rows. Not usable. |
| `aigfspres` | Upper-air/pressure support data only in this pull. No daily surface Tmax candidate. |
| `aigefssfc` | Rows exist and are leakage-safe, but only 67 of 373 target days had usable 2m/Tmax candidate values. Blocked as a daily Tmax source until a later selector/provider probe fixes coverage. |

## Canonical Safe Row Query

Use this shape for row-level feature extraction:

```sql
WITH full_scope AS (
  SELECT fw.*
  FROM nwp_tactical.forecast_wide fw
  JOIN nwp_tactical.raw_response_object r
    ON r.response_object_id = fw.source_response_object_id
  WHERE r.object_uri LIKE '%full_tactical_backfill_ok_tmax%'
),
h24n_safe AS (
  SELECT *
  FROM full_scope fw
  WHERE fw.run_time_utc + interval '6 hours'
      <= ((fw.target_date_hkt - 1) + time '15:00') AT TIME ZONE 'Asia/Hong_Kong'
),
usable_tmax_sources AS (
  SELECT *
  FROM h24n_safe
  WHERE dataset_code NOT IN ('nbmoc', 'aigfspres', 'aigefssfc')
)
SELECT *
FROM usable_tmax_sources;
```

This is the minimum safe retrieval pattern.

Any code implementation should expose the publication buffer as a parameter, defaulting to 6 hours for the current audited H24N backfill.

## Candidate Temperature Field

For daily Tmax candidate extraction, the current practical row-level candidate is:

```sql
COALESCE(interval_tmax_2m_k, temperature_2m_k)
```

In Celsius:

```sql
COALESCE(interval_tmax_2m_k, temperature_2m_k) - 273.15
```

Interpretation:

- `interval_tmax_2m_k` is preferred where the model/provider returned it.
- `temperature_2m_k` is the fallback where only instantaneous 2m temperature exists.

Important caution:

Some model/provider interval semantics need careful treatment. The audit found rows where `interval_tmax_2m_k < temperature_2m_k` for some GFS/GEFS-mean rows. That does not make the whole source unusable, but it means feature code must not blindly assume row-level interval Tmax always dominates instantaneous temperature.

## Daily Tmax Feature Shape

There is no single universal aggregation that fits every model family.

Correct feature extraction must respect three dimensions:

```text
target date
location
ensemble member
```

Deterministic 12-location models:

```text
dataset + target_date_hkt + 12 locations + member 0
```

Full-member ensemble models:

```text
dataset + target_date_hkt + HKO center + many members
```

Do not accidentally collapse the 12 locations and ensemble members into one unexplained number unless the feature definition explicitly says that is intended.

Examples of acceptable feature families:

| Feature family | Description |
| --- | --- |
| Per-location deterministic max | Daily max for each of the 12 Hong Kong stencil locations. |
| Spatial max | Max across the 12 deterministic locations for a target date. Useful as a hot-spot proxy, but must be named as spatial max. |
| HKO-center deterministic max | Daily max at the HKO center location only. |
| Ensemble member daily max | Daily max per ensemble member. |
| Ensemble distribution features | Mean, median, quantiles, spread, or bucket probabilities across member-level daily maxima. |

## Example: GFS Daily Max For One Target Date

Validated against the live DB on 2026-06-26.

Target date:

```text
2021-03-24 HKT
```

Safe run used:

```text
2021-03-23 00:00Z
```

Rows used:

```text
288 rows = 24 valid hours x 12 locations x 1 member
```

Derived daily maximum:

```text
26.68 C
```

SQL:

```sql
WITH safe_rows AS (
  SELECT
    fw.*,
    (COALESCE(interval_tmax_2m_k, temperature_2m_k) - 273.15) AS tmax_candidate_c
  FROM nwp_tactical.forecast_wide fw
  JOIN nwp_tactical.raw_response_object r
    ON r.response_object_id = fw.source_response_object_id
  WHERE r.object_uri LIKE '%full_tactical_backfill_ok_tmax%'
    AND fw.dataset_code = 'gfs'
    AND fw.target_date_hkt = DATE '2021-03-24'
    AND fw.run_time_utc + interval '6 hours'
       <= ((fw.target_date_hkt - 1) + time '15:00') AT TIME ZONE 'Asia/Hong_Kong'
    AND COALESCE(interval_tmax_2m_k, temperature_2m_k) IS NOT NULL
)
SELECT
  dataset_code,
  target_date_hkt,
  min(run_time_utc AT TIME ZONE 'UTC') AS min_run_time_utc,
  max(run_time_utc AT TIME ZONE 'UTC') AS max_run_time_utc,
  count(*) AS rows_used,
  count(DISTINCT valid_time_utc) AS valid_times,
  count(DISTINCT location_code) AS locations,
  count(DISTINCT member_number) AS members,
  round(max(tmax_candidate_c)::numeric, 2) AS daily_max_c
FROM safe_rows
GROUP BY dataset_code, target_date_hkt;
```

Expected result:

| Dataset | Target date | Run time UTC | Rows | Valid hours | Locations | Members | Daily max C |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `gfs` | `2021-03-24` | `2021-03-23 00:00Z` | 288 | 24 | 12 | 1 | 26.68 |

## Safe Rows And Tmax Candidate Counts

Verified against local PostgreSQL on 2026-06-26.

Safe rows use:

```text
full_tactical_backfill_ok_tmax source scope
+ 6-hour H24N leakage filter
```

Safe Tmax candidate rows additionally require:

```text
dataset not in ('nbmoc', 'aigfspres', 'aigefssfc')
+ COALESCE(interval_tmax_2m_k, temperature_2m_k) IS NOT NULL
```

| Dataset | Full-run rows | Safe rows, 6h | Unsafe rows, 6h | Target days in full run | Safe Tmax candidate rows | Target days with candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `aifsenfo` | 72,270 | 72,270 | 0 | 355 | 72,270 | 355 |
| `aifsoper` | 28,884 | 23,100 | 5,784 | 483 | 23,100 | 482 |
| `aigefssfc` | 46,252 | 46,252 | 0 | 373 | blocked | blocked |
| `aigfspres` | 3,660 | 2,928 | 732 | 63 | blocked | blocked |
| `aigfssfc` | 3,660 | 2,928 | 732 | 63 | 2,928 | 61 |
| `cwawrf15` | 180 | 144 | 36 | 4 | 144 | 3 |
| `fourcastnetgfs` | 37,824 | 30,252 | 7,572 | 648 | 30,252 | 631 |
| `gefsatmos` | 516,891 | 516,891 | 0 | 2,085 | 516,891 | 2,085 |
| `gefsatmosmean` | 200,436 | 200,436 | 0 | 2,088 | 200,436 | 2,088 |
| `gfs` | 575,004 | 552,000 | 23,004 | 1,919 | 551,808 | 1,918 |
| `graphcast` | 44,220 | 35,376 | 8,844 | 741 | 35,376 | 737 |
| `ifsenfo` | 343,616 | 343,616 | 0 | 843 | 343,616 | 843 |
| `ifsoper` | 91,260 | 81,120 | 10,140 | 846 | 81,120 | 845 |

The difference between "safe rows" and "safe Tmax candidate rows" is expected. Some safe rows belong to boundary target dates or rows where the selected temperature/Tmax candidate is null.

## Dataset-Specific Current Gates

| Dataset | Gate |
| --- | --- |
| `gfs` | Usable after source + H24N leakage filters. |
| `gefsatmosmean` | Usable after source + H24N leakage filters. |
| `gefsatmos` | Usable after source + H24N leakage filters; HKO-center ensemble members only. |
| `ifsoper` | Usable after source + H24N leakage filters. |
| `ifsenfo` | Usable after source + H24N leakage filters, but recent chunks missed member `0`; downstream ensemble code must tolerate missing member IDs. |
| `aifsoper` | Optional usable source after source + H24N leakage filters. |
| `aifsenfo` | Optional usable source after source + H24N leakage filters. |
| `aigfssfc` | Optional usable source over a short recent range only. |
| `graphcast` | Optional usable source through observed archive. |
| `fourcastnetgfs` | Optional usable source only through observed archive end; tail after `2026-02-18 18:00Z` returned empty. |
| `cwawrf15` | Rolling/prospective/live source only; very short historical range in this pull. |
| `aigfspres` | Do not use as daily Tmax source. Support/upper-air only. |
| `aigefssfc` | Do not use as daily Tmax source until coverage is fixed. |
| `nbmoc` | Do not use. Empty result. |

## Structural Integrity Checks

The following checks were verified clean for the full tactical source scope on 2026-06-26:

| Check | Result |
| --- | ---: |
| Rows with `cutoff_id <> 'H24N'` | 0 |
| Rows with `acquisition_version <> 'tactical_h24n_v1'` | 0 |
| Rows with empty `raw_values_jsonb` | 0 |
| Rows missing `source_response_object_id` | 0 |
| Target-date mismatches versus `valid_time_utc` in Hong Kong | 0 |
| Lead-hour mismatches | 0 |

Structural cleanliness does not remove the need for leakage filtering.

## Validation Query: Source Scopes

Use this to confirm old smoke/test rows are not accidentally mixed into modeling:

```sql
SELECT
  CASE
    WHEN r.object_uri LIKE '%full_tactical_backfill_ok_tmax%' THEN 'full_tactical_backfill_ok_tmax'
    WHEN r.object_uri LIKE '%batch_smoke_10w%' THEN 'batch_smoke_10w'
    WHEN r.object_uri LIKE '%first_week%' THEN 'first_week'
    WHEN r.object_uri LIKE '%smoke%' THEN 'smoke'
    ELSE 'other'
  END AS source_scope,
  fw.dataset_code,
  count(*) AS rows
FROM nwp_tactical.forecast_wide fw
LEFT JOIN nwp_tactical.raw_response_object r
  ON r.response_object_id = fw.source_response_object_id
GROUP BY 1, 2
ORDER BY 1, 2;
```

Expected current modeling-table result:

```text
batch_smoke_10w / gefsatmos = 933 rows
full_tactical_backfill_ok_tmax / all full-run datasets = 1,964,157 rows
```

## Validation Query: Unsafe Rows

Use this to prove why raw grouping is not safe:

```sql
WITH full_scope AS (
  SELECT fw.*
  FROM nwp_tactical.forecast_wide fw
  JOIN nwp_tactical.raw_response_object r
    ON r.response_object_id = fw.source_response_object_id
  WHERE r.object_uri LIKE '%full_tactical_backfill_ok_tmax%'
)
SELECT
  dataset_code,
  count(*) AS total_rows,
  count(*) FILTER (
    WHERE run_time_utc + interval '6 hours'
       <= ((target_date_hkt - 1) + time '15:00') AT TIME ZONE 'Asia/Hong_Kong'
  ) AS safe_rows_6h,
  count(*) FILTER (
    WHERE NOT (
      run_time_utc + interval '6 hours'
        <= ((target_date_hkt - 1) + time '15:00') AT TIME ZONE 'Asia/Hong_Kong'
    )
  ) AS unsafe_rows_6h
FROM full_scope
GROUP BY dataset_code
ORDER BY dataset_code;
```

If any downstream extractor does not apply the safe-row predicate, it can leak for deterministic/AI deterministic families.

## Validation Query: Daily Feature Coverage

Use this to check which models have usable safe daily Tmax candidates:

```sql
WITH full_scope AS (
  SELECT fw.*
  FROM nwp_tactical.forecast_wide fw
  JOIN nwp_tactical.raw_response_object r
    ON r.response_object_id = fw.source_response_object_id
  WHERE r.object_uri LIKE '%full_tactical_backfill_ok_tmax%'
),
safe_tmax_source AS (
  SELECT *
  FROM full_scope fw
  WHERE fw.run_time_utc + interval '6 hours'
     <= ((fw.target_date_hkt - 1) + time '15:00') AT TIME ZONE 'Asia/Hong_Kong'
    AND fw.dataset_code NOT IN ('nbmoc', 'aigfspres', 'aigefssfc')
    AND COALESCE(fw.interval_tmax_2m_k, fw.temperature_2m_k) IS NOT NULL
)
SELECT
  dataset_code,
  count(*) AS safe_tmax_candidate_rows,
  count(DISTINCT target_date_hkt) AS target_days_with_candidates,
  min(target_date_hkt) AS min_target_date_hkt,
  max(target_date_hkt) AS max_target_date_hkt,
  count(DISTINCT location_code) AS locations,
  count(DISTINCT member_number) AS members
FROM safe_tmax_source
GROUP BY dataset_code
ORDER BY dataset_code;
```

## Required Assertions In Future Code

Any new DB extractor or modeling dataset builder must assert these conditions:

1. Source scope is `full_tactical_backfill_ok_tmax`, unless a newer documented full-run scope replaces it.
2. `cutoff_id = 'H24N'`.
3. `acquisition_version = 'tactical_h24n_v1'`, unless a newer version is explicitly documented.
4. The publication buffer is explicitly configured, not hard-coded invisibly.
5. Default publication buffer is 6 hours for this audited dataset.
6. Rows satisfy the H24N safe predicate before aggregation.
7. `nbmoc`, `aigfspres`, and `aigefssfc` are excluded from daily Tmax-source features by default.
8. Null candidate temperatures are excluded from Tmax aggregation.
9. Ensemble member IDs are not assumed contiguous for every date/model.
10. Deterministic 12-location rows are not silently treated as one physical location unless the feature is explicitly a spatial aggregate.
11. Feature outputs store enough provenance to reconstruct dataset, target date, cutoff, buffer, source scope, model run times used, locations used, and member policy.

## Recommended Feature Output Metadata

Every derived feature table or parquet file should include:

| Field | Required meaning |
| --- | --- |
| `target_date_hkt` | Hong Kong target date. |
| `cutoff_id` | `H24N` for this flow. |
| `source_scope` | `full_tactical_backfill_ok_tmax` or documented successor. |
| `publication_buffer_hours` | Current default `6`. |
| `dataset_code` | Model/source. |
| `feature_name` | Explicit feature name. |
| `feature_value` | Numeric value. |
| `feature_unit` | Example: `C`, `K`, `mps`, `pct`. |
| `aggregation_policy` | Example: `hko_center`, `spatial_max_12pt`, `member_quantile`, `member_mean`. |
| `location_policy` | Example: `12pt_hko_stencil`, `hko_center_only`. |
| `member_policy` | Example: `deterministic_member_0`, `members_0_30`, `members_0_50_available`. |
| `min_run_time_utc` | Earliest model run used. |
| `max_run_time_utc` | Latest model run used. |
| `row_count` | Number of forecast rows used. |
| `safe_filter_applied` | Boolean, must be true. |
| `created_at_utc` | Feature build timestamp. |

## Common Failure Modes

### Failure 1: Grouping Raw Rows By Target Date

Bad pattern:

```sql
SELECT target_date_hkt, max(temperature_2m_k)
FROM nwp_tactical.forecast_wide
GROUP BY target_date_hkt;
```

Why bad:

- includes old smoke rows;
- includes unsafe runs;
- mixes models;
- mixes locations;
- mixes ensemble members;
- does not define a decision cutoff.

### Failure 2: Using Valid Time As Availability

Bad logic:

```text
valid_time_utc <= decision_cutoff
```

Why bad:

- `valid_time_utc` is the forecasted weather time;
- it does not say when the forecast became available;
- future weather valid times are exactly what forecast models predict.

Correct availability proxy:

```text
run_time_utc + publication_buffer <= decision_cutoff
```

### Failure 3: Assuming Historical Availability From Today's API Response

Bad assumption:

```text
The row exists in GribStream today, so it must have been available at that exact historical cutoff.
```

Why bad:

- the API can return historical data long after the fact;
- existence today does not prove exact point-in-time availability in the past;
- the buffer is the guardrail.

### Failure 4: Treating All 12 Locations As One Location

Bad explanation:

```text
288 GFS rows means 288 forecasts for one location.
```

Correct explanation:

```text
288 rows = 24 valid hours x 12 locations x 1 member.
```

Those 12 locations are distinct stencil points around Hong Kong. A spatial aggregate can use them, but it must be named and intentional.

### Failure 5: Treating Blocked Sources As Model Inputs

Do not let a broad SQL query accidentally include:

```text
nbmoc
aigfspres
aigefssfc
```

`aigefssfc` is especially dangerous because it has rows, but poor usable Tmax coverage.

## GribStream `asOf` Note

GribStream has `asOf` semantics for timeseries-style access. That can be useful when asking the API what would be available as of a given time.

This tactical backfill was built from `/runs` style historical forecast-run pulls. For this stored data, the DB retrieval must enforce the equivalent availability discipline itself:

```text
run_time_utc + publication_buffer <= decision cutoff
```

Do not assume `asOf` was applied to these already-stored `/runs` rows.

## Open Decisions Before Production Modeling

These are not blockers for safe retrieval, but they must be decided before final modeling:

1. Whether the default 6-hour publication buffer is sufficient for every model, or whether sensitivity runs should use 9-hour and 12-hour buffers.
2. Whether GribStream API confirmation can prove the correct availability policy more directly. The test should compare the stored `/runs` approach against GribStream `asOf`/availability semantics where applicable and should document any model-specific availability delay discovered.
3. Whether `ifsenfo` missing member `0` in recent chunks is acceptable, should be imputed, or should trigger provider follow-up.
4. Whether `fourcastnetgfs` should be treated as ending at `2026-02-18 18:00Z` or probed again later.
5. Whether `cwawrf15` should be used only prospectively or excluded from historical training due to very short coverage.
6. Whether any blocked source can be revived by a new provider selector probe.

## Completion Gate For Future Work

Before any future task claims "features are leakage-safe", it must show:

1. The SQL/query code uses the source-scope filter.
2. The SQL/query code uses the H24N safe predicate with configured buffer.
3. The code excludes blocked Tmax sources by default.
4. At least one validation query confirms zero rows in the derived feature dataset violate:

```text
run_time_utc + publication_buffer <= target_date_hkt - 1 day at 15:00 HKT
```

5. The resulting features preserve or explicitly document location/member aggregation.
6. The task record links back to this ledger and states any deviations.

## Update Rule

Update this ledger whenever any of the following changes:

- GribStream source scope or acquisition version.
- `forecast_wide` schema.
- daily Tmax source allow/block list.
- publication buffer policy.
- GribStream API/provider confirmation changes the buffer assumption or proves model-specific availability delays.
- H24N cutoff policy.
- source-specific data-quality gates.
- old smoke/test rows are purged or moved.
- a DB view or materialized feature table is created to implement this logic.

Do not let implementation drift from this document.
