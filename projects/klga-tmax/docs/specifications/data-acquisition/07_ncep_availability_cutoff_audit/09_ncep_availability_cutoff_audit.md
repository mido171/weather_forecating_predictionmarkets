# 09 — NCEP Production Status and Availability-Cutoff Audit

## 1. Purpose

This source is not a weather predictor. It is an availability-control and audit source. The goal is to measure when NCEP model products are actually complete and available in live operations so that historical backtests do not assume impossible timing.

Official source:

```text
NCEP/NCO production status page: https://www.nco.ncep.noaa.gov/pmb/nwprod/prodstat/
```

The production-status page shows operational job status and completion times for current production cycles. It is live/current rather than a complete long historical archive. Therefore, its main value is building a forward-looking empirical availability ledger from the day we start collecting it.

## 2. Required collection schedule

Codex must run a live scraper/collector:

```text
Every 5 minutes, 24/7.
```

Required retention:

```text
Store raw HTML/text response every poll.
Store parsed job rows.
Never overwrite old polls.
```

## 3. Required products to monitor

Monitor all job/product lines related to:

```text
hrrr
rap
gfs
gefs
nam if present
nbm/blend if present
rtma
urma
rrfs if present
spc/href products if present
```

Because page layout/job names may change, Codex must parse generously and preserve all unknown rows, not only known models.

## 4. Bronze schema

```text
CREATE TABLE ncep_prodstat_raw_polls (
    poll_id TEXT PRIMARY KEY,
    retrieved_at_utc TIMESTAMP NOT NULL,
    url TEXT NOT NULL,
    http_status INTEGER,
    response_sha256 TEXT NOT NULL,
    raw_html_or_text TEXT NOT NULL,
    parser_version TEXT,
    quality_flag TEXT NOT NULL DEFAULT 'ok',
    quality_note TEXT
);
```

## 5. Silver schema

```text
CREATE TABLE ncep_prodstat_jobs (
    poll_id TEXT NOT NULL,
    retrieved_at_utc TIMESTAMP NOT NULL,
    model_family TEXT,
    job_name TEXT NOT NULL,
    cycle_time_utc TIMESTAMP,
    nominal_cycle_hour INTEGER,
    status TEXT,
    completion_time_utc TIMESTAMP,
    status_color TEXT,
    raw_row_text TEXT NOT NULL,
    parse_confidence DOUBLE PRECISION,
    parser_version TEXT NOT NULL,
    PRIMARY KEY (poll_id, job_name, COALESCE(cycle_time_utc,'1900-01-01'))
);
```

## 6. Parsed availability derivation

For each model family and cycle, compute:

```text
first_seen_complete_at_utc = minimum retrieved_at_utc where status becomes COMPLETE
reported_completion_time_utc = parsed completion timestamp when present
last_noncomplete_seen_at_utc = maximum retrieved_at_utc before first complete
availability_lower_bound_utc = last_noncomplete_seen_at_utc
availability_upper_bound_utc = first_seen_complete_at_utc
availability_estimate_utc = reported_completion_time_utc if parse confidence high else first_seen_complete_at_utc
```

Store in:

```text
CREATE TABLE ncep_model_availability_estimates (
    model_family TEXT NOT NULL,
    product_or_job_group TEXT NOT NULL,
    cycle_time_utc TIMESTAMP NOT NULL,
    first_seen_complete_at_utc TIMESTAMP,
    reported_completion_time_utc TIMESTAMP,
    last_noncomplete_seen_at_utc TIMESTAMP,
    availability_lower_bound_utc TIMESTAMP,
    availability_upper_bound_utc TIMESTAMP,
    availability_estimate_utc TIMESTAMP,
    number_of_polls INTEGER,
    parser_version TEXT NOT NULL,
    quality_flag TEXT NOT NULL DEFAULT 'ok',
    quality_note TEXT,
    PRIMARY KEY (model_family, product_or_job_group, cycle_time_utc)
);
```

## 7. Integration with availability ledger

When NCEP status gives a reliable completion estimate, update source availability for corresponding GribStream/raw NOAA rows:

```text
availability_method = provider_production_status
provider_available_at_utc = max(ncep_availability_estimate_utc, provider_specific_distribution_buffer)
```

Default distribution buffer after NCEP complete:

```text
15 minutes
```

This buffer accounts for downstream provider indexing/distribution unless actual GribStream/Open-Meteo ingestion logs prove tighter timing.

## 8. Cutoff audit reports

Codex must generate daily reports comparing each cutoff to actual model availability:

For each target date `T` and cutoff:

```text
Which HRRR/RAP/GFS/GEFS/NBM cycles were actually complete by cutoff?
Which cycles were assumed eligible by conservative lag?
Were conservative lag assumptions too aggressive or too conservative?
Which model runs should be excluded from historical backtests for that cutoff?
```

Report table:

```text
target_date_local
cutoff_id
cutoff_utc
model_family
cycle_time_utc
eligible_by_actual_prodstat
eligible_by_conservative_rule
actual_minus_rule_minutes
recommended_lag_adjustment_minutes
```

## 9. Use restrictions

The NCEP production-status collector only applies to NCEP/NOAA operational products. Do not use it for ECMWF, GribStream indexing, Open-Meteo server availability, Wunderground, or Polymarket.

For non-NCEP sources, use:

```text
provider metadata if available,
actual ingestion logs,
source-specific conservative lag rules.
```

## 10. Acceptance tests

```text
[ ] Prodstat raw page is stored every poll with timestamp and hash.
[ ] Parser extracts at least job_name/status/raw_row_text even when detailed timestamps fail.
[ ] Availability estimate rows are generated for HRRR/RAP/GFS/GEFS/NBM when visible on page.
[ ] Cutoff audit report flags cycles whose run_time would be before cutoff but actual completion after cutoff.
[ ] Availability ledger can override conservative lag with actual prodstat evidence.
[ ] Forecast feature builder never reads prodstat as a meteorological predictor.
```
