# GribStream Tmax and Leakage Safety

Last updated: 2026-06-25

This file is the mandatory safety note for turning raw GribStream rows into HKG Tmax model features.

## The Rule

Rows in `nwp_tactical.forecast_wide` are raw forecast rows. They are not automatically safe as model features.

Before any row can be used for an H24N feature, the extractor must enforce this rule:

```text
run_time_utc + publication_buffer <= target_date_hkt - 1 day at 15:00 HKT
```

For the 2026-06-25 sanity check, the publication buffer was 6 hours.

In SQL terms, the cutoff instant is:

```sql
((target_date_hkt::timestamp - interval '1 day' + interval '15 hours') at time zone 'Asia/Hong_Kong')
```

The safe filter is:

```sql
run_time_utc + interval '6 hours'
  <= ((target_date_hkt::timestamp - interval '1 day' + interval '15 hours') at time zone 'Asia/Hong_Kong')
```

Do not group raw rows by `target_date_hkt` without this filter.

## Why This Matters

`target_date_hkt` is the Hong Kong calendar day that a forecast valid time belongs to.

That does not automatically mean the model run was known before the H24N forecast cutoff.

Some raw rows sit near the date boundary. If those rows are grouped naively, a feature can accidentally use a model run that would not have been available at the intended decision time.

## Correct GFS Example

Target day:

```text
2021-03-24 HKT
```

Decision cutoff:

```text
2021-03-23 15:00 HKT
= 2021-03-23 07:00 UTC
```

Safe GFS run:

```text
run_time_utc = 2021-03-23 00:00 UTC
```

With the 6-hour buffer:

```text
2021-03-23 00:00 UTC + 6 hours = 2021-03-23 06:00 UTC
```

Since `06:00 UTC <= 07:00 UTC`, this run is safe for the 2021-03-24 HKT forecast.

For that one safe GFS run, the daily max example used:

```text
24 forecast valid hours x 12 HKO stencil locations = 288 rows
```

The derived daily max was:

```text
max(interval_tmax_2m_k or temperature_2m_k across the 288 safe rows) = 26.68 C
```

This does not mean there were 288 different days. It means one target day had 288 forecast points:

- 24 forecast hours within the target day;
- 12 HKO-area locations per forecast hour;
- weather values stored as columns on each row.

## Tmax Source Status From 10-Week Smoke

Usable as daily Tmax sources after the H24N leakage filter:

```text
gfs
gefsatmosmean
gefsatmos
ifsoper
ifsenfo
cwawrf15
aifsoper
aifsenfo
aigfssfc
graphcast
fourcastnetgfs
```

Not usable as daily Tmax-producing sources right now:

```text
aigfspres
aigefssfc
nbmoc
```

Reasons:

- `aigfspres`: no surface 2m temperature or interval Tmax; upper-air support only.
- `aigefssfc`: 8,432 rows returned, but 8,308 rows had JSON null for `member_temperature_2m_k`; only one target day had usable non-null temperature.
- `nbmoc`: zero rows returned.

## Backfill Gate

Full tactical GribStream backfill may proceed only if downstream feature extraction enforces:

```text
H24N cutoff filter + configured publication buffer
```

Do not treat `aigfspres`, `aigefssfc`, or `nbmoc` as daily Tmax-producing sources unless a later selector/provider probe proves usable non-null 2m or Tmax coverage.

