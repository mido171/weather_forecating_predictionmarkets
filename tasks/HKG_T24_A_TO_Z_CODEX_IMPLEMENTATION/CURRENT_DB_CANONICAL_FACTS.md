# Current Database Canonical Facts

## Official HKO forecast archive

The project owner checked Postgres directly. The following values must be independently reproduced by Task T00 before use:

```text
Table: public.hko_historical_forecasts_2000_2026
Clean filter: row_quality_status = 'usable_local_minmax'
Clean rows: 115,795
Product type in clean subset: local
Issue range as reported: 2000-01-01 16:22:00 through 2026-06-20 23:45:00
Target-date range: 2000-01-02 through 2026-06-21
Distinct target dates: 9,667
Missing target date within range: 2003-02-02
Raw full-table rows: 324,179
Raw product counts: local 264,325; 5day 6,193; 7day 23,223; 9day 30,438
```

Clean numerical profile:

| Attribute | Non-null | Min | Median | Mean | Max |
|---|---:|---:|---:|---:|---:|
| forecast_min_c | 115,795 | 1.0 | 24.0 | 22.2375 | 30.0 |
| forecast_max_c | 115,795 | 7.0 | 28.0 | 26.6010 | 39.0 |
| forecast_range_c | 115,795 | 2.0 | 4.0 | 4.3636 | 16.0 |
| forecast_midpoint_c | 115,795 | 6.0 | 26.0 | 24.4193 | 33.0 |
| target_issue_lead_days | 115,795 | 0.0 | 1.0 | 0.7643 | 1.0 |
| stale_hours | 0 | null | null | null | null |

## Strategic consequence

This is a near-continuous, multi-vintage official forecast archive. The system must preserve all eligible forecast vintages and derive:

- latest forecast at cutoff;
- first forecast at cutoff horizon;
- number and timing of revisions;
- min/max revision path;
- forecast age and issue-hour state;
- source/product/parser era;
- forecast text and weather-regime changes;
- official residual and online source bias after outcomes settle.

Do not collapse the table to one row per target date before revision features are created.

## Other current datasets

The attached audit evidence covers 13 dataset families, 52 tables, 1,869 attributes, 36 ISD stations, and known quality blockers. The audit remains the governing source disposition, except that the official forecast facts above supersede its earlier sparse-coverage interpretation.
