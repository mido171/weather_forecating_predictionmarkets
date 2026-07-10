# Deep Sanity Audit - Full Tactical GribStream Backfill - 2026-06-25

## Executive Result

- Current `nwp_tactical.forecast_wide` rows: 1,965,090.
- Rows sourced from the full backfill raw root: 1,964,157.
- Non-full rows still present in `forecast_wide`: 933.
- API log HTTP error count: 0.
- Full raw objects checked: 1,163; missing files: 0; size mismatches: 0; sha256 files rehashed: 0; sha256 mismatches: 0.

Critical table-scope issue: the live table is not pure full-run data. Filter by joining `source_response_object_id` to `nwp_tactical.raw_response_object` and requiring `object_uri LIKE '%full_tactical_backfill_ok_tmax%'` until old smoke rows are purged or moved.

## Source Scope Counts

| Source scope | Dataset | Rows |
| --- | --- | ---: |
| `batch_10w` | `gefsatmos` | 933 |
| `full` | `aifsenfo` | 72,270 |
| `full` | `aifsoper` | 28,884 |
| `full` | `aigefssfc` | 46,252 |
| `full` | `aigfspres` | 3,660 |
| `full` | `aigfssfc` | 3,660 |
| `full` | `cwawrf15` | 180 |
| `full` | `fourcastnetgfs` | 37,824 |
| `full` | `gefsatmos` | 516,891 |
| `full` | `gefsatmosmean` | 200,436 |
| `full` | `gfs` | 575,004 |
| `full` | `graphcast` | 44,220 |
| `full` | `ifsenfo` | 343,616 |
| `full` | `ifsoper` | 91,260 |

## Full-Run Dataset Coverage

| Dataset | Rows | Run time UTC | Target dates HKT | Runs | Locations | Members | Lead h |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| `aifsenfo` | 72,270 | 2025-07-02 18:00:00+00:00 to 2026-06-21 18:00:00+00:00 | 2025-07-04 to 2026-06-23 (355) | 355 | 1 | 51 | 24.00..42.00 |
| `aifsoper` | 28,884 | 2025-02-25 18:00:00+00:00 to 2026-06-21 18:00:00+00:00 | 2025-02-26 to 2026-06-23 (483) | 482 | 12 | 1 | 18.00..42.00 |
| `aigefssfc` | 46,252 | 2025-06-01 18:00:00+00:00 to 2026-06-21 18:00:00+00:00 | 2025-06-03 to 2026-06-23 (373) | 373 | 1 | 31 | 24.00..42.00 |
| `aigfspres` | 3,660 | 2026-04-21 18:00:00+00:00 to 2026-06-21 18:00:00+00:00 | 2026-04-22 to 2026-06-23 (63) | 61 | 12 | 1 | 18.00..42.00 |
| `aigfssfc` | 3,660 | 2026-04-21 18:00:00+00:00 to 2026-06-21 18:00:00+00:00 | 2026-04-22 to 2026-06-23 (63) | 61 | 12 | 1 | 18.00..42.00 |
| `cwawrf15` | 180 | 2026-06-22 18:00:00+00:00 to 2026-06-24 18:00:00+00:00 | 2026-06-23 to 2026-06-26 (4) | 3 | 12 | 1 | 18.00..42.00 |
| `fourcastnetgfs` | 37,824 | 2024-05-02 18:00:00+00:00 to 2026-02-18 18:00:00+00:00 | 2024-05-03 to 2026-02-20 (648) | 631 | 12 | 1 | 18.00..42.00 |
| `gefsatmos` | 516,891 | 2020-10-01 18:00:00+00:00 to 2026-06-21 18:00:00+00:00 | 2020-10-03 to 2026-06-23 (2,085) | 2,085 | 1 | 31 | 24.00..45.00 |
| `gefsatmosmean` | 200,436 | 2020-10-01 18:00:00+00:00 to 2026-06-21 18:00:00+00:00 | 2020-10-03 to 2026-06-23 (2,088) | 2,088 | 12 | 1 | 24.00..45.00 |
| `gfs` | 575,004 | 2021-03-23 00:00:00+00:00 to 2026-06-22 00:00:00+00:00 | 2021-03-23 to 2026-06-23 (1,919) | 1,918 | 12 | 1 | 15.00..39.00 |
| `graphcast` | 44,220 | 2024-04-25 18:00:00+00:00 to 2026-05-04 18:00:00+00:00 | 2024-04-26 to 2026-05-06 (741) | 737 | 12 | 1 | 18.00..42.00 |
| `ifsenfo` | 343,616 | 2024-03-01 18:00:00+00:00 to 2026-06-21 18:00:00+00:00 | 2024-03-03 to 2026-06-23 (843) | 843 | 1 | 51 | 24.00..45.00 |
| `ifsoper` | 91,260 | 2024-02-28 18:00:00+00:00 to 2026-06-21 18:00:00+00:00 | 2024-02-29 to 2026-06-23 (846) | 845 | 12 | 1 | 21.00..45.00 |

## Non-Clean Chunks

| Chunk | Dataset | Status | HTTP | Rows | Expected | Window UTC | Issue |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 847 | `ifsenfo` | `failed` | 200 | 2000 | 2040 | 2026-05-15T18:00:00Z to 2026-05-19T18:00:00Z | missing_members=[0] |
| 848 | `ifsenfo` | `failed` | 200 | 2000 | 2040 | 2026-05-20T18:00:00Z to 2026-05-24T18:00:00Z | missing_members=[0] |
| 849 | `ifsenfo` | `failed` | 200 | 2000 | 2040 | 2026-05-25T18:00:00Z to 2026-05-29T18:00:00Z | missing_members=[0] |
| 850 | `ifsenfo` | `failed` | 200 | 2000 | 2040 | 2026-05-30T18:00:00Z to 2026-06-03T18:00:00Z | missing_members=[0] |
| 851 | `ifsenfo` | `failed` | 200 | 2000 | 2040 | 2026-06-04T18:00:00Z to 2026-06-08T18:00:00Z | missing_members=[0] |
| 852 | `ifsenfo` | `failed` | 200 | 2000 | 2040 | 2026-06-09T18:00:00Z to 2026-06-13T18:00:00Z | missing_members=[0] |
| 853 | `ifsenfo` | `failed` | 200 | 2000 | 2040 | 2026-06-14T18:00:00Z to 2026-06-18T18:00:00Z | missing_members=[0] |
| 854 | `ifsenfo` | `failed` | 200 | 1200 | 1224 | 2026-06-19T18:00:00Z to 2026-06-21T18:00:00Z | missing_members=[0] |
| 1162 | `fourcastnetgfs` | `completed_empty` | 200 | 0 | 600 | 2026-02-19T18:00:00Z to 2026-02-28T18:00:00Z |  |
| 1163 | `nbmoc` | `completed_empty` | 200 | 0 | 105 | 2026-06-17T18:00:00Z to 2026-06-23T18:00:00Z |  |

## Structural Checks

- Chunk policy violations: {'non_runs_endpoint': 0, 'non_timeslist_selector': 0, 'has_forecasted_from': 0, 'has_forecasted_until': 0, 'missing_timeslist': 0, 'invalid_lead_bounds': 0, 'non_200_http': 0}.
- Lead-hour mismatches: 0.
- Target-date mismatches vs valid time HKT date: 0.
- Non-H24N rows: 0; non tactical version rows: 0; non raw-valid rows: 0.
- Requested coordinate mismatches vs stencil: 0.
- Rows with empty raw_values_jsonb: 0.

## Member Coverage Anomalies

- `aifsenfo`: 3 run/valid groups affected; missing counts {'1': 3, '2': 3, '3': 3, '4': 3, '5': 3, '6': 3, '7': 3, '8': 3, '9': 3, '10': 3, '11': 3, '12': 3, '13': 3, '14': 3, '15': 3, '16': 3, '17': 3, '18': 3, '19': 3, '20': 3, '21': 3, '22': 3, '23': 3, '24': 3, '25': 3, '26': 3, '27': 3, '28': 3, '29': 3, '30': 3, '31': 3, '32': 3, '33': 3, '34': 3, '35': 3, '36': 3, '37': 3, '38': 3, '39': 3, '40': 3, '41': 3, '42': 3, '43': 3, '44': 3, '45': 3, '46': 3, '47': 3, '48': 3, '49': 3, '50': 3}; extra counts {}.
- `gefsatmos`: 2 run/valid groups affected; missing counts {'18': 2, '23': 2, '27': 2, '29': 2, '30': 2, '1': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 1, '8': 1, '10': 1, '11': 1, '12': 1, '13': 1, '14': 1, '15': 1, '16': 1, '17': 1, '19': 1, '20': 1, '21': 1, '22': 1, '24': 1, '25': 1, '26': 1, '28': 1}; extra counts {}.
- `ifsenfo`: 328 run/valid groups affected; missing counts {'0': 328}; extra counts {}.

## H24N Leakage Filter and Tmax Derivability

| Dataset | Rows | Safe rows | Unsafe rows | Tmax days any | Tmax days hko_center | Daily C range | Usable rows/day | Locations/day | Members/day |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `aifsenfo` | 72,270 | 72,270 | 0 | 355 | 355 | 15.27..32.81 | 154..204 | 1..1 | 51..51 |
| `aifsoper` | 28,884 | 23,100 | 5,784 | 482 | 482 | 16.27..33.92 | 24..48 | 12..12 | 1..1 |
| `aigefssfc` | 46,252 | 46,252 | 0 | 67 | 67 | 18.04..31.85 | 0..124 | 0..1 | 0..31 |
| `aigfspres` | 3,660 | 2,928 | 732 | 0 | 0 |  | 0..0 | 0..0 | 0..0 |
| `aigfssfc` | 3,660 | 2,928 | 732 | 61 | 61 | 24.15..36.35 | 48..48 | 12..12 | 1..1 |
| `cwawrf15` | 180 | 144 | 36 | 3 | 3 | 29.72..33.22 | 48..48 | 12..12 | 1..1 |
| `fourcastnetgfs` | 37,824 | 30,252 | 7,572 | 631 | 631 | 14.52..34.70 | 12..48 | 12..12 | 1..1 |
| `gefsatmos` | 516,891 | 516,891 | 0 | 2,085 | 2,085 | 11.89..34.95 | 59..248 | 1..1 | 31..31 |
| `gefsatmosmean` | 200,436 | 200,436 | 0 | 2,088 | 2,088 | 14.48..37.35 | 84..96 | 12..12 | 1..1 |
| `gfs` | 575,004 | 552,000 | 23,004 | 1,918 | 1,918 | 13.60..37.13 | 144..288 | 12..12 | 1..1 |
| `graphcast` | 44,220 | 35,376 | 8,844 | 737 | 737 | 16.05..34.64 | 48..48 | 12..12 | 1..1 |
| `ifsenfo` | 343,616 | 343,616 | 0 | 843 | 843 | 15.38..32.60 | 400..408 | 1..1 | 50..51 |
| `ifsoper` | 91,260 | 81,120 | 10,140 | 845 | 845 | 16.25..35.41 | 96..96 | 12..12 | 1..1 |

## Target-Date Gap Scan

| Dataset | Dates present | Missing between min/max | First missing ranges |
| --- | ---: | ---: | --- |
| `aifsenfo` | 355 | 0 |  |
| `aifsoper` | 483 | 0 |  |
| `aigefssfc` | 373 | 13 | 2025-06-28, 2025-07-26, 2025-08-13, 2025-08-18, 2025-09-04, 2025-10-08, 2025-10-26, 2025-12-18, 2026-01-03, 2026-01-09, 2026-01-14, 2026-02-05, 2026-02-16 |
| `aigfspres` | 63 | 0 |  |
| `aigfssfc` | 63 | 0 |  |
| `cwawrf15` | 4 | 0 |  |
| `fourcastnetgfs` | 648 | 11 | 2025-01-12..2025-01-13, 2025-06-25, 2025-07-17, 2025-08-08..2025-08-14 |
| `gefsatmos` | 2,085 | 5 | 2020-11-24..2020-11-26, 2023-05-24, 2025-06-29 |
| `gefsatmosmean` | 2,088 | 2 | 2023-05-24, 2025-06-29 |
| `gfs` | 1,919 | 0 |  |
| `graphcast` | 741 | 0 |  |
| `ifsenfo` | 843 | 0 |  |
| `ifsoper` | 846 | 0 |  |

## Physical Range Anomalies

| Dataset | Column | Non-null | Min | Max | Anomaly rows |
| --- | --- | ---: | ---: | ---: | ---: |
| `cwawrf15` | `accumulated_precip_kg_m2` | 180 | -5.222366191446781e-07 | 8.627999682156952 | 75 |
| `aifsoper` | `total_precip_m` | 28,884 | 0.0 | 146.33984375 | 14,395 |
| `aifsoper` | `shortwave_down_j_m2` | 28,884 | 3248907.5625 | 54987776.0 | 239 |
| `ifsoper` | `shortwave_down_j_m2` | 90,612 | 2195456.0 | 54654976.0 | 1,249 |
| `ifsoper` | `relative_humidity_700_pct` | 91,260 | 0.04963874816894531 | 106.08843207359314 | 266 |

## Known Blockers / Do-Not-Model Warnings

- The live DB table has 933 older `batch_smoke_10w` `gefsatmos` rows mixed into `forecast_wide`.
- `ifsenfo` has recent missing-member-0 chunks, although HTTP status was 200 and other members persisted.
- `fourcastnetgfs` full-run rows end before the requested tail; the final tail request returned empty.
- `nbmoc` returned zero rows and is not a usable HKO Tmax source from this pull.
- `aigfspres` is upper-air support only, not a surface Tmax source.
- `aigefssfc` has very poor usable 2m temperature coverage and should remain blocked as a Tmax source unless a selector/provider probe fixes it.
- Raw rows are not feature-safe unless the H24N cutoff filter is applied in downstream feature extraction.

Full machine-readable detail is in `deep_sanity_audit_20260625.json`.
