# T07-T12 Deep Sanity Audit - 2026-06-25

This is the durable human summary for the post-backfill sanity pass over the tactical GribStream data.

Machine-readable audit output:

```text
experiments/0214_tactical_h24n_gribstream_backfill/full_tactical_backfill_ok_tmax/deep_sanity_audit_20260625.json
experiments/0214_tactical_h24n_gribstream_backfill/full_tactical_backfill_ok_tmax/DEEP_SANITY_AUDIT_20260625.md
```

Audit command:

```powershell
.\.venv\Scripts\python.exe scripts\audit_tactical_gribstream_deep_sanity.py --skip-file-hash
```

`--skip-file-hash` means the audit checked raw-file existence and byte size, but did not rehash all payloads. The DB still stores the SHA256 recorded at acquisition time.

## Bottom Line

The full tactical backfill is API-clean and structurally consistent, but not modeling-ready without source and leakage filters.

Confirmed clean:

- Full-run rows in `nwp_tactical.forecast_wide`: 1,964,157.
- Current total rows in `nwp_tactical.forecast_wide`: 1,965,090.
- Non-full rows still mixed into `forecast_wide`: 933 older `batch_smoke_10w` `gefsatmos` rows.
- Full-run raw objects checked: 1,163.
- Missing full-run raw files: 0.
- Raw byte-size mismatches: 0.
- API event log HTTP errors: 0.
- Structural mismatches: 0 lead-hour mismatches, 0 target-date mismatches, 0 non-H24N rows, 0 non-tactical-version rows, 0 coordinate-policy mismatches, and 0 empty `raw_values_jsonb` rows.

Critical source filter:

```sql
JOIN nwp_tactical.raw_response_object r
  ON r.response_object_id = fw.source_response_object_id
WHERE r.object_uri LIKE '%full_tactical_backfill_ok_tmax%'
```

Use that source filter until old smoke/test rows are purged or moved out of the modeling surface.

## Current Fetched Scopes

Raw objects currently present in `nwp_tactical.raw_response_object`:

| Source scope | Raw objects | Row count sum | Bytes |
| --- | ---: | ---: | ---: |
| `smoke` | 14 | 1,728 | 50,853 |
| `first_week` | 13 | 11,796 | 298,866 |
| `batch_smoke_10w` | 95 | 123,652 | 3,129,036 |
| `full_tactical_backfill_ok_tmax` | 1,163 | 1,964,157 | 56,488,866 |

Rows currently visible in `nwp_tactical.forecast_wide`:

| Source scope | Dataset | Rows |
| --- | --- | ---: |
| `batch_smoke_10w` | `gefsatmos` | 933 |
| `full_tactical_backfill_ok_tmax` | all full-run datasets | 1,964,157 |

The older smoke and first-week raw files remain in the raw ledger. Only 933 old smoke rows remain in the wide modeling table.

## Full-Run Dataset Coverage

| Dataset | Rows | Run-time UTC coverage | Target-date HKT coverage | Runs | Locations | Members |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| `gfs` | 575,004 | 2021-03-23 00Z to 2026-06-22 00Z | 2021-03-23 to 2026-06-23 | 1,918 | 12 | 1 |
| `gefsatmosmean` | 200,436 | 2020-10-01 18Z to 2026-06-21 18Z | 2020-10-03 to 2026-06-23 | 2,088 | 12 | 1 |
| `gefsatmos` | 516,891 | 2020-10-01 18Z to 2026-06-21 18Z | 2020-10-03 to 2026-06-23 | 2,085 | 1 | 31 |
| `ifsoper` | 91,260 | 2024-02-28 18Z to 2026-06-21 18Z | 2024-02-29 to 2026-06-23 | 845 | 12 | 1 |
| `ifsenfo` | 343,616 | 2024-03-01 18Z to 2026-06-21 18Z | 2024-03-03 to 2026-06-23 | 843 | 1 | 51 |
| `cwawrf15` | 180 | 2026-06-22 18Z to 2026-06-24 18Z | 2026-06-23 to 2026-06-26 | 3 | 12 | 1 |
| `aifsoper` | 28,884 | 2025-02-25 18Z to 2026-06-21 18Z | 2025-02-26 to 2026-06-23 | 482 | 12 | 1 |
| `aifsenfo` | 72,270 | 2025-07-02 18Z to 2026-06-21 18Z | 2025-07-04 to 2026-06-23 | 355 | 1 | 51 |
| `aigfssfc` | 3,660 | 2026-04-21 18Z to 2026-06-21 18Z | 2026-04-22 to 2026-06-23 | 61 | 12 | 1 |
| `aigfspres` | 3,660 | 2026-04-21 18Z to 2026-06-21 18Z | 2026-04-22 to 2026-06-23 | 61 | 12 | 1 |
| `aigefssfc` | 46,252 | 2025-06-01 18Z to 2026-06-21 18Z | 2025-06-03 to 2026-06-23 | 373 | 1 | 31 |
| `graphcast` | 44,220 | 2024-04-25 18Z to 2026-05-04 18Z | 2024-04-26 to 2026-05-06 | 737 | 12 | 1 |
| `fourcastnetgfs` | 37,824 | 2024-05-02 18Z to 2026-02-18 18Z | 2024-05-03 to 2026-02-20 | 631 | 12 | 1 |
| `nbmoc` | 0 | empty probe | empty probe | 0 | 0 | 0 |

## Tmax Derivability

After applying the conservative H24N leakage filter, daily Tmax candidates are derivable for:

- `gfs`
- `gefsatmosmean`
- `gefsatmos`
- `ifsoper`
- `ifsenfo`
- `cwawrf15` for rolling/prospective use only
- `aifsoper`
- `aifsenfo`
- `aigfssfc`
- `graphcast`
- `fourcastnetgfs` through its available archive end

Do not treat these as daily Tmax-producing sources from this pull:

- `nbmoc`: returned zero rows.
- `aigfspres`: upper-air support only, no surface Tmax candidate.
- `aigefssfc`: only 67 of 373 target days had usable 2m temperature/Tmax candidates; keep blocked unless a later selector/provider probe fixes this.

## Leakage Safety

The raw table is not automatically leakage-safe. Downstream feature extraction must enforce:

```text
run_time_utc + publication_buffer <= target_date_hkt - 1 day at 15:00 HKT
```

The audit used a 6-hour conservative publication buffer.

Unsafe rows exist for deterministic/AI deterministic families if queried naively:

| Dataset | Rows | Safe rows | Unsafe rows |
| --- | ---: | ---: | ---: |
| `gfs` | 575,004 | 552,000 | 23,004 |
| `ifsoper` | 91,260 | 81,120 | 10,140 |
| `aifsoper` | 28,884 | 23,100 | 5,784 |
| `graphcast` | 44,220 | 35,376 | 8,844 |
| `fourcastnetgfs` | 37,824 | 30,252 | 7,572 |
| `aigfssfc` | 3,660 | 2,928 | 732 |
| `aigfspres` | 3,660 | 2,928 | 732 |
| `cwawrf15` | 180 | 144 | 36 |

The ensemble HKO-center families in this pull were fully safe under the 6-hour filter:

- `gefsatmos`
- `gefsatmosmean`
- `ifsenfo`
- `aifsenfo`
- `aigefssfc`

## Known Data-Quality Flags

`ifsenfo`:

- 8 recent chunks returned HTTP 200 but missed member `0`.
- Affected run windows: 2026-05-15T18Z through 2026-06-21T18Z.
- The data for other members was persisted.

`fourcastnetgfs`:

- Requested tail 2026-02-19T18Z through 2026-02-28T18Z returned HTTP 200 with zero rows.
- Persisted DB rows end at 2026-02-18T18Z.

`nbmoc`:

- Probe returned HTTP 200 with zero rows.

Member coverage anomalies:

- `ifsenfo`: 328 run/valid groups missing member `0`.
- `aifsenfo`: 3 run/valid groups missing members `1..50`.
- `gefsatmos`: 2 run/valid groups had partial missing members.

Target-date gaps inside min/max ranges:

- `gfs`, `ifsoper`, `ifsenfo`, `aifsenfo`, `aifsoper`, `graphcast`, `aigfssfc`, `aigfspres`, and `cwawrf15`: no internal target-date gaps in their pulled ranges.
- `gefsatmos`: 5 missing target dates.
- `gefsatmosmean`: 2 missing target dates.
- `fourcastnetgfs`: 11 missing target dates.
- `aigefssfc`: 13 missing target dates and poor usable Tmax coverage.

## Value and Cross-Field Sanity

No dewpoint-above-temperature anomalies were found.

Known physical-range flags:

- `cwawrf15.accumulated_precip_kg_m2`: tiny negative values down to about `-5.22e-7`, likely floating-point/accumulation noise.
- `aifsoper.total_precip_m`: max `146.34`; likely unit/selector semantics. Do not use blindly as meters of daily rain.
- `aifsoper.shortwave_down_j_m2` and `ifsoper.shortwave_down_j_m2`: some values above the conservative threshold.
- `ifsoper.relative_humidity_700_pct`: 266 rows above 100 percent, max `106.09`.

Cross-field flags:

- `gefsatmosmean`: 382 rows where `interval_tmax_2m_k < temperature_2m_k`.
- `gfs`: 27,776 rows where `interval_tmax_2m_k < temperature_2m_k`.

Interpretation: use `interval_tmax_2m_k` where it is a valid model-provided Tmax candidate, but do not assume every row-level interval Tmax dominates every instantaneous 2m temperature row without understanding the accumulation/interval semantics.

## Modeling Gate

Do not mark the consolidated T07-T12 GribStream task as fully complete from row counts alone.

Before modeling, the next agent must:

1. Filter to `full_tactical_backfill_ok_tmax` source rows or purge/move the 933 old smoke rows.
2. Enforce the H24N leakage cutoff with a publication buffer.
3. Exclude `nbmoc`, `aigfspres`, and `aigefssfc` from daily Tmax source features unless explicitly accepted as blocked/support-only inputs.
4. Decide whether `ifsenfo` missing member `0` in recent chunks is acceptable or needs provider follow-up.
5. Treat `fourcastnetgfs` as available only through its observed archive end.
