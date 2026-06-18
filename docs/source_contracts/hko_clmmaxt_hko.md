# hko_clmmaxt_hko

## Purpose

Candidate historical target-label proxy for HKO daily maximum temperature.

## Official Evidence

Endpoint:

`https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMAXT&rformat=csv&station=HKO`

Archived SHA-256:

`5a0a646b4d125e40c25871abbccd5cd24e4f552063a547803ebac820166be4c9`

## Endpoint/Query

HTTP CSV from HKO Open Data API with `dataType=CLMMAXT`, `rformat=csv`, and
`station=HKO`.

## Authentication

None observed.

## Terms/License

Subject to HKO Open Data terms.

## Cadence

Monthly or provider-updated daily-history file. The current snapshot retrieved
on 2026-06-18 ends at 2026-05-31.

## Coverage

Snapshot contains dated rows from 1884-01-01 through 2026-05-31, with provider
footer rows documenting unavailable, incomplete, and complete markers.

## Response Schema

The current CSV has bilingual headers:

```text
年/Year, 月/Month, 日/Day, 數值/Value, 數據完整性/data Completeness
```

Footer-only rows are not observations.

## Units

Degrees Celsius, expected source precision `0.1`.

## Station/Entity Mapping

The title line names Daily Maximum Temperature at the Hong Kong Observatory.
Station code is fixed by query parameter `station=HKO`.

## Timestamp Semantics

Date columns are Hong Kong local calendar dates. `retrieved_at` is the archive
time of the full latest-history file and is not proof of historical
availability.

## Expected Latency

Not operationally eligible as a feature or canonical first-published target.
Use only as a proxy after G1 parity evidence.

## Revision Behavior

Finalized history may be revised or quality-controlled. Preserve every retrieved
file by hash.

## Point-In-Time Role

`PROXY_WITH_LIMITATIONS`.

## Quality Checks

- Archive raw CSV and sidecar before parsing.
- Accept bilingual HKO headers only through canonical header mapping.
- Preserve completeness code.
- Treat invalid calendar dates as parse issues.
- Compare only rows with complete, numeric values.

## Failure Handling

Fail closed on missing headers, non-numeric dated values, invalid schema, or
source failure.

## Rate Limits

No explicit limit observed. Use conservative request intervals.

## Raw Path

`data/raw/hko_clmmaxt_hko/YYYY/MM/DD/`

## Parser Version

`hkg_tmax.hko.parse_daily_climate_csv`, updated in EXP-0002 for real HKO
bilingual headers and footer rows.

## Tests

- `tests/test_hko.py::test_parse_hko_daily_climate_with_bilingual_header`
- `tests/test_hko.py::test_parse_hko_daily_climate_with_title_line`

## Known Limitations

CLMMAXT is not accepted as canonical until first-published Daily Extract parity
is proven. The May 2026 latest-payload parity sample matched 31/31 rows but is
not enough to pass G1.
