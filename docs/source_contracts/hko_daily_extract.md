# hko_daily_extract

## Purpose

Canonical target-source candidate for the Hong Kong Observatory target station:
Daily Extract field `Absolute Daily Max (deg. C)`.

## Official Evidence

- HTML shell: `https://www.hko.gov.hk/en/cis/dailyExtract.htm`
- Coverage/index backing payload: `https://www.hko.gov.hk/cis/hko.xml`
- Monthly backing payload pattern:
  `https://www.hko.gov.hk/cis/dailyExtract/dailyExtract_YYYYMM.xml`
- Annual backing payload pattern:
  `https://www.hko.gov.hk/cis/dailyExtract/dailyExtract_YYYY.xml`

Archived evidence:

- `hko_daily_extract` HTML shell SHA-256:
  `347eb605218a9c9db46eee3aa6b187cb2b557c261d1ba8e6e1527d55f81aa11c`
- `hko_daily_extract_catalog` SHA-256:
  `f80772b68545c56e6842c34998696fd11b7b9a80c0088bb1f6e4da65102616eb`
- `hko_daily_extract_202605` SHA-256:
  `a97230cd78e0a11c4455c23288c96542ec3c13584071619750900a385146dc95`

## Endpoint/Query

The HTML shell is not sufficient for target extraction. It references
root-relative backing endpoints under `/cis/`. The backing payloads have an
`.xml` extension and `text/xml` content type, but the body is JSON text consumed
by the page JavaScript.

## Authentication

None observed.

## Terms/License

Subject to HKO terms. Do not bypass rate limits or access controls.

## Cadence

Daily Extract values can change during publication/finalization. The catalog
records `endYear` and `endMonth`; current months may be served as monthly
payloads until rolled into annual payloads.

## Coverage

Catalog snapshot on 2026-06-18 lists Daily Extract coverage from 1884-01 through
2026-05. The 2026-06 monthly payload was also available with rows through
2026-06-17 at retrieval.

## Response Schema

Backing payload root:

```text
stn.data[].month
stn.data[].dayData[]
```

For the current monthly schema used in 2026-05 and 2026-06:

```text
dayData[0] = day
dayData[1] = mean pressure hPa
dayData[2] = Absolute Daily Max (deg. C)
dayData[3] = mean temperature deg. C
dayData[4] = Absolute Daily Min (deg. C)
```

Summary rows such as `Mean/Total` and `Normal` are not target rows.

## Units

Degrees Celsius, expected source precision `0.1`.

## Station/Entity Mapping

The table header names Hong Kong Observatory. Target station code is `HKO` and
is cross-checked against `hko_station_metadata`.

## Timestamp Semantics

Target date is the Hong Kong local calendar day shown in the Daily Extract row.
Raw archive `retrieved_at` is the only observed availability timestamp unless a
future polling job captures first publication.

## Expected Latency

Not yet measured. G1 cannot pass first-publication parity until polling observes
the first public payload for target dates.

## Revision Behavior

Later payloads may revise, fill, or annualize earlier values. Never overwrite a
first archived payload.

## Point-In-Time Role

`TARGET_ONLY`. Latest historical payloads are not forecast features.

## Quality Checks

- Archive raw bytes before parsing.
- Require exactly one row for the target local date.
- Require field position/name evidence from archived HTML shell.
- Require station `HKO`.
- Require one-decimal precision.
- Reject missing (`***`) or incomplete (`#`) target values for canonical labels.

## Failure Handling

Fail closed on missing source, missing field, ambiguous date, unsupported
precision, station mismatch, missing value, parse issue, or HTTP/source failure.

## Rate Limits

No explicit limit observed. Use conservative polling and preserve user agent.

## Raw Path

`data/raw/hko_daily_extract*/YYYY/MM/DD/`

## Parser Version

`hkg_tmax.hko.parse_daily_extract_json` introduced in EXP-0002.

## Tests

- `tests/test_hko.py::test_parse_daily_extract_json_extracts_absolute_daily_max`
- `tests/test_hko.py::test_parse_daily_extract_json_marks_incomplete_values`
- `tests/test_target.py`

## Known Limitations

First-publication timing is not proven for historical May 2026 rows. Current
parity artifacts compare latest archived Daily Extract payloads to latest
CLMMAXT only.
