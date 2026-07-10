# hko_station_metadata

## Purpose

Station identity and metadata evidence for the target station `HKO`.

## Official Evidence

Endpoint:

`https://www.hko.gov.hk/en/cis/stn.htm`

Archived SHA-256:

`9d54c0d4ff6df23ea435f7d5079765474b38829b54cf4723aefbb392ee870371`

## Endpoint/Query

HTTP HTML station information page.

## Authentication

None observed.

## Terms/License

Subject to HKO terms.

## Cadence

Provider-updated metadata page. Current archived response header reports
`last-modified: Thu, 18 Sep 2025 09:15:01 GMT`.

## Coverage

Archived page includes a row for Hong Kong Observatory `(HKO)` with effective
date `01/01/1884`, latitude `22°18'07"`, longitude `114°10'27"`, and elevation
`32`.

## Response Schema

HTML table. EXP-0002 does not implement a station-history parser yet; source
contract only records archived evidence and target-station row location.

## Units

Latitude/longitude in degrees/minutes/seconds. Elevation in metres.

## Station/Entity Mapping

Target code: `HKO`.

Target name: Hong Kong Observatory.

## Timestamp Semantics

Metadata effective date appears in the station row. Archive `retrieved_at` is
the page retrieval time. A full station-history timeline remains G4 work.

## Expected Latency

Not a forecast feature. Metadata changes require versioned review before
ingestion or modelling.

## Revision Behavior

Provider may update station metadata. Archive each changed page and require
review before changing station mappings.

## Point-In-Time Role

`METADATA`.

## Quality Checks

- Archive raw HTML and sidecar.
- Verify target row contains expected station code/name.
- No silent station substitution.
- Any station-coordinate or elevation change requires a versioned timeline.

## Failure Handling

Fail closed if the `HKO` target row is missing, duplicated, or semantically
changed.

## Rate Limits

No explicit limit observed. Use conservative polling.

## Raw Path

`data/raw/hko_station_metadata/YYYY/MM/DD/`

## Parser Version

No parser yet. Contract evidence only.

## Tests

No dedicated parser tests yet. G4 should add station metadata extraction tests.

## Known Limitations

This does not complete the station-history audit. It only anchors the target
station identity used by the G1 parser/adapter.
