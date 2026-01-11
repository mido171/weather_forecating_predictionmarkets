# WX-104 — Implement Time Semantics Library (asOf + CLI standard-time day windows) with exhaustive tests

## Objective
Create a single shared library (in `common` module) that calculates:
- `asOfUtc` from:
  - station ZoneId
  - target date T
  - as-of policy local time (applied to date T-1)
- CLI “standard-time day window” start/end instants in UTC for date T:
  - Start = T 00:00 at station standard offset
  - End   = (T+1) 00:00 at station standard offset

This library is the foundation for “no forward looking leakage.”

## Inputs / outputs
### asOfUtc
Inputs:
- `targetDateLocal` (LocalDate)
- `asofLocalTime` (LocalTime)
- `stationZoneId` (ZoneId)
Output:
- `asOfUtc` (Instant)
- `asOfLocalZdt` (ZonedDateTime)

### CLI standard-time window
Inputs:
- `targetDateLocal` (LocalDate)
- `standardOffsetMinutes` (int)
Output:
- `windowStartUtc` (Instant)
- `windowEndUtc` (Instant)

## DST edge cases (MUST TEST)
You must include tests for both DST start and DST end days for:
- America/New_York
- America/Chicago
- America/Los_Angeles

Examples:
- DST starts (spring forward): second Sunday in March
- DST ends (fall back): first Sunday in November
(Use known dates in recent years.)

## Acceptance Criteria
- [ ] Library functions are pure and deterministic (no system clock dependency).
- [ ] Unit tests cover DST start/end for each timezone and validate expected UTC conversions.
- [ ] Library includes a “leakage guard” helper:
  - `assertRuntimeNotAfterAsOf(runtimeUtc, asOfUtc)` throws with clear diagnostics.
- [ ] Documented examples in Javadoc match `docs/time-semantics.md`.

## Source references
- Kalshi weather markets: NWS climate reports use local standard time; during DST daily highs correspond to 1AM→12:59AM local time.
- IEM CLI notes: daily climate report totals are midnight local standard time; during DST it maps to 1AM→1AM local daylight time.

