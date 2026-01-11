# Time Semantics (Canonical)

This document defines the only acceptable interpretation of target dates, as-of cutoffs,
and CLI settlement windows. All modules must follow this contract.

## Key terms
### Station time zone (ZoneId)
- Example values: America/New_York, America/Chicago, America/Los_Angeles
- Used for local calendar dates and asOfLocal timestamps.

### Station standard offset (ZoneOffset)
Kalshi notes that NWS daily climate reports use local standard time for the
"calendar day" definition. During DST, the climate day runs 01:00-01:00 local
daylight time rather than midnight-midnight.

Implementation definition:
- Standard offset is the winter offset for the zone.
- Store as `standard_offset_minutes` (example: -300, -360, -480).

### Target date (T) - `target_date_local`
Calendar date in the station's local calendar (YYYY-MM-DD).

### CLI settlement day window (local standard time day)
For `target_date_local = T`, define the CLI window as:
- Start: `T 00:00` at the station standard offset
- End: `(T+1) 00:00` at the station standard offset

During DST, this corresponds to 01:00-01:00 local daylight time.

### As-of timestamp - `asof_utc`
Latest allowed instant for forecast availability when predicting day T.

Canonical policy for this project:
- `asOfLocalTime = 23:00` at station ZoneId on date (T-1)
- `asOfLocal = (T-1) at asOfLocalTime`
- Convert to UTC to store `asof_utc`

Policies live in the `asof_policy` table.

## No-leakage rule (hard requirement)
For every MOS feature used to predict day T:
- `mos_asof_feature.chosen_runtime_utc` must be `<= mos_asof_feature.asof_utc`.
- If no run exists, feature is NULL with a `missing_reason`.

## Required timestamps for auditing
For MOS as-of features:
- `target_date_local` (DATE)
- `asof_utc` (TIMESTAMP)
- `asof_local` (TIMESTAMP) with `station_zoneid` stored alongside
- `chosen_runtime_utc` (TIMESTAMP)
- `retrieved_at_utc` (TIMESTAMP)
- `raw_payload_hash_ref`

For CLI daily truth:
- `target_date_local` (DATE)
- `tmax_f` (DECIMAL)
- `report_issued_at_utc` (TIMESTAMP, nullable)
- `retrieved_at_utc` (TIMESTAMP)
- `raw_payload_hash`

## Worked examples
### Example A: DST season (Miami)
Station: KMIA
ZoneId: America/New_York
Standard offset: -05:00
Target date T: 2025-07-10 (DST in effect)

Settlement window (standard time day):
- Start: 2025-07-10T00:00-05:00 -> 2025-07-10T05:00Z
- End:   2025-07-11T00:00-05:00 -> 2025-07-11T05:00Z
Local clock view (EDT):
- Start: 2025-07-10 01:00 EDT
- End:   2025-07-11 01:00 EDT

As-of cutoff:
- asOfLocal: 2025-07-09 23:00 EDT
- asOfUtc:   2025-07-10T03:00Z

### Example B: standard time season (Chicago)
Station: KMDW
ZoneId: America/Chicago
Standard offset: -06:00
Target date T: 2025-01-15 (standard time)

Settlement window (standard time day):
- Start: 2025-01-15T00:00-06:00 -> 2025-01-15T06:00Z
- End:   2025-01-16T00:00-06:00 -> 2025-01-16T06:00Z
Local clock view (CST):
- Start: 2025-01-15 00:00 CST
- End:   2025-01-16 00:00 CST

As-of cutoff:
- asOfLocal: 2025-01-14 23:00 CST
- asOfUtc:   2025-01-15T05:00Z
