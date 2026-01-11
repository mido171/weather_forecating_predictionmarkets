# Time Semantics (MUST BE CONSISTENT EVERYWHERE)

This document defines the *only* acceptable interpretation of “day T”, “as-of time”, and the settlement climate-report day window.

## 1) Key terms
### Station Time Zone (ZoneId)
- Example: `America/New_York`, `America/Chicago`, `America/Los_Angeles`
- Used for:
  - interpreting user-facing dates/times
  - producing `asOfLocal` timestamps

### Station Standard Offset (ZoneOffset)
Kalshi explicitly notes that **NWS climate reports use local standard time** for the “calendar day” definition; during DST, the “calendar day” corresponds to 1:00AM→1:00AM local daylight time rather than midnight→midnight.  
Source: Kalshi weather markets help article.

**Implementation definition:**
- Standard offset is the *winter* offset for the zone:
  - America/New_York → -05:00
  - America/Chicago → -06:00
  - America/Los_Angeles → -08:00
- Store this per station as `standard_offset_minutes` (e.g., -300, -360, -480)

### Target date (T) — `targetDateLocal`
A calendar date (YYYY-MM-DD) in the **station’s local calendar** (e.g., Miami local date).

### Settlement “day window” (for CLI truth) — local standard time day
To compare apples-to-apples with the CLI daily high:
- Define the CLI day window for `targetDateLocal = T` as:
  - Start: `T 00:00` at the station’s **standard offset**
  - End: `(T+1) 00:00` at the station’s **standard offset**

**Notes:**
- During DST, local clocks are 1 hour ahead of standard time.
- So the above window appears as:
  - Start: `T 01:00` local daylight time
  - End: `(T+1) 01:00` local daylight time
- On DST transition days, the duration may be 23 or 25 hours in local clock terms, but this is correct for “standard time day.”

IEM states the same concept: daily climate report totals reflect midnight local standard time; during DST that maps to 1AM→1AM local daylight time.  
Source: IEM CLI table notes.

### “as-of” timestamp for forecast availability — `asOfUtc`
An instant in UTC representing the latest time we’re allowed to have “known” data when forecasting day T.

**Canonical policy for this project:**
- Default: `asOfLocalTime = 23:00` at station timezone on date (T-1)
- `asOfLocal = (T-1) at 23:00` in `stationZoneId`
- Convert to UTC → `asOfUtc`

This is configurable via DB table `asof_policy`.

## 2) No-leakage rule (hard requirement)
For every model feature used to predict day T:
- The MOS model run chosen must satisfy:
  - `mos_run.runtime_utc <= asOfUtc`
- If no run exists, feature is NULL and marked missing.

## 3) Required timestamps stored in DB for auditing
Every persisted forecast feature row must store:
- `target_date_local` (DATE)
- `asof_utc` (TIMESTAMP)
- `asof_local` (TIMESTAMP + zone stored separately)
- `station_zoneid` (string)
- `chosen_run_utc` (TIMESTAMP) — the MOS run that produced this feature
- `retrieved_at_utc` (TIMESTAMP) — when we fetched the MOS run from IEM

Every CLI daily truth row must store:
- `target_date_local` (DATE) — as the climate report date
- `tmax_f` (DECIMAL)
- `report_issued_at_utc` (TIMESTAMP if available, else NULL)
- `retrieved_at_utc` (TIMESTAMP)
- `raw_payload_hash` / raw reference

## 4) Worked example: Miami, DST vs standard
Station: KMIA
ZoneId: America/New_York
Standard offset: -05:00

Target date T = 2025-07-10 (DST in effect)
Settlement day window (standard time day):
- StartUtc = 2025-07-10 00:00 at -05:00 → 2025-07-10 05:00Z
- EndUtc   = 2025-07-11 00:00 at -05:00 → 2025-07-11 05:00Z
Local clock view (EDT is -04:00):
- StartLocal = 2025-07-10 01:00 EDT
- EndLocal   = 2025-07-11 01:00 EDT

So the CLI “daily max for 2025-07-10” is the max over that window.

## 5) Worked example: as-of time
Same station/date:
- asOfLocal = 2025-07-09 23:00 EDT
- asOfUtc   = 2025-07-10 03:00Z
A MOS run at 2025-07-10 06:00Z must NOT be used. A run at 2025-07-10 00:00Z can be used.

