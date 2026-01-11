# Station Mapping Strategy (Kalshi -> NWS CLI -> IEM)

## 1) Kalshi settlement sources
Kalshi series metadata includes `settlement_sources[].url` pointing to the
NWS Daily Climate Report (CLI) page on forecast.weather.gov.

Series endpoint example:
- https://api.elections.kalshi.com/trade-api/v2/series/KXHIGHMIA

Settlement source URL pattern:
- https://forecast.weather.gov/product.php?site=<WFO>&product=CLI&issuedby=<ISSUEDBY>

We parse and store:
- `site` (WFO identifier, stored as `wfo_site`)
- `issuedby` (3-letter station code, stored as `issuedby`)

## 2) Mapping to IEM station IDs
IEM MOS archive and IEM CLI JSON typically use ICAO identifiers:
- `K` + 3-letter issuedby code

Rule:
- If `issuedby` is 3 letters and does not start with `K`, map to `K{issuedby}`.
- Store both values in `station_registry`.

## 3) Validation step (required)
Validate the mapping by calling IEM CLI:
- `/json/cli.py?station=<ICAO>&year=<YYYY>&fmt=json`

If data is returned, set `mapping_status=AUTO_OK`.
If no data, set `mapping_status=NEEDS_OVERRIDE` and require a row in
`station_override` with the correct ICAO and zone id.

## 4) Initial station set for this repo
Derived from Kalshi series endpoints:
- KXHIGHNY -> issuedby=NYC -> KNYC, WFO=OKX
- KXHIGHPHIL -> issuedby=PHL -> KPHL, WFO=PHI
- KXHIGHMIA -> issuedby=MIA -> KMIA, WFO=MFL
- KXHIGHCHI -> issuedby=MDW -> KMDW, WFO=LOT
- KXHIGHLAX -> issuedby=LAX -> KLAX, WFO=LOX

## 5) Default time zone mapping
These ship as defaults and can be overridden in DB:
- KNYC -> America/New_York
- KPHL -> America/New_York
- KMIA -> America/New_York
- KMDW -> America/Chicago
- KLAX -> America/Los_Angeles
