# Station Mapping Strategy (Kalshi → NWS CLI → IEM station IDs)

## 1) What Kalshi provides (source of truth)
For each temperature series, Kalshi’s public API provides:
- `settlement_sources[].url` pointing to the NWS Daily Climate Report (CLI) page on forecast.weather.gov
- Example series endpoint: `GET https://api.elections.kalshi.com/trade-api/v2/series/KXHIGHMIA`

The response includes a settlement source URL of the form:
`https://forecast.weather.gov/product.php?site=<WFO>&product=CLI&issuedby=<ISSUING_STATION>`

We will parse:
- `site` (WFO identifier used by NWS pages)
- `issuedby` (the 3-letter station code used by CLI product ID)

## 2) Mapping to IEM station IDs for MOS + CLI
IEM MOS archive and IEM CLI JSON both use a “station identifier” that, for the U.S., is typically the ICAO code:
- `K` + the 3-letter issuedby code
Examples:
- issuedby=MIA → station=KMIA
- issuedby=LAX → station=KLAX
- issuedby=PHL → station=KPHL
- issuedby=MDW → station=KMDW
- issuedby=NYC → station=KNYC (Central Park station)

**Rule:**
- If `issuedby` is 3 letters and not already starting with `K`, map to `K{issuedby}`.
- Store both values: `issuedby` and `icao_guess`.

**Validation step (mandatory):**
- Call IEM CLI `json/cli.py` for the station/year.
- If station returns data, mapping is valid.
- If not, mark station as `mapping_status=NEEDS_MANUAL_OVERRIDE` and require a row in `station_override`.

## 3) Initial station set for this repo (must work on day 1)
Derived from Kalshi series endpoints:

- KXHIGHNY settlement source uses issuedby=NYC (Central Park) and site=OKX.
- KXHIGHPHIL settlement source uses issuedby=PHL and site=PHI.
- KXHIGHMIA settlement source uses issuedby=MIA and site=MFL.
- KXHIGHCHI settlement source uses issuedby=MDW and site=LOT.
- KXHIGHLAX settlement source uses issuedby=LAX and site=LOX.

We will ingest for these ICAO stations:
- KNYC, KPHL, KMIA, KMDW, KLAX

## 4) Time zone mapping
This repo must ship with a default timezone mapping table (can be overridden in DB):
- KNYC → America/New_York
- KPHL → America/New_York
- KMIA → America/New_York
- KMDW → America/Chicago
- KLAX → America/Los_Angeles

## 5) Why we store BOTH station code types
- Kalshi uses NWS CLI station (“issuedby=XXX”).
- IEM services often use ICAO (“KXXX”).
- Storing both removes ambiguity and makes auditing easy.

