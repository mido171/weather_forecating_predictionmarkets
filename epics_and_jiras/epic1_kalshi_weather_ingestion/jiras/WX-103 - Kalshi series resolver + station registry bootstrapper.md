# WX-103 — Kalshi series resolver + station registry bootstrapper

## Objective
Implement a component that:
1) Given a Kalshi `series_ticker` (e.g., `KXHIGHMIA`),
2) Fetches series metadata from Kalshi API,
3) Extracts settlement source URL(s),
4) Parses `site` and `issuedby`,
5) Maps `issuedby` → candidate IEM station id (ICAO, usually `K` + issuedby),
6) Persists `kalshi_series` and `station_registry` records.

## Inputs
- `series_ticker` (string)
- Optional: overrides config (DB `station_override`)

## Required series (must work)
- KXHIGHNY, KXHIGHPHIL, KXHIGHMIA, KXHIGHCHI, KXHIGHLAX

## Kalshi API endpoint
- `GET https://api.elections.kalshi.com/trade-api/v2/series/{series_ticker}`

Kalshi series JSON includes:
- `settlement_sources[]` with `name` and `url`
- `contract_terms_url` (PDF)

## Parsing logic (MUST be deterministic)
From settlement source url:
`https://forecast.weather.gov/product.php?site=<WFO>&product=CLI&issuedby=<ISSUEDBY>`
Extract:
- `wfo_site = site`
- `issuedby = issuedby`

Then default mapping:
- If `issuedby` is 3 letters: `station_id = "K" + issuedby` (e.g., MIA → KMIA)
- Store `issuedby` and `station_id` separately.

## Timezone assignment (day-1 default)
Populate `station_registry.zone_id` using the built-in defaults:
- KNYC/KPHL/KMIA → America/New_York
- KMDW → America/Chicago
- KLAX → America/Los_Angeles
Allow override via `station_override`.

## Acceptance Criteria
- [ ] `station_registry` rows created for each series ticker above.
- [ ] The stored settlement source URLs exactly match the ones returned by Kalshi.
- [ ] Parsing unit tests cover:
  - valid settlement URL
  - missing query params (fail fast)
  - unexpected format (fail fast)
- [ ] The component is safe to re-run (upsert; no duplicates).
- [ ] On every run, `retrieved_at_utc` is updated and raw JSON stored (optional).

## Source references
- Kalshi “Get Series” endpoint format and fields are documented in Kalshi API docs.
- Example KXHIGHNY settlement source points to `issuedby=NYC` and is described as Central Park in Kalshi quick-start.

