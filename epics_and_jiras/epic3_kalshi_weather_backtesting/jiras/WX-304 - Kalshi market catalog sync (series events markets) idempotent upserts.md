# WX-304 — Kalshi market catalog sync (series → events → markets) with idempotent upserts

## Objective
Implement a repeatable “catalog sync” job that downloads Kalshi market metadata for our configured weather series and persists it to MySQL.

This job is the foundation for:
- identifying all historical events/dates available for backtesting
- obtaining strike/bin definitions for probability mapping
- obtaining settlement results (`result`) for P&L accounting

## Data sources (official)
- `GET /series/{series_ticker}` — series metadata
- `GET /markets?series_ticker=...` — list markets in the series (paginated)

See `docs/kalshi-api.md`.

## Requirements

### 1) Inputs
- A configuration list of series tickers we care about (at minimum):
  - `KXHIGHMIA`, `KXHIGHNY`, `KXHIGHPHIL`, `KXHIGHCHI`, `KXHIGHLAX`
- Environment config:
  - Kalshi env (prod/demo)
  - optional API key for higher rate limits

### 2) Algorithm (required)
For each `series_ticker`:
1) Fetch series metadata:
   - store in existing `kalshi_series` (Epic #1 already has this table)
2) Fetch markets list:
   - call `GET /markets?series_ticker=<series>&limit=1000`
   - follow cursor until complete
3) Upsert each market into `kalshi_market`:
   - preserve strike fields: `strike_type`, `floor_strike`, `cap_strike`, etc.
   - preserve timestamps: `open_time`, `close_time`, `expiration_time`
   - preserve settlement: `status`, `result`, `settlement_ts`
4) Upsert event rows into `kalshi_event`:
   - create/update `kalshi_event` for every distinct `event_ticker` observed
   - if event metadata is not included in the market list response, store minimal event rows (event_ticker + series_ticker)
   - optional enhancement: call `GET /events/{event_ticker}` to enrich with strike_date/open/close

### 3) Idempotency & correctness
- Upserts must be keyed on:
  - `kalshi_market.market_ticker`
  - `kalshi_event.event_ticker`
- Sync can be run multiple times; it should only update changed fields and timestamps.

### 4) Auditing
- Store `raw_json` for market rows (optional but strongly recommended).
- Store `updated_at_utc` to show when we last synced each market.

### 5) Output
- A “catalog sync report” (log + summary JSON) with:
  - # markets fetched per series
  - # events discovered per series
  - earliest and latest `close_time_utc` per series
  - # markets updated vs inserted

## Acceptance Criteria
- [ ] Running the job twice produces the same row counts (idempotent).
- [ ] Cursor pagination is used; job handles >1000 markets.
- [ ] For a settled historical market, `result` is persisted (yes/no) and not overwritten with null.
- [ ] A unit test verifies upsert logic preserves non-null settlement fields.
