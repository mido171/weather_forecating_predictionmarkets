# Kalshi API Usage Blueprint (Epic #3)

This doc defines exactly which Kalshi endpoints Epic #3 uses and how.

## Base URLs
### Production (real)
- REST: `https://api.elections.kalshi.com/trade-api/v2`
- WebSocket: `wss://api.elections.kalshi.com/trade-api/ws/v2`

### Demo
- REST: `https://demo-api.kalshi.co/trade-api/v2`
- WebSocket: `wss://demo-api.kalshi.co/trade-api/ws/v2`

## Authentication (when needed)
Many market-data endpoints can be accessed without authentication, but authenticated access is required for:
- WebSockets
- portfolio endpoints
- order placement (out of scope for Epic #3)

Kalshi signing headers:
- `KALSHI-ACCESS-KEY`: API Key ID
- `KALSHI-ACCESS-TIMESTAMP`: unix time **milliseconds**
- `KALSHI-ACCESS-SIGNATURE`: base64(signature)

Message to sign (critical):
- `timestamp + HTTP_METHOD + path_without_query`
- Example:
  - `1703123456789GET/trade-api/v2/portfolio/balance`

Signature:
- RSA‑PSS with SHA256
- base64 encode the signature bytes

Reference: official “Quick Start: Authenticated Requests”.

## Pagination (cursor-based)
Kalshi list endpoints use cursor-based pagination:
- Make first request without `cursor`.
- If response includes `"cursor": "<token>"`, request next page with `?cursor=<token>`.
- Stop when cursor is empty/null.

Reference: official “Understanding Pagination”.

## Primary endpoints used

### 1) List markets in a series (discover all daily bins)
`GET /markets`

We use this to get the complete market catalog for a station series (e.g., `KXHIGHMIA`), including:
- market `ticker`
- `event_ticker`
- `title`
- strike fields (`strike_type`, `floor_strike`, `cap_strike`, etc.)
- trading windows (`open_time`, `close_time`, `expiration_time`)
- settlement (`status`, `result`, `settlement_ts`)

Key query parameters we use:
- `series_ticker=<SERIES>`
- `status=open|closed|settled` (primarily `settled` for historical)
- `limit=<1..1000>`
- `cursor=<token>`

### 2) Historical quotes & prices — event candlesticks (recommended)
`GET /series/{series_ticker}/events/{event_ticker}/candlesticks`

We use this to fetch **1‑minute** best bid/ask/price time series for **all markets in the event** in one call.

Query parameters:
- `start_ts` (unix seconds)
- `end_ts` (unix seconds)
- `period_interval` ∈ {1, 60, 1440} minutes
- Response may return `adjusted_end_ts` if too many candles; if present, call again starting from that ts.

This endpoint is the backbone of the historical “orderbook snapshot” approximation.

### 3) Historical quotes & prices — batch market candlesticks (optional)
`GET /markets/candlesticks`

Use when you want candlesticks for a specific set of market tickers across events.

Query parameters:
- `tickers=<comma separated market tickers>`
- `start_ts`, `end_ts` (unix seconds)
- `period_interval` (1/60/1440)

Hard limits (from docs):
- up to 100 tickers
- up to 10,000 total candles per request
- `include_latest_before_start=true` can create a synthetic first candle that may include `previous_price`

### 4) Trades feed (optional, for fill modeling)
`GET /markets/trades`

We may optionally backfill trades to improve fill realism, but the default execution model in Epic #3 is based on best bid/ask from candlesticks.

Query parameters:
- `ticker=<market_ticker>` (filter)
- `min_ts`, `max_ts` (unix seconds)
- `limit` (1..1000)
- `cursor`

### 5) Live orderbook (NOT historical, optional)
`GET /markets/{ticker}/orderbook?depth=<n>`

This is useful for real-time system sanity checks, and for forward recording (via WebSockets) — but it does not provide historical timestamped snapshots.

## Rate limits
Rate limits depend on your API usage tier. Implement:
- client-side throttling
- 429 retry with exponential backoff

Reference: official “Rate Limits and Tiers”.

## Practical ingestion strategy (recommended)
1) Market catalog sync:
   - nightly: `GET /markets?series_ticker=...` paginated
2) Historical backfill:
   - iterate events from earliest available to latest
   - fetch event candlesticks for the configured trade window
   - store candlesticks idempotently
3) Backtest run:
   - only read from DB (no API calls during backtest) unless explicitly enabled
