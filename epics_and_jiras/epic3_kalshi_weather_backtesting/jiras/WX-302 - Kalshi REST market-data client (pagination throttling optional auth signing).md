# WX-302 — Kalshi REST market-data client (pagination + throttling + optional auth signing)

## Objective
Implement a production-grade Kalshi REST client for **market data retrieval** that supports:
- cursor-based pagination
- client-side rate limiting
- retries/backoff on 429 + transient 5xx
- optional authenticated signing (for endpoints that require auth, and for higher rate limits)

## Why this matters
Epic #3 needs to download large volumes of historical market data safely:
- market catalogs for series (thousands+ markets)
- candlestick history
- optional trades

Without pagination + throttling, we will get incomplete data or be rate-limited.

## Must-follow references (official)
- Pagination: https://docs.kalshi.com/getting_started/pagination
- Rate limits: https://docs.kalshi.com/getting_started/rate_limits
- Auth signing: https://docs.kalshi.com/getting_started/quick_start_authenticated_requests

## Requirements

### 1) Client configuration
Support:
- `env`: `prod|demo`
- `base_url`: default based on env
- `timeout_seconds`
- `max_retries`
- `rate_limit_reads_per_sec` (default from tier; configurable)
- optional auth:
  - `api_key_id`
  - `private_key_pem` OR `private_key_path`

### 2) HTTP primitives
Implement:
- `get(path, query_params, auth_required=False)`:
  - builds URL
  - (if auth enabled) signs request:
    - timestamp in **ms**
    - message = `timestamp + HTTP_METHOD + path_without_query`
    - signature = RSA-PSS SHA256 base64
    - headers: `KALSHI-ACCESS-KEY`, `KALSHI-ACCESS-TIMESTAMP`, `KALSHI-ACCESS-SIGNATURE`
  - applies rate-limiter
  - retries:
    - 429: exponential backoff with jitter
    - 5xx: limited retries
  - returns parsed JSON + response metadata (status, headers)

### 3) Pagination helper
Generic:
- `paginate(endpoint, query_params, page_limit=1000)`:
  - loops `cursor` until null
  - yields pages/items
  - logs progress

Must support endpoints used in this epic:
- `/markets`
- `/events` (optional)
- `/series` (optional)
- `/markets/trades`

### 4) Endpoint wrappers (minimum)
Implement typed wrapper methods (thin):
- `get_series(series_ticker)`
- `list_markets(series_ticker, status, min_close_ts, max_close_ts, cursor, limit)`
- `get_event_candlesticks(series_ticker, event_ticker, start_ts, end_ts, period_interval)`
- `batch_market_candlesticks(tickers[], start_ts, end_ts, period_interval, include_latest_before_start)`
- `list_trades(ticker, min_ts, max_ts, cursor, limit)`

### 5) Logging + diagnostics
- Each request logs:
  - method, path, query (redact secrets)
  - latency
  - status code
  - retry count
- On persistent failure, raise a typed exception that includes the request context.

## Acceptance Criteria
- [ ] A unit test demonstrates cursor pagination using mocked responses.
- [ ] A unit test demonstrates signature creation:
  - query params are NOT included in the signed path
  - timestamp uses ms
- [ ] A “smoke” integration test script exists (manual run) that can:
  - list the first page of markets for a series ticker (e.g., `KXHIGHNY`)
- [ ] Client exposes a single place to configure read rate limit and concurrency.
