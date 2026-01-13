# WX-305 — Kalshi event candlesticks backfill (1-minute top-of-book) with checkpoints + upserts

## Objective
Backfill historical Kalshi **1-minute** candlestick data for weather events and persist it in MySQL so backtests can run offline (no API calls during simulation).

We treat 1-minute candlestick best bid/ask as the “orderbook snapshot” for backtesting.

## Data sources (official)
- `GET /series/{series_ticker}/events/{event_ticker}/candlesticks`
  - parameters: `start_ts`, `end_ts`, `period_interval`
  - if the request would exceed the API’s max candles, response includes `adjusted_end_ts` → continue from there

Optional alternative:
- `GET /markets/candlesticks` (batch market candlesticks)

## Requirements

### 1) Inputs
- `series_ticker`
- derived list of `event_ticker` from `kalshi_event` table
- configuration:
  - `period_interval_minutes` default **1**
  - `window_policy`:
    - `FULL_MARKET_WINDOW`: [open_time, close_time]
    - `TRADE_WINDOW_ONLY`: [tradeStartUtc, tradeEndUtc] using Epic #1 as-of policy
  - max concurrency

### 2) Backfill algorithm (required)
For each event:
1) Determine `start_ts` and `end_ts` (unix seconds):
   - based on window_policy
2) Checkpoint lookup:
   - if `kalshi_backfill_checkpoint` indicates SUCCESS for this window, skip
3) Fetch candlesticks:
   - call event candlesticks endpoint
   - if response includes `adjusted_end_ts`:
     - persist returned candles
     - set next `start_ts = adjusted_end_ts`
     - loop until done
4) Upsert each candle row:
   - key: `(market_ticker, end_period_ts_utc, period_interval_minutes)`
   - store yes_bid/ask OHLC, volume, open_interest, retrieved_at_utc
5) Mark checkpoint SUCCESS for the event window.

### 3) Idempotency
- Backfill can be aborted mid-event and restarted:
  - if partial rows exist, upsert must not duplicate
  - checkpoint must move from RUNNING → SUCCESS only after the full window is complete

### 4) Missing data handling
Candles may contain null OHLC fields (illiquid periods or synthetic “previous_price” behavior). Persist nulls explicitly.

### 5) Output & observability
For each event:
- log # candles saved per market
- log # markets present in response
- log missing-candle statistics (how many end_period_ts missing in the window)

## Acceptance Criteria
- [ ] Backfill can run for a multi-month range without exceeding rate limits (throttling works).
- [ ] Killing the job mid-run and restarting does not increase row counts beyond expected (idempotent).
- [ ] At least one integration run demonstrates candles persisted for an event and can be queried by (market_ticker, time range).
- [ ] Code path supports both FULL_MARKET_WINDOW and TRADE_WINDOW_ONLY.
