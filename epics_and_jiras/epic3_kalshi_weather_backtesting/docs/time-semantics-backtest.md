# Time Semantics for Backtesting (Epic #3)

Epic #3 inherits the canonical time semantics from Epic #1 and adds *trading-time* semantics.

## 1) Canonical definitions (from Epic #1)
- Station timezone: `ZoneId` (e.g., America/New_York)
- Station “standard-time day window” for CLI Tmax on local date T:
  - Start = `T 00:00` at station **standard offset**
  - End   = `(T+1) 00:00` at station **standard offset**
- Default as-of time for forecasting day T:
  - `asOfLocal = (T-1) 23:00` in station timezone
  - `asOfUtc = toUtc(asOfLocal)`

These MUST be reused, not redefined.

## 2) Kalshi event date mapping (critical)
Kalshi market data includes:
- `series_ticker` (e.g., `KXHIGHMIA`)
- `event_ticker` (string)
- each market belongs to an event

We must map each Kalshi event to a station local target date T.

### Required algorithm (robust)
For each market row:
1) Prefer the explicit `event_ticker` field from the API.
2) Determine `targetDateLocal` by parsing the date token **if present**:
   - Many Kalshi weather tickers include a token like `25DEC11` (YYMMMDD).
   - If parse succeeds, interpret it as station local date.
3) If parsing fails, fallback to event metadata:
   - call `GET /events/{event_ticker}` (or use event metadata available in market rows if present)
   - parse `strike_date` and convert to station timezone
   - use the **date component** as `targetDateLocal`
4) Validate:
   - compare `targetDateLocal` to the NWS CLI date list for the station
   - log and quarantine any event that fails validation

**Acceptance rule:** If `targetDateLocal` cannot be resolved deterministically, the event is skipped and a diagnostic is emitted.

## 3) Trading window semantics (no-leakage)
We define a trading simulation window for each event:

### Inputs
- `asOfUtc` (from Epic #1)
- Kalshi market `open_time` and `close_time` (UTC timestamps from API)
- optional config:
  - `tradeWindowEndPolicy = "market_close" | "station_day_end"`

### Default policy
- Start: `tradeStartUtc = max(asOfUtc, open_time_utc)`
- End:   `tradeEndUtc = close_time_utc`

### Safer (no “post-observation” trading) policy
Because a daily high temperature may become known before Kalshi’s official close (Kalshi may keep markets open while waiting for official confirmation), we support:
- End: `tradeEndUtc = min(close_time_utc, stationStandardDayEndUtc(T))`

Where `stationStandardDayEndUtc(T)` is the end of the CLI standard-time day window.

This prevents backtests from accidentally assuming you can trade *after the measured day has ended* using only T-1 forecasts.

## 4) Candlestick timestamp interpretation
Kalshi candlesticks include:
- `end_period_ts` (unix seconds)
- OHLC fields for bid/ask/price at that period

We interpret:
- A 1-minute candle labeled with `end_period_ts = t` represents the interval `(t-60, t]`.
- For decision-making at time `tDecisionUtc`, we only use candles where `end_period_ts <= tDecisionUtc`.

## 5) Example (walkthrough)
Station: MIA (America/New_York)  
Target date: 2026-01-10

1) Compute `asOfLocal`:
- (T-1) 23:00 local → 2026-01-09 23:00 America/New_York
2) Convert to UTC (`asOfUtc`):
- 2026-01-10 04:00Z (because EST is UTC-5)
3) Trade window:
- tradeStartUtc = max(asOfUtc, market_open)
- tradeEndUtc = min(market_close, stationStandardDayEndUtc(T)) if using the safer policy

**No-leakage:** Any data after `tradeEndUtc` is forbidden in this event’s simulation.
