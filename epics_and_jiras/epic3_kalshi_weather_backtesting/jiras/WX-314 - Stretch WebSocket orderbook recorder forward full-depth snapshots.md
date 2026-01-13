# WX-314 (Stretch) — WebSocket orderbook recorder for forward data (full-depth snapshots for future backtests)

## Objective
Implement a forward-only recorder that connects to Kalshi WebSockets and records:
- orderbook deltas (full depth)
- trade prints (if subscribed)
so that future backtests can use sub-minute full-depth data.

This does NOT solve historical full-depth backtesting, but it enables it going forward.

## References (official)
- WebSocket quick start: https://docs.kalshi.com/getting_started/quick_start_websockets
- Auth signing: https://docs.kalshi.com/getting_started/quick_start_authenticated_requests

## Requirements
### 1) Connection + auth
- Use `wss://api.elections.kalshi.com/trade-api/ws/v2` (prod) or demo.
- WebSocket requires the same signing headers:
  - sign message: `timestamp + "GET" + "/trade-api/ws/v2"`
  - timestamp in ms
  - RSA-PSS SHA256 base64

### 2) Subscriptions
- Subscribe to:
  - orderbook delta channel for the set of market tickers for weather events
- Handle reconnects and resubscribe.

### 3) Book reconstruction
- Maintain an in-memory orderbook per market ticker:
  - price → quantity for YES and NO
- Apply deltas in-order.
- Periodically snapshot (e.g., every 5 seconds) to storage.

### 4) Storage
Because data volume is high, prefer:
- append-only compressed files (Parquet/JSONL.gz) on disk
- DB only for snapshot indexes/metadata

### 5) Acceptance Criteria
- [ ] Recorder can run for 1 hour without crashing.
- [ ] Snapshots are written and can be replayed to reconstruct book state.
- [ ] Reconnect logic works (simulate network drop).
