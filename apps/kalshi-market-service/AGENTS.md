# Kalshi market service agent rules

The root `AGENTS.md` applies. Read this service's `README.md`, execution/trading
properties, security filters, exposure guards, and focused tests before work.

- Default environment is DEMO. Authentication, trading, WebSocket streaming,
  inference, smoke/backtest runners, reconciliation, and live controllers stay
  disabled unless separately and explicitly armed.
- Production requires the exact acknowledgement contract. Trading additionally
  requires authentication, risk limits, and a healthy fail-closed exposure
  guard. Never place an order as a test.
- Sensitive/mutating endpoints require explicit registration and the local
  control token; CORS remains localhost-only.
- Python/inference subprocesses need bounded runtime, concurrency one, cooldown,
  caller authorization, output limits, and an exact stop path.
- WebSocket reconnect, refresh, retries, and polling are finite. Startup
  reconciliation failure must preserve the global buy halt.
- Use mocked/offline focused tests. Live account, orderbook, or order calls need
  explicit user authorization and a written financial/network budget.
