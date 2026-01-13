# WX-308 — Backtest simulation engine core (deterministic time loop + portfolio state)

## Objective
Implement the core simulation engine that:
- iterates through time in fixed steps (default 1-minute)
- maintains portfolio state (cash, positions, realized P&L, fees)
- records every simulated trade with full audit fields
- resolves positions at settlement using Kalshi market results

This Jira builds the engine skeleton; trade selection logic comes in WX-309.

## Requirements

### 1) Core domain objects (Python)
Define internal classes/structs (dataclasses or pydantic):
- `BacktestConfig`
- `RunManifest`
- `EventContext`
- `MarketQuote` (bid/ask at time t)
- `DecisionInput` (P_yes per market at asOf)
- `SimTrade` (executed trade record)
- `Position` (open position)
- `PortfolioState` (cash, open positions, realized pnl, fees)

### 2) Deterministic time loop
For each event:
1) compute `tradeStartUtc` and `tradeEndUtc` per `docs/time-semantics-backtest.md`
2) generate a list of decision timestamps:
   - default: every minute boundary, aligned to candlestick `end_period_ts`
3) for each timestamp:
   - load quotes for all markets at that timestamp (from candlesticks table)
   - call strategy module (WX-309) to get desired trades
   - apply fill + fee model (from `docs/fees-and-fills.md`)
   - update portfolio state
4) after trade loop:
   - resolve event settlement:
     - use `kalshi_market.result` for each market
     - compute payout and realized P&L
   - close positions

### 3) Persistence
- Write:
  - `backtest_run` row at start (RUNNING)
  - `backtest_trade` rows as trades occur
  - `backtest_run.metrics_json` + status SUCCESS/FAILED at end
- Write run artifacts to `backtests/<run_id>/` as defined in `docs/backtest-metrics.md`.

### 4) Failure handling
If an event is missing:
- decision inputs (probabilities) OR
- quote data (candles)
Then the engine must:
- skip the event (configurable) OR fail (configurable)
- always record the skip reason in the report

## Acceptance Criteria
- [ ] A deterministic unit test:
  - fixed seed
  - small synthetic event with 2 markets and 5 timestamps
  - produces identical trades and P&L across runs
- [ ] Engine writes a `run_manifest.json` including:
  - seed
  - config snapshot
  - data watermark (latest candle timestamp used)
- [ ] Engine resolves positions using Kalshi market results and computes correct payouts.
