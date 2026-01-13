# agents.md (Epic #3) — Backtesting Handoff Rules

## Scope
This module implements **historical Kalshi market data backfills** and a **probabilistic backtest engine** for the weather series we trade.

## Non‑negotiables
1) **Read and follow**:
   - Epic #1 `docs/time-semantics.md`
   - Epic #2 `docs/model-contract.md` (inputs/outputs of μ̂, σ̂)
2) **No leakage**
   - Any forecast feature used at time `tDecisionUtc` must have `runtimeUtc <= tDecisionUtc`.
   - Any price used must have `price_ts <= tDecisionUtc`.
3) **Deterministic**
   - Backtest runs must set and store a global seed in the run manifest.
4) **Idempotent**
   - Backfills and backtests must be restartable and must not duplicate rows.
5) **Auditable**
   - Every backtest trade must store: market ticker, side, qty, price, decision timestamp, fee, and the probability used.

## Implementation defaults (unless a Jira overrides)
- “Most frequent historical quote” = **1‑minute candlesticks** (Kalshi period_interval=1).
- Execution model default = **taker** at the current best ask/bid (from candlestick).
- Exit model default = **hold to settlement** (no early closing).
