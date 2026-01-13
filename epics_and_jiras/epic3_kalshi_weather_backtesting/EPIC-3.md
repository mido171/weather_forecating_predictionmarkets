# EPIC #3 — Kalshi Market Backtesting (Weather Range Markets)

## Goal
Build a **reproducible, idempotent, and auditable** backtesting system that simulates trading Kalshi “daily high temperature” markets for the Kalshi settlement stations we trade (MIA / NYC / PHL / MDW / LAX), using the **Epic #2 probabilistic forecast models** (μ̂ and σ̂ → bin probabilities) and **historical Kalshi market prices**.

This epic answers: “If we had our T-1 forecast distribution for day T, what trades would we have taken given the Kalshi market prices available at the time, and what would the P&L have been at settlement?”

## Non‑negotiables
1) **No leakage**  
   Any decision made at time `tDecisionUtc` may only use:
   - MOS / forecast features whose `runtimeUtc <= tDecisionUtc`
   - Kalshi market prices whose timestamp `<= tDecisionUtc`
   - Never use CLI settlement values before they are officially published.

2) **Time semantics MUST match Epic #1**  
   - “Target date T” is the station **standard‑time day** used by NWS CLI.
   - Default as‑of policy: `asOfLocal = (T-1) 23:00` in station timezone → `asOfUtc`.
   - These rules are already defined in Epic #1 `docs/time-semantics.md`. Epic #3 must reuse them.

3) **Idempotent ingest & backtest**  
   - A backfill or backtest run can be killed and restarted without duplication or corruption.
   - Every stored row has a natural key and upsert logic.
   - Every run writes a run manifest and deterministic seed.

4) **Auditable**  
   For any backtest result, we can reconstruct:
   - which Kalshi market tickers were considered
   - what timestamps were used
   - what prices/quotes were used
   - what probabilities were used
   - what fees were applied

## What “Kalshi historical orderbook” means in this epic
Kalshi’s REST API provides:
- **Current** orderbook snapshots (top-of-book + depth) via `/markets/{ticker}/orderbook`
- **Historical** 1‑minute market quote/price series via **candlesticks** endpoints

For *historical* backtesting, we will treat the **1‑minute candlestick best bid/ask** as the “orderbook snapshot at minute resolution.”  
(Deep historical L2 orderbook depth is not available via a timestamped REST endpoint; full-depth backtests require recording WebSocket orderbook deltas going forward.)

## Deliverables
1) **Kalshi market data ingestion**
   - Series → events → markets metadata sync (for configured series tickers)
   - Candlestick backfill for each event (1‑minute default, configurable)
   - Optional trade prints backfill for higher‑fidelity fill modeling

2) **Backtest engine**
   - Deterministic time loop (minute cadence by default)
   - +EV scanning logic across all markets in an event
   - Configurable execution model (taker at ask/bid; maker optional later)
   - Fee model applied per Kalshi published formula
   - Portfolio + risk constraints

3) **Reporting**
   - Machine output: JSON metrics + CSV trades
   - Human output: markdown report with charts and key diagnostics
   - Metrics: P&L, profit factor, win%, drawdown, Sharpe (optional), calibration diagnostics

## Data sources
Kalshi API endpoints used (see `docs/kalshi-api.md`):
- List markets with filters/pagination (`GET /markets`)
- Event candlesticks for historical best bid/ask (`GET /series/{series_ticker}/events/{event_ticker}/candlesticks`)
- (Optional) Trade prints (`GET /markets/trades`)
- Rate limits, pagination, and authenticated request signing are handled per Kalshi docs.

Kalshi fee formula used (see `docs/fees-and-fills.md`):
- fees = round_up(0.07 × C × P × (1 − P)) for taker trades
- maker fee variant where applicable

## Out of scope (explicit)
- Live trading execution (order placement) — that is a separate production trading epic.
- Perfect historical L2 orderbook reconstruction (not feasible with official REST alone).
- Alternative settlement sources (we settle to NWS CLI per Kalshi rules).

## Definition of Done
- A single command can backtest:
  - one station series ticker (e.g., `KXHIGHMIA`)
  - a date range
  - using a configured as‑of policy
- Results include a full audit trail and deterministic reproduction instructions.
