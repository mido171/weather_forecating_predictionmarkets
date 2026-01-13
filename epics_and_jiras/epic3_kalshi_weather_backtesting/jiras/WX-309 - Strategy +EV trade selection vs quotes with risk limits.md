# WX-309 — Strategy: +EV trade selection from bin probabilities vs Kalshi quotes (with risk limits)

## Objective
Implement the “trade decision” module that, given:
- current market quotes (best bid/ask) at time t
- model probabilities P_yes per market (computed as-of)
returns an ordered list of simulated trades to execute.

## Definitions (must be explicit)
Let:
- `p = model_prob_yes` for a market
- `a_yes = best YES ask price` at time t (dollars)
- `a_no  = best NO ask price` at time t (dollars)
- `fee(qty, price)` from WX-306

Expected value per contract (gross, ignoring fees):
- Buy YES: `EV_yes = p - a_yes`
- Buy NO:  `EV_no  = (1 - p) - a_no`

Net EV per contract:
- `EV_net = EV_gross - fee_per_contract`

## Requirements

### 1) Input contract
`select_trades(event_ctx, decision_inputs, quotes_by_market, portfolio_state, config) -> list[ProposedTrade]`

Where `quotes_by_market` includes:
- yes_bid, yes_ask
- no_bid, no_ask (if available; if not available, derive conservatively or skip)

### 2) Trade candidate generation
For each market at time t:
1) If ask is missing/null → skip
2) Compute EV_yes_net for BUY_YES at ask
3) Compute EV_no_net for BUY_NO at ask
4) If either EV exceeds `config.min_ev_per_contract`:
   - create a candidate trade with:
     - side, price, qty (to be sized later)
     - EV_net_per_contract
     - implied_prob = ask (optional)

### 3) Risk constraints (minimum)
Configurable limits:
- `max_contracts_per_event`
- `max_contracts_per_market`
- `max_total_open_contracts`
- `max_notional_per_event_usd` (optional)
- `max_trades_per_event_per_day`

Constraints enforcement:
- The selector must never propose trades that violate limits given current open positions.

### 4) Position sizing (minimum viable)
Provide at least two sizing modes:
- `FIXED_QTY`: always trade N contracts when EV>threshold
- `EV_PROPORTIONAL`: qty proportional to EV, capped by risk limits

Kelly sizing can be added later, but not required for MVP.

### 5) Conflict handling
- Do not allow both BUY_YES and BUY_NO on the same market at the same timestamp.
- Optional: prevent holding conflicting exposure across mutually-exclusive bins unless explicitly allowed.

### 6) Traceability
Every proposed trade must include:
- decision_ts_utc
- p_yes used
- quote used (ask)
- EV computed (gross & net)
- fee assumed
- reason code (e.g., EV_THRESHOLD, LIQUIDITY, RISK_LIMIT)

## Acceptance Criteria
- [ ] Unit tests for EV math with concrete numbers.
- [ ] Unit test that risk limits cap trade qty.
- [ ] Trade proposals include full traceability fields and deterministic ordering.
- [ ] Strategy produces 0 trades when EV thresholds not met.
