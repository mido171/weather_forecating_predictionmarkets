# WX-307 — Build backtest “decision inputs”: join Epic #2 forecast distribution with Kalshi market bins (P_yes per market)

## Objective
For each Kalshi weather event (daily high temp):
- compute the Epic #2 probabilistic forecast distribution **as-of** T-1
- map that distribution to **P(YES)** for every Kalshi market/bin in the event
- persist (or compute on the fly) a structured “decision input” object used by the trading simulator

## Why this matters
Backtesting requires apples-to-apples:
- prices at time t
- probabilities computed from information available at time t (no leakage)
- probabilities aligned to Kalshi’s exact strike/bin definitions

## Inputs
- `station_id` (e.g., KMIA) — from station registry
- `series_ticker` (e.g., KXHIGHMIA)
- `event_ticker`
- `targetDateLocal` (resolved from event)
- `asOfUtc` (from Epic #1 time library)

## Required algorithm
1) Load Epic #2 persisted models:
   - μ̂ model (predict mean CLI Tmax)
   - σ̂ model (predict uncertainty)
2) Build the feature vector for the target date T using Epic #1 dataset tables:
   - MOS as-of features (must satisfy runtimeUtc <= asOfUtc)
   - any lagged climatology features if implemented
3) Compute:
   - `mu_hat_f`
   - `sigma_hat_f` (must be positive; enforce min floor like 0.25°F)
4) For each market in the event:
   - read its strike definition from `kalshi_market`:
     - `strike_type`
     - `floor_strike` / `cap_strike` etc
   - compute P_yes = Prob( Tmax falls in that strike )
     - Example:
       - between [a, b): `P = CDF((b-mu)/sigma) - CDF((a-mu)/sigma)`
       - lt a: `P = CDF((a-mu)/sigma)`
       - gt a: `P = 1 - CDF((a-mu)/sigma)`
5) Validation:
   - If the event is a mutually-exclusive range set:
     - sum(P_yes over all markets in event) should be within [0.98, 1.02]
     - if not, emit diagnostics (likely strike gaps/overlaps or wrong parsing)
6) Output object (in-memory and optionally persisted):
   - one row per market:
     - station_id, series_ticker, event_ticker, market_ticker
     - asOfUtc
     - mu_hat_f, sigma_hat_f
     - p_yes
     - notes (missing features, etc)

## No-leakage guardrails (hard)
- Any MOS runtime used must satisfy `runtimeUtc <= asOfUtc`
- If no valid MOS run exists for a feature:
  - feature=NULL
  - model still runs if it can handle nulls, else skip event

## Acceptance Criteria
- [ ] A unit test validates probability mapping for:
  - between bin
  - lt tail
  - gt tail
- [ ] For at least one real event with many bins, probabilities sum ~1 and are logged.
- [ ] If any feature runtime is after asOfUtc, the computation fails fast (exception).
