# WX-312 — Validation & correctness suite (no-leakage + comparability checks)

## Objective
Create a validation suite that makes it **hard to get a wrong backtest**.

This is a hard requirement because weather + markets have many subtle time pitfalls.

## Requirements

### 1) Add a `weather_backtest.validate` module with checks
At minimum implement these checks (at least 10):

1) **No leakage (forecast):**
   - For every decision, assert all MOS runtimes used satisfy `runtimeUtc <= decision_ts_utc`.

2) **No leakage (prices):**
   - For every decision at `t`, only use candles with `end_period_ts <= t`.

3) **Event date mapping:**
   - Every event must map to exactly one `targetDateLocal`. If ambiguous → skip.

4) **Mutual exclusivity (settlement):**
   - For a range event, exactly one market should settle YES (others NO). If not, quarantine event.

5) **Probability mass check:**
   - Sum P_yes across markets in an event is ~1 (configurable tolerance).

6) **Strike coverage check:**
   - Ensure there are no gaps/overlaps in strike intervals unless explicitly allowed.

7) **Fee sanity:**
   - Fee must be >= 0 and <= (qty × 0.07 × 0.25) approx; assert bounds.

8) **Price bounds:**
   - Ask/bid prices must be within [0,1]; reject otherwise.

9) **Idempotency of ingest:**
   - Re-running catalog sync and candle backfill does not change row counts unexpectedly.

10) **Decision window policy enforcement:**
   - Ensure `decision_ts` is within [tradeStartUtc, tradeEndUtc].

11) **DST regression tests:**
   - At least one test date around DST start and end for NY/CHI/LA.

12) **Station series mapping check:**
   - For each series_ticker in config, station_registry mapping exists.

### 2) Validation report output
Every backtest run must include:
- counts of failed validations
- list of quarantined events with reasons

## Acceptance Criteria
- [ ] The validation suite can be run standalone: `python -m weather_backtest.validate --config ...`
- [ ] A failing validation fails the run (unless configured to skip) and leaves clear diagnostics.
- [ ] At least 10 validations are implemented and covered by tests.
