# Fees & Fill Modeling (Epic #3)

Backtests are only as good as the execution assumptions. This doc defines the **default** execution model and fee model.

## 1) Default execution model (taker)
Unless configured otherwise:
- We assume every trade is a **taker** trade that crosses the spread:
  - Buy YES at best ask
  - Sell YES at best bid (equivalently buy NO at best ask, depending on implementation)
- We assume immediate fill at the quoted best price at that timestamp (no queue modeling).

**Why:** With only historical candlesticks, we reliably have best bid/ask time series. We do not have full depth or queue position.

## 2) “Orderbook sampling frequency”
Historical top-of-book is obtained from **candlestick** endpoints at:
- `period_interval = 1` minute (most frequent historical series exposed by Kalshi candlestick endpoints).

## 3) Fee model (Kalshi published quadratic fee)
Kalshi publishes a “quadratic fee” structure.

### Taker fees (default)
For a trade of C contracts executed at price P (in **dollars**, 0..1):
- `fee = ceil( 0.07 × C × P × (1 − P) )`
- fee is in **dollars**

### Maker fees (optional future)
If we implement maker simulation:
- `fee = ceil( 0.0175 × C × P × (1 − P) )`

### Rounding rule
- Use a strict “round up” (ceiling) to the smallest currency unit supported by the backtest engine:
  - default: **$0.01** increments (cent rounding)
  - if Kalshi fees are tracked with more precision, store raw float and a rounded display value

**Important:** fee schedules can change. We will:
- store the fee constants used in each backtest run manifest
- optionally ingest `series_fee_changes` for future improvements

## 4) P&L accounting (hold-to-settlement default)
If you BUY YES:
- Cost at entry: `C × P_entry`
- Settlement payout: `C × 1` if YES occurs else `0`
- Profit: `payout − cost − fee_entry`  (and fee_exit if we later implement exits)

If you BUY NO at price `P_no_entry`:
- Settlement payout: `C × 1` if NO occurs else `0`
- Profit: `payout − (C × P_no_entry) − fee_entry`

**Note:** Kalshi quotes both YES and NO. Prefer using the side’s explicit ask/bid rather than assuming `P_no = 1 − P_yes` (spread breaks that identity).

## 5) Optional higher-fidelity fill model (future)
If trade prints are ingested:
- A conservative fill assumption:
  - A taker buy at time t fills at the first trade price >= ask within Δ seconds (configurable), else no fill.
- This requires consistent trade timestamps and may reduce optimistic fills.

This is out of scope for Epic #3 unless explicitly added as a Jira.
