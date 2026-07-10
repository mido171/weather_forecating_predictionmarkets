# Market Microstructure Research

## Separation of concerns

1. weather model estimates true bucket probabilities;
2. market layer observes executable prices and depth;
3. decision layer accounts for costs, uncertainty, and risk.

Do not contaminate weather validation with optimistic execution assumptions.

## Required market archive

- event/market/token mapping;
- exact outcomes;
- rules hash;
- book snapshots;
- deltas/sequence;
- trades;
- best bid/ask;
- tick-size changes;
- fees;
- receive timestamps;
- reconnect/gap events;
- resolution.

## Price comparison

For buying Yes:

```text
gross probability edge = model probability - executable ask
```

Then subtract:

- taker fee;
- expected slippage;
- adverse-selection reserve;
- model uncertainty reserve;
- operational risk reserve.

For maker quotes, estimate fill probability and conditional adverse selection. A rebate does not make a stale quote profitable.

## Mutually exclusive inventory

Positions across buckets are correlated and jointly settle to one winner. Track state-contingent payoff by possible outcome, not nominal position sum alone.

## Backtest limitations

- historical price history is not full depth;
- displayed volume is not executable capacity;
- midpoint is not a fill;
- future knowledge of the winning token must not affect simulated order placement;
- book gaps and latency need conservative treatment.

## Paper-trading requirements

Store every intended order before observing subsequent book movement:

```text
decision_time
book_snapshot_hash
model_version
probabilities
limit price
size
expiry
reason
risk state
simulated fill rule
```

Score fill rules against actual subsequent messages.

## No-trade state

No trade when:

- rules unknown;
- probability advantage below threshold;
- book stale;
- model disagreement extreme;
- critical source missing;
- station reading anomalous;
- exposure/risk limit reached.
