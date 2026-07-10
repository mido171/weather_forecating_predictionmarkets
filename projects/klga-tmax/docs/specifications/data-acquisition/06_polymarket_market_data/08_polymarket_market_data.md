# 08 — Polymarket Market Data Acquisition

## 1. Purpose

The weather model is only useful if converted into profitable trades. Polymarket market data is required for:

```text
market discovery,
question/rules parsing,
outcome bucket mapping,
orderbook and price snapshots,
edge calculation,
historical trading simulation,
liquidity/slippage estimation,
execution monitoring,
post-trade P&L attribution.
```

Official source pages:

```text
API introduction:      https://docs.polymarket.com/api-reference/introduction
Market data overview:  https://docs.polymarket.com/market-data/overview
Order book endpoint:   https://docs.polymarket.com/api-reference/market-data/get-order-book
Prices history:        https://docs.polymarket.com/api-reference/markets/get-prices-history
Prices/orderbook docs: https://docs.polymarket.com/concepts/prices-orderbook
Trading/orderbook:     https://docs.polymarket.com/trading/orderbook
```

Polymarket documentation states that Gamma API and Data API are public/no-auth, while the CLOB API has public market-data endpoints and authenticated trading endpoints.

## 2. Required APIs

### 2.1 Gamma API

Use Gamma for market metadata and event discovery.

Base:

```text
https://gamma-api.polymarket.com
```

Required endpoints to implement:

```text
GET /events
GET /markets
GET /public-search
```

Codex must discover KLGA/NYC high-temperature markets by querying terms such as:

```text
LaGuardia
KLGA
NYC highest temperature
New York City temperature
highest temperature recorded at the LaGuardia Airport Station
```

### 2.2 CLOB API public market-data endpoints

Base:

```text
https://clob.polymarket.com
```

Required endpoints:

```text
GET /book
POST /books
GET /price
GET /prices
GET /midpoint
GET /spread
GET /prices-history
```

Store every response as bronze raw payload.

### 2.3 Data API

Base:

```text
https://data-api.polymarket.com
```

Use for:

```text
trades
activity
open interest
holders
positions if authenticated/available to user
```

### 2.4 Websocket streaming

If available in the user's implementation environment, subscribe to CLOB market channels for target token ids. Websocket data is required for high-frequency orderbook history after live system launch.

## 3. Market metadata fields to capture

For every candidate weather market, store:

```text
market_id
condition_id
event_id
question
slug
description
resolution_source_text
resolution_url
market_open_time
market_close_time
end_date
active
closed
archived
accepting_orders
enable_order_book
outcomes
outcome_prices
clob_token_ids
minimum_order_size
minimum_tick_size
fee_schedule if exposed
raw_gamma_json
```

Outcome mapping is critical. Codex must create:

```text
outcome_index
outcome_name
clob_token_id
market_bucket_low_f
market_bucket_high_f
market_bucket_type        # bounded_range, below_or_equal, above_or_equal, exact, other
```

For every KLGA weather market, parse the question/rules and store the exact settlement text.

## 4. Bucket parsing rules

Codex must implement a deterministic bucket parser.

Examples:

```text
"84° or lower" -> bucket_type=below_or_equal, low=-inf, high=84
"85-86°"       -> bucket_type=bounded_range, low=85, high=86
"87°"          -> bucket_type=exact, low=87, high=87
"91° or higher" -> bucket_type=above_or_equal, low=91, high=inf
```

The parser must handle:

```text
°F symbols
F suffixes
words such as lower, below, under, higher, above, or more
hyphenated ranges
integer-only temperatures
```

If parsing fails, set:

```text
bucket_parse_status = "failed"
```

and do not trade that market until manually resolved.

## 5. Orderbook snapshot acquisition

### 5.1 Required snapshot times

For every active KLGA market and every outcome token:

```text
Every 1 minute while the market is active.
Every 10 seconds during the final 2 hours before target local day starts if trading is enabled.
At every model forecast cutoff, force-refresh all outcome books immediately before and after forecast generation.
At every detected large price move >= 3 cents in any outcome, snapshot all outcomes.
```

### 5.2 Required fields from orderbook responses

```text
market_condition_id
asset_id / token_id
timestamp
hash
bids
asks
last_trade_price
raw_response_json
retrieved_at_utc
```

Bids/asks must be stored as depth levels:

```text
side
price
size
level_index
```

### 5.3 Silver schemas

```text
CREATE TABLE poly_markets (
    market_id TEXT PRIMARY KEY,
    condition_id TEXT,
    event_id TEXT,
    question TEXT NOT NULL,
    slug TEXT,
    description TEXT,
    resolution_source_text TEXT,
    resolution_url TEXT,
    market_open_time_utc TIMESTAMP,
    market_close_time_utc TIMESTAMP,
    end_date_utc TIMESTAMP,
    active BOOLEAN,
    closed BOOLEAN,
    archived BOOLEAN,
    accepting_orders BOOLEAN,
    enable_order_book BOOLEAN,
    raw_gamma_json JSON,
    first_seen_at_utc TIMESTAMP NOT NULL,
    last_seen_at_utc TIMESTAMP NOT NULL
);
```

```text
CREATE TABLE poly_outcomes (
    market_id TEXT NOT NULL,
    outcome_index INTEGER NOT NULL,
    outcome_name TEXT NOT NULL,
    clob_token_id TEXT NOT NULL,
    bucket_low_f INTEGER,
    bucket_high_f INTEGER,
    bucket_type TEXT,
    bucket_parse_status TEXT NOT NULL,
    raw_outcome_json JSON,
    PRIMARY KEY (market_id, outcome_index)
);
```

```text
CREATE TABLE poly_orderbook_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    clob_token_id TEXT NOT NULL,
    condition_id TEXT,
    retrieved_at_utc TIMESTAMP NOT NULL,
    provider_timestamp_utc TIMESTAMP,
    book_hash TEXT,
    last_trade_price DOUBLE PRECISION,
    best_bid DOUBLE PRECISION,
    best_ask DOUBLE PRECISION,
    midpoint DOUBLE PRECISION,
    spread DOUBLE PRECISION,
    raw_response_json JSON,
    source_request_id TEXT NOT NULL
);
```

```text
CREATE TABLE poly_orderbook_levels (
    snapshot_id TEXT NOT NULL,
    side TEXT NOT NULL,              -- bid or ask
    level_index INTEGER NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    size DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (snapshot_id, side, level_index)
);
```

```text
CREATE TABLE poly_price_history (
    market_id TEXT NOT NULL,
    clob_token_id TEXT NOT NULL,
    timestamp_utc TIMESTAMP NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    source_name TEXT NOT NULL,        -- prices-history, websocket, trade, orderbook-mid
    source_request_id TEXT,
    PRIMARY KEY (market_id, clob_token_id, timestamp_utc, source_name)
);
```

## 6. Market-data features

For each forecast cutoff and outcome bucket:

```text
best_bid
best_ask
midpoint
spread
last_trade_price
orderbook_depth_within_1c
orderbook_depth_within_2c
orderbook_depth_within_5c
weighted_average_fill_price_for_$10
weighted_average_fill_price_for_$50
weighted_average_fill_price_for_$100
price_change_last_5m
price_change_last_30m
price_change_last_2h
realized_volatility_last_2h
liquidity_score
market_implied_probability_raw
market_implied_probability_normalized_across_outcomes
```

Normalize across mutually exclusive outcome buckets:

```python
raw_mid_probs = [midpoint_i]
normalized_prob_i = midpoint_i / sum(midpoint_all_outcomes)
```

Keep both raw and normalized probabilities because CLOB spreads can make the sum differ from 1.

## 7. Trading simulation data requirements

To backtest trading decisions, Codex must reconstruct executable prices using only orderbook snapshots available at the decision time.

For a hypothetical buy of size `S` dollars/USDC:

```text
Use ask-side depth at or after forecast timestamp.
Compute volume-weighted average fill price.
Reject if insufficient depth within max_slippage.
```

For a sell:

```text
Use bid-side depth.
```

Do not use price history alone to simulate fills. Price history can be used for approximate analysis, but execution backtest must use orderbook depth where possible.

## 8. Edge features and no-trade logic

For each outcome bucket at decision time:

```text
model_prob = final calibrated probability
ask_price = best ask
bid_price = best bid
buy_edge = model_prob - executable_buy_price
sell_edge = executable_sell_price - model_prob
```

Trade only if:

```text
edge_after_costs > calibration_uncertainty_buffer + slippage_buffer + minimum_edge_threshold
```

Default thresholds are defined in the strategy spec.

## 9. Live acquisition and reliability

### 9.1 Snapshot daemon

Run continuously while markets are active:

```text
1. Discover active KLGA markets every 5 minutes.
2. Resolve clob_token_ids for all outcomes.
3. Snapshot all books every minute.
4. Increase to every 10 seconds near critical times if rate limits allow.
5. Persist raw bronze and parsed silver rows.
```

### 9.2 Data gaps

If orderbook snapshot fails:

```text
Write market_data_gap row.
Do not trade from stale orderbook older than MAX_ORDERBOOK_STALENESS_SECONDS.
Default MAX_ORDERBOOK_STALENESS_SECONDS = 15 for live execution, 120 for research.
```

## 10. Acceptance tests

```text
[ ] Gamma discovery can find active/recent KLGA temperature markets.
[ ] Outcome parser maps every outcome to a bucket or marks it failed.
[ ] CLOB orderbook snapshots are captured for every outcome token.
[ ] Best bid/ask/mid/spread/depth features are generated.
[ ] Historical trading simulator uses orderbook depth, not future settlement or future price history.
[ ] No trade is allowed when bucket parsing fails or orderbook data is stale.
[ ] Market data rows carry actual retrieved_at_utc timestamps and are never reconstructed as if observed earlier.
```
