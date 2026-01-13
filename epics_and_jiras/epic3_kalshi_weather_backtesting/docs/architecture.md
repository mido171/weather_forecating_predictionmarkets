# Architecture (Epic #3)

## High-level components
1) **Kalshi market data sync (Python)**
   - `weather_backtest.kalshi_client`:
     - REST client with retry, pagination, throttling
   - `weather_backtest.ingest`:
     - series/events/markets metadata sync
     - candlestick backfill (event candlesticks preferred)
     - optional trades backfill

2) **Backtest engine (Python)**
   - `weather_backtest.features`:
     - loads Epic #2 models (μ̂, σ̂) and produces bin probabilities as-of
   - `weather_backtest.strategy`:
     - scans markets for +EV trades vs current quotes
     - risk / position sizing rules
   - `weather_backtest.execution`:
     - applies fill model + fees
     - updates portfolio state
   - `weather_backtest.settlement`:
     - resolves trades using Kalshi market results (or CLI as verification)

3) **Persistence**
   - MySQL tables (defined in `models` module via Flyway + JPA entities):
     - Kalshi series/events/markets
     - candlesticks
     - backtest runs + backtest trades
   - Python reads/writes via SQLAlchemy (or JDBC if preferred, but SQLAlchemy recommended)

4) **Reporting**
   - `weather_backtest.reporting`:
     - metrics.json
     - report.md
     - plots

## Data flow (per run)
1) Ensure market catalog exists (sync if missing)
2) Ensure candlesticks exist for the time range (backfill if missing)
3) For each event:
   - compute asOfUtc
   - compute bin probabilities for that event’s markets
   - iterate time steps in trading window
   - simulate trades
4) At end:
   - compute P&L using settlement results
   - write artifacts to disk + summary rows to DB

## Scalability notes
- Candlestick data volume can be large. Support:
  - limiting trade window
  - partitioning tables (by month or by event)
  - optionally storing candlesticks in Parquet and only indexing DB metadata
