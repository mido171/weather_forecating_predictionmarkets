# 06 Polymarket Market Data

Source spec:

```text
08_polymarket_market_data.md
```

Execution role:

This task captures market metadata, order books, prices, spreads, and activity needed to link weather probabilities to tradable contracts.

Persistence target:

```text
postgresql://postgres:root@127.0.0.1:5432/klga_tmax_research
```

Start condition:

Use Gamma API discovery and CLOB public market-data endpoints as specified. Keep market metadata, books, trades, and historical prices in separate normalized surfaces.
