# References (primary)

## Kalshi API docs (official)
- API home: https://docs.kalshi.com/
- Pagination: https://docs.kalshi.com/getting_started/pagination
- Rate limits: https://docs.kalshi.com/getting_started/rate_limits
- Market list (Get Markets): https://docs.kalshi.com/api-reference/market/get-markets
- Market candlesticks: https://docs.kalshi.com/api-reference/market/get-market-candlesticks
- Event candlesticks: https://docs.kalshi.com/api-reference/events/get-event-candlesticks
- Batch market candlesticks: https://docs.kalshi.com/api-reference/market/batch-get-market-candlesticks
- Trades: https://docs.kalshi.com/api-reference/market/get-trades
- Orderbook: https://docs.kalshi.com/api-reference/market/get-market-orderbook
- WebSockets quick start (for forward orderbook capture): https://docs.kalshi.com/getting_started/quick_start_websockets
- Authenticated request signing: https://docs.kalshi.com/getting_started/quick_start_authenticated_requests
- API keys: https://docs.kalshi.com/getting_started/api_keys
- Series fee changes: https://docs.kalshi.com/api-reference/exchange/get-series-fee-changes
- Get Series (series metadata): https://docs.kalshi.com/api-reference/market/get-series

## Kalshi fees
- Fee schedule page: https://kalshi.com/fee-schedule
- Help Center “Fees”: https://help.kalshi.com/trading/fees
- Fee schedule PDF (historical): https://www.cftc.gov/sites/default/files/filings/orgrules/22/09/rule091222kexdcm003.pdf

## Notes
- For *historical* backtesting, the most reliable official historical “quote” source is the candlestick endpoints.
- Full historical depth orderbook snapshots are not provided via an official timestamped REST endpoint; to get full depth you must record WebSocket deltas going forward.
