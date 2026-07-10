# Kalshi market service

This service is fail-closed. Its default configuration performs no authenticated Kalshi access, opens no upstream WebSocket, invokes no inference process, and cannot place an order.

## Safe default contract

- The server binds to `127.0.0.1`.
- The environment is `DEMO`.
- Authentication, trading, upstream WebSocket, live-trading services, inference invocation, smoke execution, backtest-grid execution, and startup reconciliation are disabled.
- Read and write rate limits default to one request per second, retry and reconnect budgets default to one, and backtests default to one thread without overwriting an existing SQLite file.
- Live controllers and WebSocket beans are not registered while live trading is disabled.
- Browser CORS is restricted to the local development origins on port `5173`.

## Explicit activation contract

Activation is intentionally multi-key so a single accidental flag cannot create live activity.

| Capability | Required settings |
| --- | --- |
| Authenticated API | `KALSHI_AUTH_ENABLED=true` and credentials supplied through environment-backed paths/values |
| Upstream WebSocket | Authenticated API plus `KALSHI_WEBSOCKET_ENABLED=true` |
| Local live dashboard | Authenticated API, upstream WebSocket, `KALSHI_LIVE_TRADING_ENABLED=true`, and a non-empty `KALSHI_LOCAL_CONTROL_TOKEN` |
| Inference process invocation | Local live dashboard plus `KALSHI_INFERENCE_INVOKE_ENABLED=true` |
| Trading | Authenticated API plus `KALSHI_TRADING_ENABLED=true`; guardrails must remain enabled |
| Authenticated production access | All applicable settings above plus `KALSHI_ENVIRONMENT=PROD` and `KALSHI_PRODUCTION_ACKNOWLEDGED=true` |

The account and inference routes require the `X-Local-Control-Token` request header. Missing or incorrect tokens return `401`; live routes are absent while live trading is disabled.

Production acknowledgement is a deliberate safety interlock, not a substitute for checking credentials, market tickers, position caps, and the selected environment. Never store API keys or private keys in tracked files.

## Exposure behavior

If startup reconciliation is explicitly enabled and provider state cannot be established, the exposure guard enters a global halt. Subsequent buy preflight checks fail closed without making additional portfolio calls. A process restart after the provider issue is resolved is required to clear that startup halt.

## Verification

From the repository root:

```powershell
mvn -pl apps/kalshi-market-service -am test
```

The safety tests cover inert defaults, invalid activation combinations, local token enforcement, absent live beans, and fail-closed startup reconciliation.
