# WX-313 — Documentation pack & onboarding for Epic #3 (runbook + assumptions + limitations)

## Objective
Finalize documentation so Codex / a new developer can implement and run Epic #3 without tribal knowledge.

## Requirements

### 1) Ensure these docs exist and are accurate
- `EPIC-3.md`
- `agents.md`
- `docs/architecture.md`
- `docs/kalshi-api.md`
- `docs/time-semantics-backtest.md`
- `docs/fees-and-fills.md`
- `docs/backtest-metrics.md`
- `docs/runbook.md`
- `docs/references.md`

### 2) Explicitly document key limitations
Must include a clear statement:
- Historical backtests use candlestick best bid/ask (top-of-book) as proxy.
- Full historical depth orderbook is not available via official timestamped REST.
- If we want full depth, we must record WebSocket orderbook deltas going forward.

### 3) Add a “How to reproduce a run” section
Include:
- run_id
- config snapshot
- git sha
- seed
- DB watermark (latest candle ts used)

## Acceptance Criteria
- [ ] A new developer can run through the runbook end-to-end and produce a sample report.
- [ ] Docs clearly specify all time semantics and fee assumptions.
- [ ] Limitations are documented prominently to prevent overconfidence in backtest results.
