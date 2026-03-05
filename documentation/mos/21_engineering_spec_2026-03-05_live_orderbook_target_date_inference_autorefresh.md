# 21) Engineering Spec — Live Orderbook Target-Date Switching + Inference Auto-Refresh (2026-03-05)

## 1. Purpose

This spec documents the implementation that turns the Live Trading page into a date-aware execution surface with active inference refresh control.

The implementation target is:

1. The operator can switch opportunities between `T` and `T+1` directly from UI.
2. After local cutoff (`17:45`), default opportunities view should prefer `T+1`.
3. If selected date has no opportunities yet, operator can click one button to auto-invoke live inference every 10 seconds for that selected date.
4. Once selected date becomes populated, auto-invoke stops automatically and EV continues updating from live orderbook frames only.

This behavior is implemented without changing leakage policy semantics in the Python inference script; it only orchestrates invocation and display.

## 2. Problem Statement

Observed operator pain points before this change:

1. Date switching did not always guarantee immediate fetch for the selected date.
2. Tomorrow date (`T+1`) could remain empty in UI even when valid inference artifacts existed.
3. There was no operator control to trigger inference generation from the live screen itself.
4. Manual terminal invocation workflow created friction and delayed opportunity visibility.

## 3. High-Level Architecture

### 3.1 Backend responsibilities

1. Serve date-aware snapshot data:
   - `GET /api/live-trading/orderbooks/snapshot?targetDateLocal=YYYY-MM-DD`
2. Resolve station events/markets and maintain orderbook stream for both primary and secondary target dates.
3. Compute opportunities (YES/NO side) from:
   - latest station quantiles,
   - bucket probability interpolation,
   - current live orderbook prices.
4. Expose inference trigger endpoint:
   - `POST /api/live-trading/inference/run?targetDateLocal=YYYY-MM-DD`
5. Guard inference invocation against overlap (`busy` response).

### 3.2 Frontend responsibilities

1. Show date toggles in opportunities panel.
2. Fetch selected date opportunities immediately.
3. Maintain separate opportunity caches per date (`rowsByDate`).
4. Offer `Auto Refresh Inference (10s)` button.
5. Stop auto mode as soon as selected date has non-empty opportunities.
6. Continue normal live EV updates from orderbook stream/snapshot polling once populated.

## 4. Backend Detailed Design

## 4.1 Date-aware snapshots and opportunities

Core service:

- `kalshi-api-service/src/main/java/com/predictionmarkets/weather/kalshiapi/live/LiveOrderbookStreamService.java`

Key behavior:

1. Maintains per-date station runtime state:
   - `Map<LocalDate, Map<String, StationRuntimeState>>`
2. Supports:
   - `currentSnapshot()` for primary date
   - `currentSnapshot(LocalDate targetDateLocal)` for explicit date
3. For each configured station:
   - resolves markets for both primary date and alternate date,
   - tracks markets so orderbook service has data available for both dates.
4. Applies date-specific inference selection:
   - station inference is selected only if `inference.targetDateLocal == requested targetDateLocal`

Outcome:

1. `snapshot?targetDateLocal=2026-03-05` and `snapshot?targetDateLocal=2026-03-06` return independent date-scoped station/opportunity views.
2. Opportunity emptiness is now truthful per target date rather than mixed/fallback behavior.

## 4.2 Local cutoff policy

Config:

- `kalshi.live-trading.next-target-date-cutoff-local-time: "17:45"`

Behavior:

1. For each station zone:
   - before `17:45`, primary target date is local today (`T`)
   - at/after `17:45`, primary target date is local tomorrow (`T+1`)
2. Secondary date is the opposite day, so both remain queryable.

This directly supports operator expectation that evening workflow should bias toward tomorrow.

## 4.3 Inference invocation endpoint

New endpoint:

- `POST /api/live-trading/inference/run?targetDateLocal=YYYY-MM-DD`

Controller:

- `LiveInferenceController`

Service:

- `LiveInferenceInvokeService`

Response model:

- `LiveInferenceRunResponse`
  - `targetDateLocal`
  - `status` (`success`, `busy`, `timeout`, `failed`, `disabled`, `error`, `invalid`)
  - `exitCode` (if available)
  - `reportPath` (parsed from script stdout where available)
  - `message` (trimmed script output or error text)

Invocation details:

1. Command pattern:
   - `<python> <script> --target-date <date> --stdout-json summary --log-level ERROR`
2. Defaults:
   - python executable: `python`
   - script path: `tools/live/mos_quantile_live_inference.py`
3. Working directory resolves to repo root (based on script location).
4. Concurrency guard:
   - single-flight via `AtomicBoolean running`
   - concurrent requests return `busy`
5. Timeout:
   - configurable, default `180s`

## 4.4 Backend configuration knobs

Added in `LiveTradingProperties` and `application.yml`:

1. `kalshi.live-trading.inference-invoke-enabled` (default `true`)
2. `kalshi.live-trading.inference-invoke-python-executable` (default `python`)
3. `kalshi.live-trading.inference-invoke-script-path` (default `tools/live/mos_quantile_live_inference.py`)
4. `kalshi.live-trading.inference-invoke-timeout-seconds` (default `180`)
5. `kalshi.live-trading.next-target-date-cutoff-local-time` (default `17:45`)

These are operator-facing safety and portability controls.

## 5. Frontend Detailed Design

Primary file:

- `ui/result_viewer/src/LiveTradingPage.jsx`

Styles:

- `ui/result_viewer/src/liveTrading.css`
- supporting page-shell styles in `ui/result_viewer/src/styles.css`

## 5.1 Date-specific opportunities cache

State model:

1. `selectedOpportunitiesDate`
2. `opportunityRowsByDate` (`{ [YYYY-MM-DD]: OpportunityRow[] }`)
3. `opportunityRowsLoading`

Fetch helpers:

1. `snapshotUrlForTargetDate(baseUrl, targetDateLocal)`
2. `fetchOpportunitiesForDate(date)`:
   - calls date-scoped snapshot endpoint,
   - transforms stations -> rows,
   - sorts by EV.

Key UX guarantee:

1. On date click, selected date fetch starts immediately.
2. UI does not silently reuse today rows for tomorrow selection.

## 5.2 Date toggle and display behavior

Opportunities panel now includes:

1. Date chips for available options (typically `T` and `T+1`)
2. Mode toggle:
   - threshold mode (`win >= 85%`)
   - all-by-EV mode
3. Count visibility:
   - `Showing X / Y`

After local `17:45`, default selected date is `T+1` unless overridden by explicit user selection.

## 5.3 Auto inference refresh control

New control in opportunities actions:

1. Button label toggles:
   - `Auto Refresh Inference (10s)`
   - `Stop Auto Refresh`
2. Status line underneath header explains current state:
   - invoking
   - busy
   - refreshed/waiting
   - populated + stopped
   - error

Loop behavior:

1. Interval: 10 seconds
2. On each tick:
   - POST inference-run for selected date
   - fetch selected-date snapshot
   - update `opportunityRowsByDate[selectedDate]`
3. Stop condition:
   - if selected date opportunities length > 0, auto mode turns off
4. Protection:
   - single in-flight tick via ref guard, no overlapping invocations

This matches requested behavior exactly:

1. Keep invoking until populated.
2. Stop invoking once populated.
3. Continue normal EV updates from live orderbook stream thereafter.

## 6. Operator Workflow

Example on March 5, 2026:

1. Open live trading page.
2. Date toggle shows `Mar 5` and `Mar 6`.
3. Click `Mar 6`.
4. If opportunities empty, click `Auto Refresh Inference (10s)`.
5. UI invokes inference for target date `2026-03-06` every 10 seconds.
6. Once snapshot for `2026-03-06` returns opportunities:
   - list appears,
   - button auto loop stops,
   - status confirms populated.
7. Ongoing EV movement continues from orderbook updates.

## 7. API Contracts and Examples

### 7.1 Snapshot by date

Request:

`GET /api/live-trading/orderbooks/snapshot?targetDateLocal=2026-03-06`

Expected:

1. `stations[].targetDateLocal == "2026-03-06"`
2. opportunities list calculated from that date’s inference + live orderbook prices

### 7.2 Run inference by date

Request:

`POST /api/live-trading/inference/run?targetDateLocal=2026-03-06`

Success response example fields:

1. `status: "success"`
2. `exitCode: 0`
3. `reportPath: "...\\inference_report.json"`

Busy response:

1. `status: "busy"`
2. message indicates another invocation is running

## 8. Verification Completed

Validation performed in implementation session:

1. Backend compile:
   - `mvn -pl kalshi-api-service -DskipTests compile` passed.
2. Frontend build:
   - `npm run build` in `ui/result_viewer` passed.
3. Endpoint verification:
   - `POST /api/live-trading/inference/run?targetDateLocal=2026-03-06` returned `success`.
4. Snapshot verification:
   - `GET snapshot?targetDateLocal=2026-03-06` returned populated stations and opportunities.

## 9. Files Added/Updated

Backend:

1. `kalshi-api-service/src/main/java/com/predictionmarkets/weather/kalshiapi/config/LiveTradingProperties.java`
2. `kalshi-api-service/src/main/java/com/predictionmarkets/weather/kalshiapi/live/LiveInferenceController.java`
3. `kalshi-api-service/src/main/java/com/predictionmarkets/weather/kalshiapi/live/LiveInferenceInvokeService.java`
4. `kalshi-api-service/src/main/java/com/predictionmarkets/weather/kalshiapi/live/LiveInferenceRunResponse.java`
5. `kalshi-api-service/src/main/java/com/predictionmarkets/weather/kalshiapi/live/LiveOrderbookController.java`
6. `kalshi-api-service/src/main/java/com/predictionmarkets/weather/kalshiapi/live/LiveOrderbookStreamService.java`
7. `kalshi-api-service/src/main/resources/application.yml`

Frontend:

1. `ui/result_viewer/src/LiveTradingPage.jsx`
2. `ui/result_viewer/src/liveTrading.css`
3. `ui/result_viewer/src/styles.css`
4. `ui/result_viewer/src/App.jsx`
5. `ui/result_viewer/src/liveOrderbookUtils.js`

Supporting live/inference plumbing updates (already in feature branch scope):

1. `kalshi-api-service/src/main/java/com/predictionmarkets/weather/kalshiapi/api/KalshiMarketDataApi.java`
2. `kalshi-api-service/src/main/java/com/predictionmarkets/weather/kalshiapi/model/market/Event.java`
3. `kalshi-api-service/src/main/java/com/predictionmarkets/weather/kalshiapi/model/market/EventResponse.java`
4. `kalshi-api-service/src/main/java/com/predictionmarkets/weather/kalshiapi/service/DefaultKalshiOrderBookService.java`
5. `kalshi-api-service/src/main/java/com/predictionmarkets/weather/kalshiapi/ws/model/WsSubscription.java`
6. `kalshi-api-service/src/test/java/com/predictionmarkets/weather/kalshiapi/service/DefaultKalshiOrderBookServiceTest.java`
7. `tools/live/mos_quantile_live_inference.py`
8. `tools/live/run_kmia_live.py`

## 10. Operational Caveats

1. Auto-refresh inference depends on Python runtime availability from backend process context (`python` on PATH unless overridden).
2. If inference invocation is disabled (`inference-invoke-enabled=false`), UI button reports disabled and auto mode halts.
3. `busy` status can appear if one run exceeds 10s interval; this is expected and safe.
4. If report generation succeeds but opportunities still empty, likely causes are:
   - no qualifying opportunities at current market prices,
   - date mismatch between selected date and latest inference target date,
   - transient snapshot/orderbook lag.

## 11. Troubleshooting Playbook

If tomorrow (`T+1`) still shows empty unexpectedly:

1. Check inference endpoint directly:
   - `POST /api/live-trading/inference/run?targetDateLocal=<T+1>`
2. Confirm response:
   - `status=success`, `exitCode=0`, `reportPath` present
3. Query snapshot:
   - `GET /api/live-trading/orderbooks/snapshot?targetDateLocal=<T+1>`
4. Inspect fields:
   - `stations[].targetDateLocal`
   - `stations[].predictionPointTmaxF`
   - `opportunities[]` count
5. If inference works but opportunities remain zero:
   - this can be valid market state (no rows pass filters at current ask prices)
   - use "Show All by EV" mode to inspect full sorted opportunity set.

## 12. Result Summary

The live trading screen now supports a complete operator loop for date-targeted execution preparation:

1. Select date quickly.
2. Trigger and automate inference refresh from UI.
3. Auto-stop when usable opportunity surface is present.
4. Continue real-time EV movement from live orderbook updates without repeated script invocations.

This closes the gap between offline script invocation and live execution monitoring while keeping date semantics explicit and operator-controlled.
