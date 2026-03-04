# 17 - Run Record (2026-03-04, UI: Full Trade Table Virtualization + Station Contribution Scoring)

## 1) Objective

Implement and document two UI upgrades for the MOS result viewer:

1. Fix sluggish scrolling in `Full Trade Table`.
2. Add clear per-station contribution analytics (profit, PF, win, DD, and a composite score) above Monthly Breakdown.

Files updated:

1. `ui/result_viewer/src/App.jsx`
2. `ui/result_viewer/src/styles.css`

## 2) Dataset Wiring Update

`2024-2025` tab was updated to point at the new 3-station co-joined side-aware table:

- `/data/all_trades_sideaware_cojoined_blend12_knyc_kmia_kmdw_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_risk7p5_cap700_2024_2025_with_balance.csv`

Parameter chips were aligned to actual run rules:

1. stations include `KNYC + KMIA + KMDW`
2. period `2024-10-01 -> 2025-12-31`
3. `EV >= 0.25`
4. `Win >= 0.85`
5. `Side price >= 25c`
6. `Risk fraction 7.5%`
7. `Stake cap $700`
8. `Entry >= max(T-1 12:00Z, open+30m)`

## 3) Full Trade Table Performance Fix

### 3.1 Root cause

The table rendered all rows in the DOM (`filteredRows.map(...)`) with all columns. This causes noticeable scroll jank as row count grows.

### 3.2 Implementation

A simple row virtualization layer was added:

1. constants:
   - `TABLE_ROW_HEIGHT = 42`
   - `TABLE_OVERSCAN_ROWS = 14`
2. state:
   - `tableScrollTop`
   - `tableViewportHeight`
3. refs:
   - `tableWrapRef`
4. computed slice:
   - `virtualizedTable.rows = filteredRows.slice(start, end)`
   - `topPad` and `bottomPad` spacer rows preserve scroll height
5. sticky header behavior remains unchanged

### 3.3 Render contract

Instead of mapping all rows:

- render only `virtualizedTable.rows`
- insert two spacer rows:
  1. top spacer with `height = topPad`
  2. bottom spacer with `height = bottomPad`

Additional UI hint:

- `Showing X-Y of N rows` line above the table.

## 4) New Station Contribution Panel

Inserted above:

- `Monthly Breakdown (click to filter)`

Panel title:

- `Station Contribution Score`

Scope note:

- `All trades in selected dataset` or month-specific scope if month filter is active.

## 5) Scoring Model (0-100)

Per-station score rewards:

1. win rate
2. profit factor (capped normalization)
3. number of trades
4. absolute dollar pnl

Per-station score penalizes:

1. loss rate
2. drawdown contribution relative to total portfolio peak balance

Displayed formula:

- `score = 100 x (0.32 win + 0.24 pf + 0.16 volume + 0.20 pnl - 0.05 losses - 0.13 dd_vs_portfolio)`

Normalization details:

1. `pf_norm = clamp(min(PF, 3) / 3, 0, 1)` (`INF` treated as `3`)
2. `trade_norm = clamp(sqrt(trades / max_trades), 0, 1)`
3. `pnl_norm = clamp(max(0, pnl) / max_positive_pnl, 0, 1)`
4. `dd_vs_portfolio = station_max_drawdown_abs_usd / portfolio_peak_balance`

## 6) Drawdown Semantics Correction

The first implementation used station-isolated drawdown percent, which can appear as `100%` and confuse interpretation.

This was corrected to:

1. compute station drawdown in dollars from station cumulative PnL stream (`maxDrawdownAbsUsd`)
2. divide by total portfolio peak balance in current scope (`maxDrawdownPortfolioPct`)
3. use this value for both:
   - score penalty
   - displayed `Max DD vs Portfolio`

UI now displays:

- `Max DD vs Portfolio: -X% ($Y)`

## 7) Displayed Per-Station Metrics

Each card includes:

1. rank and score badge
2. station id
3. balance contribution (`$ pnl`)
4. portfolio pnl share (`%`)
5. win rate
6. profit factor
7. trades
8. W-L
9. max DD vs portfolio (percent and dollars)

## 8) Current `2024-2025` Snapshot (All Months, 3-Station Dataset)

Computed with current code on active dataset:

1. `KMDW`: score `76.86`, pnl `+$25,517.92`, PF `2.19`, DD vs portfolio `5.17%` (`$3,256.21`)
2. `KMIA`: score `72.56`, pnl `+$19,897.84`, PF `2.32`, DD vs portfolio `5.18%` (`$3,263.16`)
3. `KNYC`: score `59.18`, pnl `+$13,404.20`, PF `1.64`, DD vs portfolio `4.51%` (`$2,838.56`)

## 9) Validation and Limits

Validation performed:

1. code-level checks and deterministic recomputation of scoring terms
2. dataset path and field names verified against the loaded CSV

Environment limit:

- frontend build/runtime command could not be executed in this shell due to missing `node/npm` executable in the environment.

## 10) Why This Improves Decision Quality

1. table virtualization improves operational usability for large trade logs.
2. per-station contribution panel makes cross-station impact explicit without needing ad-hoc spreadsheet analysis.
3. drawdown normalization vs portfolio resolves ambiguity and aligns risk interpretation with portfolio-level decision making.
