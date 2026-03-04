# 00 - Scope and Objective

This document defines the exact scope of the MOS backtesting track, and the invariants that must never be broken.

## 1) Primary Objective

Produce reproducible, leakage-safe, execution-time-correct backtests for Kalshi temperature markets using MOS prediction artifacts.

Current supported strategy families:

1. Single-station runs (`KNYC`) for baseline/matrix analysis.
2. Co-joined multi-station runs (`KNYC` + `KMIA` + optional additional stations such as `KMDW`) with one trade/day globally.

The system is designed to be robust against:

- bucket parsing failures,
- YES/NO side inversion bugs,
- UTC/local time conversion mistakes,
- first-eligible-timestamp selection bugs,
- stake and PnL arithmetic mistakes,
- stale or ambiguous market-file mapping.

## 2) Strategy Modes In Scope

### 2.1 Fixed-Risk Mode

- per-trade stake:
  - `stake = min(balance_before * risk_fraction, stake_cap_usd)`

Used by:

- single-station baseline/matrix runs,
- co-joined baseline runs,
- co-joined open+30m fixed-risk base runs.

### 2.2 Fractional-Kelly Mode (Post-Processed Variant)

- per-trade risk fraction:
  - `full_kelly = (q - p) / (1 - p)` (clamped)
  - `risk_fraction_used = kelly_fraction * full_kelly`
- per-trade stake:
  - `stake = min(balance_before * risk_fraction_used, stake_cap_usd)`

Used by:

- strict co-joined reporting variants with outlier-filtered recalc.

## 3) Non-Negotiable Invariants

1. Date/file alignment:
   - target date `T` must map only to day file `*_YYYYMMDD.csv` with `YYYYMMDD == T`.
2. Time gating:
   - entries must be at or after configured gate,
   - if open-delay is enabled, entries must be at or after effective cutoff:
     - `effective_cutoff = max(gate_cutoff, market_open + delay)`.
3. Chronological selection:
   - choose first eligible timestamp globally (co-joined) or station-local (single station).
4. Side-aware probability mapping:
   - YES side uses normalized price directly,
   - NO side uses complement.
5. Model probability computation:
   - must be recomputed from quantiles -> integer PMF -> bucket sum.
6. EV consistency:
   - `EV = p_model_side - p_market_side`.
7. Settlement and bankroll:
   - hold-to-expiry binary payoff,
   - stake and PnL must reconcile exactly to formulas.
8. Co-joined daily cap:
   - max one trade/day globally across stations.

## 4) Explicit Non-Goals

These runs do not model:

- order book depth/slippage,
- partial fills,
- exchange fees,
- intraday exits,
- execution latency simulation.

This is an end-of-market hold-to-expiry simulation framework.

## 5) Current Canonical References

Historical baseline:

- `documentation/mos/04_run_record_2026-03-01_entry1530z_cap400.md`

Runtime matrix:

- `documentation/mos/06_run_record_2026-03-01_leakage_free_runtime_matrix.md`

Co-joined baseline:

- `documentation/mos/07_run_record_2026-03-01_knyc_kmia_cojoined_blend12.md`

Strict historical (no open-delay):

- `documentation/mos/08_run_record_2026-03-01_knyc_kmia_cojoined_blend12_fractionalkelly_no_outlier_gt2000.md`

Current strict reference (open+30m + fractional-Kelly + outlier-filtered recalc):

- `documentation/mos/09_run_record_2026-03-02_knyc_kmia_cojoined_blend12_openplus30m_fractionalkelly_no_outlier_gt2000.md`

Extended fixed-risk 3-station co-joined reference (`KNYC` + `KMIA` + `KMDW`):

- `documentation/mos/16_run_record_2026-03-04_cojoined_blend12_knyc_kmia_kmdw_openplus30m_risk7p5_cap700_2024_2025.md`
