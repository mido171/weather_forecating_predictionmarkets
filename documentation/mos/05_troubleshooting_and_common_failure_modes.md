# 05 - Troubleshooting and Common Failure Modes

This document is the MOS backtesting debug playbook.

## 1) Bucket Parsing Corruption

Symptoms:

- malformed labels in outputs,
- impossible parsed ranges (often from hyphen/minus confusion).

Cause:

- parser treated range hyphen as sign.

Fix:

- parse labels by extracting unsigned integers only,
- normalize to canonical forms (`XF to YF`, `XF or below`, `XF or above`).

Validation:

- sanity `bucket_unparseable == 0`
- sanity `bucket_not_found == 0`

## 2) YES/NO Side Inversion Bugs

Symptoms:

- NO-side market probability equals YES column unexpectedly,
- EV sign looks inverted.

Rule:

- YES side price = normalized bucket column value
- NO side price = `1 - YES`

Validation:

- sanity `market_price_mismatch == 0`
- sanity `ev_mismatch == 0`

## 3) Entry-Time Confusion (Gate vs Effective Cutoff)

Symptoms:

- entries occur later than expected gate timestamp,
- user expects exact gate-time fill.

Clarification:

- gate is start of eligibility scan,
- actual entry is first eligible quote at/after effective cutoff,
- if open-delay enabled:
  - `effective_cutoff = max(gate, market_open + delay)`.

Validation:

- sanity `entry_before_gate == 0`
- sanity `entry_before_effective_cutoff == 0` (for open-delay runs)

## 4) Target-Date/File-Date Mismatch

Symptoms:

- wrong day file used for a target date.

Rule:

- target `T` must use `*_YYYYMMDD.csv` where date is exactly `T`.
- `T-1` applies only to gate time.

Validation:

- re-check mapping contract in `01_data_contracts_and_file_mapping.md`,
- verify day-level debug `market_file` and `target_date_local`.

## 5) Duplicate Day Files Across Folders

Symptoms:

- results change when root traversal order changes.

Cause:

- same day file exists in multiple folders.

Fix:

- use deterministic folder priority or dedicated run roots.

Preferred for co-joined:

- use dedicated roots:
  - `kxhighny_2024_10_01_to_2025_12_31`
  - `kxhighmia_2024_10_01_to_2025_12_31`

## 6) One-Trade/Day Violations (Co-Joined)

Symptoms:

- more than one row for same `target_date_local`.

Cause:

- arbitration logic broken or multiple entries not prevented.

Validation:

- sanity `more_than_one_trade_per_day_global == 0`
- verify tie-break check:
  - `tie_break_policy_violation == 0`

## 7) Zero-Price Trades and "Win with PnL=0"

Symptoms:

- trade row shows `market_price = 0`,
- result may be `Win`, but PnL is `0`.

Why this happens:

- selected side can be a complement side priced at exactly 0,
- shares are computed as `stake / price`; when `price <= 0`, shares are set to `0`,
- win formula gives `pnl = shares * (1 - price) = 0`.

This is mathematically consistent with current code path, not a parsing error.

Optional policy hardening:

- skip trades where side price `<= 0`,
- or enforce a minimum executable price floor.

## 8) Extreme PnL Dominance

Symptoms:

- one/few trades dominate equity curve.

Cause:

- very low entry prices can produce very high share counts.

Current mitigation pattern used in strict variants:

- outlier-filtered recalc (`remove pnl > threshold`, then recompute bankroll path).

## 9) Empty/Very Sparse Trade Set

Symptoms:

- near-zero or zero trades.

Typical causes:

- thresholds too strict (`EV`, `win_min`),
- open-delay + gate reduce eligible rows too aggressively,
- date range has sparse market files.

Checks:

- summary `days_without_trade_candidate`
- summary market-file coverage counts
- day-debug station status (`has_rows_after_effective_cutoff`)

## 10) Quick Debug Checklist

When a run looks wrong, inspect in this order:

1. summary JSON (counts + headline metrics),
2. sanity JSON (failure counters),
3. day-debug JSON (first eligible timestamps and effective cutoffs),
4. side-aware table (manual row inspection).

For current strict reference, start with:

- `summary_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p18_win67_fractionalkelly0p15_cap500_no_outlier_gt2000_recalc.json`
- `sanity_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p18_win67_fractionalkelly0p15_cap500_no_outlier_gt2000_recalc.json`
