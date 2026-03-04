# 02 - Backtest Logic and Formulas

This document is the canonical execution + arithmetic spec for MOS backtests.

It covers:

- single-station execution,
- co-joined multi-station execution,
- effective-cutoff (gate + open-delay) semantics,
- sizing formulas,
- outlier-filtered post-processing.

## 1) Per-Day Execution Logic

### 1.1 Single-Station Flow

For each `target_date_local = T`:

1. Load day file `KNYC_YYYYMMDD.csv` where `YYYYMMDD == T`.
2. Parse/sort `timestamp` as UTC.
3. Compute gate:
   - `gate_cutoff_utc = (T - 1 day) + HH:MM Z`
4. If open-delay is configured:
   - `effective_cutoff_utc = max(gate_cutoff_utc, market_open_utc + delay_minutes)`
5. Select first quote row with `timestamp >= effective_cutoff_utc`.
6. Evaluate all bucket/side candidates on that selected row.
7. Apply filters (`EV`, `model_win_prob`), choose best candidate by strategy policy.
8. Enter max one trade for day.

### 1.2 Co-Joined Flow (N Stations)

For each `target_date_local = T`:

1. Load each configured station day file if present:
   - `<FILE_PREFIX>_YYYYMMDD.csv` per station.
2. Parse/sort timestamps in UTC.
3. Compute `gate_cutoff_utc`.
4. Per station compute `effective_cutoff_utc`:
   - `effective_cutoff_utc = max(gate_cutoff_utc, market_open_utc + delay_minutes)`
5. Keep station rows where `timestamp >= effective_cutoff_utc`.
6. Walk the union of all station timestamps in ascending order.
7. At each timestamp evaluate all station/bucket/side candidates.
8. First timestamp with any eligible candidate is entry timestamp for that day.
9. If multiple candidates exist at that timestamp, apply deterministic tie-break:
   - highest `model_win_prob`
   - then highest `EV`
   - then lowest `market_price`
   - then station alphabetical
10. Enforce one-trade/day globally across stations.

## 2) Quantiles -> Integer PMF

Input quantiles:

- `q_0.05`, `q_0.10`, `q_0.25`, `q_0.50`, `q_0.75`, `q_0.90`, `q_0.95`

Conversion:

1. enforce non-decreasing quantiles,
2. build piecewise-linear CDF by interpolation,
3. integer mass:
   - `p(t) = F(t + 0.5) - F(t - 0.5)`
4. clamp negative numerical residue to zero,
5. renormalize PMF to sum to 1.

Script support bounds:

- `t in [-20, 130]` degF.

## 3) Bucket Probability from PMF

- range bucket `[lo, hi]`: `sum_{t=lo..hi} p(t)`
- `X or below`: `sum_{t<=X} p(t)`
- `X or above`: `sum_{t>=X} p(t)`

## 4) Side Probabilities and EV

Let:

- `p_model_yes = P_model(bucket)`
- `p_model_no = 1 - p_model_yes`
- `p_mkt_yes = normalized market YES price`
- `p_mkt_no = 1 - p_mkt_yes`

Then:

- `EV_yes = p_model_yes - p_mkt_yes`
- `EV_no = p_model_no - p_mkt_no`

Eligibility is side-specific:

- `model_win_prob_side >= win_min`
- `EV_side >= ev_min`

## 5) Sizing Formulas

### 5.1 Fixed-Risk Mode

- `stake = min(balance_before * risk_fraction, stake_cap_usd)`

Used by:

- baseline single-station runs,
- baseline co-joined runs,
- co-joined open+30m fixed-risk base run.

### 5.2 Fractional-Kelly Mode (Post-Processed Variant)

Let:

- `p = market_price`
- `q = model_win_prob`

Then:

- `full_kelly = (q - p) / (1 - p)` for `0 < p < 1`, else `0`
- clamp `full_kelly` to `[0, 1]`
- `risk_fraction_used = kelly_fraction * full_kelly`
- `stake = min(balance_before * risk_fraction_used, stake_cap_usd)`

Example strict settings:

- `kelly_fraction = 0.15`
- `stake_cap_usd = 500`

Shares:

- `shares = stake / p` if `p > 0`, else `0`

## 6) Settlement and Bankroll

Hold-to-expiry binary payoff:

- win: `pnl = shares * (1 - price)`
- loss: `pnl = -stake`

Balance:

- `balance_after = balance_before + pnl`

Drawdown:

- `peak_balance = max(previous_peak, balance_after)`
- `drawdown = (peak_balance - balance_after) / peak_balance`

## 7) Summary Metrics

- `trades`, `wins`, `losses`, `win_rate`
- `profit_factor = gross_profit / gross_loss_abs`
- `final_balance`, `total_pnl`
- `avg_ev_at_trade`, `median_ev_at_trade`
- `max_drawdown`
- co-joined only: `station_counts` and `side_counts`

## 8) Effective-Cutoff Clarification

Entry timing is always quote-driven:

- entry is never forced at exact gate timestamp,
- entry is first available eligible quote at/after effective cutoff.

For open-delay runs:

- `delay_minutes = 30` (open+30m variants)
- each station has its own `market_open_utc`
- therefore effective cutoff can differ across stations on same day.

## 9) Outlier-Filtered Recalculation Protocol

When enabled:

- remove trades where `pnl > threshold` (current strict threshold: `2000`)

Then recompute sequence on remaining trades:

1. sort remaining trades chronologically,
2. reset bankroll to original `start_balance`,
3. re-apply sizing formulas per row,
4. recompute `stake`, `shares`, `pnl`, `balance`, `drawdown`,
5. write filtered artifacts (`trades`, `summary`, `sanity`, removed-trade log).

This must always be labeled as post-processed output (not raw stream output).

## 10) Current Strict Reference Rule Pack

Latest strict reference (see run record `09`):

- gate: `T-1 12:00Z`
- open delay: `+30 minutes after station market open`
- filters: `EV >= 0.18`, `model_win_prob >= 0.67`
- base stream sizing: fixed risk `6.5%`, cap `$500`
- strict reporting transform: fractional Kelly `0.15`, cap `$500`, remove `pnl > 2000`, recalc bankroll

Extended fixed-risk multi-station reference (see run record `16`):

- gate: `T-1 12:00Z`
- open delay: `+30 minutes after station market open`
- filters: `EV >= 0.25`, `model_win_prob >= 0.85`, `min_market_price >= 0.25`
- sizing: fixed risk `7.5%`, cap `$700`
- stations: `KNYC`, `KMIA`, `KMDW`
