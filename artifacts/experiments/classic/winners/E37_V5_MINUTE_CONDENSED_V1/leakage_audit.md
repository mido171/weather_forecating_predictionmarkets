# Leakage Audit

Rows checked: 8745

## Cutoff checks
- max_ts_used <= decision_utc: pass 8740 / fail 0 / missing 5

## T-1 local-day checks
- max_ts_utc_t1 local date == T-1: pass 8740 / fail 0 / missing 5

## Early window checks
- max_ts_utc_early UTC date == T: pass 8736 / fail 0 / missing 9
- max_ts_utc_early time <= 06:00Z: pass 8736 / fail 0 / missing 9

## Translator consistency
- diff_lag1 == y(T-1) - iem_tmax_t1: pass 8472 / fail 0 / missing 273
- diff_ewma_30 max abs error vs recompute: 0

Max(max_ts_used - decision_utc): 0 days 00:00:00