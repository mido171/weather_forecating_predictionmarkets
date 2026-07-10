# HKG-T24-001 Schema Contract Report

## Status

PASS

## Discovered Target Labels

`label_core.hko_daily_tmax` with date column `local_date` and value column `target_tmax_c`.

## Discovered Official Forecasts

`public.hko_historical_forecasts_2000_2026`.

## Detailed Checks

- `public.hko_daily_tmax_target_labels`: WARNING; rows=None; Primary table absent; selected ordered fallback `label_core.hko_daily_tmax`.
- `hko_target_labels`: WARNING; rows=None; Additional fallback candidates also exist but are lower priority: feature_safe.hko_target_history_pre2024
- `label_core.hko_daily_tmax`: PASS; rows=48577; Fallback table selected.
- `public.hko_historical_forecasts_2000_2026`: PASS; rows=324179; Primary table exists.
- `public.hko_historical_forecasts_2000_2026`: PASS; rows=324179; Required columns present.
- `public.hko_historical_forecasts_2000_2026`: PASS; rows=115795; Usable official local min/max rows = 115795.
- `nwp_tactical.forecast_wide`: PASS; rows=1965090; Required columns present.
- `nwp_tactical.raw_response_object`: PASS; rows=1285; Required columns present.
- `nwp_tactical.full_tactical_backfill_ok_tmax`: PASS; rows=1964157; Full tactical scoped rows = 1964157.
