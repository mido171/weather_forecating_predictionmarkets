# HKG-T24-001 Source Inventory Report

## Status

PASS

## Checks

- `public.hko_daily_tmax_target_labels`: WARNING; Primary table absent; selected ordered fallback `label_core.hko_daily_tmax`.
- `hko_target_labels`: WARNING; Additional fallback candidates also exist but are lower priority: feature_safe.hko_target_history_pre2024
- `label_core.hko_daily_tmax`: PASS; Fallback table selected.
- `public.hko_historical_forecasts_2000_2026`: PASS; Primary table exists.
- `public.hko_historical_forecasts_2000_2026`: PASS; Required columns present.
- `public.hko_historical_forecasts_2000_2026`: PASS; Usable official local min/max rows = 115795.
- `nwp_tactical.forecast_wide`: PASS; Required columns present.
- `nwp_tactical.raw_response_object`: PASS; Required columns present.
- `nwp_tactical.full_tactical_backfill_ok_tmax`: PASS; Full tactical scoped rows = 1964157.

## Warnings

- WARNING: ARWF source table absent. E11_ARWF_LIVE_SHADOW will emit placeholder rows with SOURCE_TABLE_ABSENT.

## Failures

- None.
