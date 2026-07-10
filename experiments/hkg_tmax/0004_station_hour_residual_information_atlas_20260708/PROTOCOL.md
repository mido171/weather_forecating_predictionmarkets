# Protocol

This is a diagnostic information atlas, not a promoted model.

- Target frame: pre-2024 `label_core.hko_daily_tmax`, 2000-01-02 through 2023-12-31.
- Forecast anchor: latest eligible `public.hko_historical_forecasts_2000_2026` local forecast for target date T with `issue_at_utc <= T-1 23:59 HKT`.
- Hourly observations: `public.hko_info_gov_hourly_readings_1998_2026`, filtered to `dispatch_at_utc <= cutoff` and the prior 24 hours.
- Features: HKO, network, role, and station latest/snapshot/window features plus official-forecast contradiction transforms.
- Metrics: Pearson correlations, Spearman correlations for top features, temporal split stability, quantile residual spread, and guarded single-feature walk-forward residual correction.
- Confirmation guard: no rows on or after 2024-01-01.
