# Point-In-Time Eligibility

Primary cutoff: `T-1 15:00 HKT`.

The registry below is generated from `metadata/feature_eligibility_registry.*`. Rows marked target-only, retrospective, or prospective-only are forbidden as operational predictors.

## Eligible Or Conditionally Eligible

| feature_family | feature_name | role | eligible_at_tminus1_1500_hkt | available_at_rule | notes |
| --- | --- | --- | --- | --- | --- |
| official_daily_hko_lagged_tmax | hko_tminus2_official_tmax_c | PROXY_WITH_LIMITATIONS | True | requires empirical publication-lag proof before production | Useful benchmark feature, but publication timing must be proven before production. |
| hko_high_frequency_hq | hko_temp_at_tminus1_1500_c | OPERATIONAL_WITH_CONSERVATIVE_LATENCY | True | observed_at + 20 minutes | Use latest observation available by cutoff. |
| hko_high_frequency_hq | hko_rh_at_tminus1_1500_pct | OPERATIONAL_WITH_CONSERVATIVE_LATENCY | True | observed_at + 20 minutes | Moisture state at cutoff. |
| hko_high_frequency_hq | hko_mslp_at_tminus1_1500_hpa | OPERATIONAL_WITH_CONSERVATIVE_LATENCY | True | observed_at + 20 minutes | Synoptic pressure state at cutoff. |
| hko_high_frequency_hq | hko_tminus1_max_so_far_1500_c | OPERATIONAL_WITH_CONSERVATIVE_LATENCY | True | observed_at + 20 minutes | Only valid for T-1, not target day T. |
| static_geospatial | station_distance_bearing_static_context | STATIC_DETERMINISTIC | True | static | Eligible once station identity/history is resolved. |

## Rejected For Operational T-24 Features

| feature_family | feature_name | role | eligible_at_tminus1_1500_hkt | available_at_rule | notes |
| --- | --- | --- | --- | --- | --- |
| official_daily_hko_tmax | target_tmax_c | TARGET_ONLY | False | after target day completion and publication | Target label only. Never a predictor. |
| official_daily_climate_same_day | daily_rainfall/cloud/sunshine/etc_for_target_day | RETROSPECTIVE_MECHANISM_ONLY | False | after target day completion and publication | Allowed for mechanism EDA only. Forbidden as operational T-24 predictors. |
| current_nwp_only | current_gfs_gefs_cycle_payloads | PROSPECTIVE_ONLY_NOT_YET_BACKTESTABLE | False | only current/prospective cycles acquired | Needs historical cycle archive before retrospective evaluation. |
| tc_best_track | retrospective_best_track_intensity_position | RETROSPECTIVE_MECHANISM_ONLY | False | final best track after event | Use only for mechanism analysis unless advisory vintage archive is built. |
