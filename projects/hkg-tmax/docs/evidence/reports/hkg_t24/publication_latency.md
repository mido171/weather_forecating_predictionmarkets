# HKG T24 Publication Latency

Generated: `2026-06-20T10:26:21.028268Z`

Current enforceable latency rules are conservative and source-specific. No experiment may select by `observed_at` alone.

## Active Rules

- HKO historical high-frequency station observations: `available_at = observed_at + 20 minutes`.
- Exact retrieved live HKO vintages: available no earlier than successful immutable retrieval time.
- HKO daily climate and Daily Extract labels: target/label-side only unless first-publication timing is proven.
- Reanalysis, final IMERG, final TC best track and retrospective archives: mechanism-only unless exact operational vintage and release lag are reconstructed.

HKO operational source contracts currently allowed by tier: `34`.

| Source | Cadence | Availability rule | Revision policy |
|---|---|---|---|
| hko_daily_extract_catalog | provider_updated | archive before resolving Daily Extract month/year payload choice | provider may update endYear/endMonth and coverage ranges |
| hko_daily_extract_year | annual_or_provider_rollup | latest annual payload is not proof of first-publication timing | provider may update annualized historical payloads |
| hko_clmmaxt_hko | monthly_update_of_daily_history | latest file is revised/finalized history, not necessarily first publication | verify against first-published Daily Extract |
| hko_daily_climate_download | monthly_or_element_specific | use for historical mechanisms; record update/revision behavior per element | finalized climate data may be revised |
| hko_open_data_catalog | provider_updated | archive catalog snapshots | provider may change endpoints/cadence |
| hko_api_documentation | provider_updated | archive PDF and hash | revalidate parsers when hash changes |
| hko_latest_1min_temperature | every_10_minutes | retrieved payload becomes available at successful retrieval; track source timestamp | provisional observations |
| hko_since_midnight_maxmin | every_10_minutes | retrieved payload becomes available at successful retrieval | provisional observations |
| hko_latest_relative_humidity | every_10_minutes | record provider timestamp and retrieval timestamp | provisional |
| hko_latest_pressure | every_10_minutes | record provider timestamp and retrieval timestamp | provisional |
| hko_latest_wind | every_10_minutes | record provider timestamp and retrieval timestamp | provisional |
| hko_latest_solar_radiation | every_10_minutes | preserve station/product differences | provisional |
| hko_latest_uv_index | every_15_minutes | retrieved payload becomes available at successful retrieval; track source timestamp | provisional |
| hko_automatic_rainfall | every_15_minutes | preserve product-specific station semantics | provisional; HKO-labelled rainfall may differ from official climatological rainfall |
| hko_latest_visibility | every_10_minutes | retrieved payload becomes available at successful retrieval; track source timestamp | provisional |
| hko_current_weather_report | provider_updated | archive every changed payload and source timestamp | each changed payload is a separate vintage |
| hko_gridded_rainfall_nowcast | every_12_minutes | archive every issue with valid times | new issue supersedes but never overwrites older issue |
| hko_radar | approximately_every_6_minutes | archive native image plus metadata; no future frames | preserve each frame |
| hko_radar_image_manifest | approximately_every_6_minutes | archive every changed manifest and only use frames referenced by retrieval-time manifests | preserve each manifest/frame vintage |
| hko_lightning | page_approximately_5_minutes_counts_15_minutes | archive source timestamps | provisional |
| hko_lightning_counts_latest | every_15_minutes | archive source timestamps and retrieval timestamps | provisional |
| hko_satellite | product_specific | preserve channel, scan time, publication time, projection | preserve each image |
| hko_local_weather_forecast | hourly_and_as_needed | archive every changed payload and issue/update time | each issue is a separate vintage |
| hko_nine_day_forecast | at_least_twice_daily_and_as_needed | archive each forecast vintage | each issue is a separate vintage |
| hko_weather_warning_summary | warning_update | archive every changed warning payload and issue/update time | each issue is a separate vintage |
| hko_weather_warning_information | warning_update | archive every changed warning payload and issue/update time | each issue is a separate vintage |
| hko_special_weather_tips | provider_updated | archive every changed payload and source timestamp | each issue is a separate vintage |
| hko_automatic_regional_forecast | product_specific | archive issue time and station/valid hour | each issue is a vintage |
| hko_upper_air | typically_twice_daily | sounding launch/receipt time plus processing latency | distinguish preliminary and corrected |
| hko_station_metadata | provider_updated | archive snapshots and effective dates | provider may update metadata |
| hko_south_china_coastal_waters_bulletin | provider_updated | archive every changed bulletin and source timestamp | each issue is a separate vintage |
| hko_latest_tidal_information | provider_updated | archive every changed payload and source timestamp | provisional |
| hko_tropical_cyclone_realtime | advisory_specific | archive each advisory and forecast track | never substitute final best track |
| hko_tropical_cyclone_track_realtime | advisory_specific | archive each retrieved track-list payload with retrieval timestamp | never substitute final best track |
