# HKO ARWF Station Forecasts

## Dataset identity and use

| Field | Value |
| --- | --- |
| dataset_id | 09_hko_arwf_station_forecasts |
| recommended_db_inclusion | YES-critical live collection |
| recommended_layer | live_nwp_anchor |
| current_operational_predictor_value_0_100 | 5 |
| diagnostic_research_value_0_100 | 35 |
| future_potential_0_100 | 98 |
| verdict | Potentially one of the best future independent anchors; only one cycle now, timestamps/station mapping need normalization. |
| source_tables_or_files | 1 |
| audited_attributes | 14 |
| profiled_rows_across_files | 530 |
| data_min | 2026-06-19T00:00:00+00:00 |
| data_max | 2026-06-28T00:00:00+00:00 |

## Source tables/files

| source_file | type | rows | attributes | data_min | data_max | db_action | db_layer | model_status | priority | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 09_hko_arwf_station_forecasts/hko_arwf_station_daily_forecasts.parquet | parquet | 530 | 14 | 2026-06-19T00:00:00+00:00 | 2026-06-28T00:00:00+00:00 | LOAD_LIVE_COLLECTION | live_nwp_anchor | HIGH_POTENTIAL_AFTER_HISTORY | CRITICAL | Potentially valuable unadjusted station NWP anchor; parse model/issue timestamps and station metadata. |

## Attribute nature summary

### Semantic classes

| semantic_class | attribute_count |
| --- | --- |
| boolean | 1 |
| datetime_or_date | 2 |
| hash_identifier | 1 |
| identifier_or_code | 2 |
| nominal_text | 3 |
| numeric | 5 |

### Model roles

| model_role | attribute_count |
| --- | --- |
| ARWF_STATION_METADATA | 3 |
| ARWF_STATION_TMAX_FORECAST | 1 |
| ARWF_STATION_TMIN_FORECAST | 1 |
| LEAKAGE_POLICY_METADATA | 3 |
| MODEL_INIT_ISSUE_VALID_TIME | 3 |
| PROVENANCE_ONLY | 1 |
| PROVENANCE_TIMESTAMP | 1 |
| SOURCE_OR_CATALOG_METADATA | 1 |

### Operational status

| operational_status | attribute_count |
| --- | --- |
| CONDITIONAL_CATEGORICAL_ONLY | 1 |
| ELIGIBILITY_AND_STALENESS | 3 |
| GATING_ONLY | 3 |
| GATING_OR_AUDIT_ONLY | 1 |
| HIGH_POTENTIAL_FUTURE_PREDICTOR | 2 |
| NOT_A_PREDICTOR | 1 |
| SPATIAL_CONTEXT | 3 |

### Storage decisions

| storage_decision | attribute_count |
| --- | --- |
| KEEP | 14 |

## Dataset-specific quality issues

| severity | source_table | attributes | evidence | required_action |
| --- | --- | --- | --- | --- |
| HIGH | hko_arwf_station_daily_forecasts.parquet | model_time; last_modified | Model time is numeric and constant; last_modified is string; only one cycle exists. | Parse cycle/issue/valid timestamps, calculate lead hours, and collect many cycles before scoring. |
| MEDIUM | hko_arwf_station_daily_forecasts.parquet | forecast_min/max_temperature_c | 28.3% missing. | Determine whether missingness is station capability, lead, or parser behavior; model only eligible station/lead cells. |

## Complete audited attribute dictionary

This table includes every audited attribute found for this dataset. It records the physical source file, source type, inferred semantic class, missingness, storage decision, DB layer, model role, operational status, quality action, usefulness score, observed profile min/max, and the audit rationale.

| source_file | attribute | source_dtype | semantic_class | row_count | non_null_count | null_count | null_pct | profile_min | profile_max | storage_decision | db_layer | model_role | operational_status | quality_action | usefulness_score_0_100 | rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 09_hko_arwf_station_forecasts/hko_arwf_station_daily_forecasts.parquet | source_id | string | identifier_or_code | 530 | 530 | 0 | 0.0 |  |  | KEEP | live_nwp_anchor | SOURCE_OR_CATALOG_METADATA | CONDITIONAL_CATEGORICAL_ONLY | NONE | 15 | Useful for lineage/source-era grouping, not meteorology by itself. |
| 09_hko_arwf_station_forecasts/hko_arwf_station_daily_forecasts.parquet | content_sha256 | string | hash_identifier | 530 | 530 | 0 | 0.0 |  |  | KEEP | live_nwp_anchor | PROVENANCE_ONLY | NOT_A_PREDICTOR | NONE | 0 | Integrity/deduplication key; essential for provenance, not forecasting. |
| 09_hko_arwf_station_forecasts/hko_arwf_station_daily_forecasts.parquet | raw_retrieved_at_utc | string | datetime_or_date | 530 | 530 | 0 | 0.0 | 2026-06-19T06:12:34.954063+00:00 | 2026-06-19T06:14:15.039351+00:00 | KEEP | live_nwp_anchor | PROVENANCE_TIMESTAMP | GATING_OR_AUDIT_ONLY | VALIDATE_TIMEZONE | 95 | Needed to prove retrieval/availability and reproduce vintages; not a weather variable. |
| 09_hko_arwf_station_forecasts/hko_arwf_station_daily_forecasts.parquet | station_code | string | identifier_or_code | 530 | 530 | 0 | 0.0 |  |  | KEEP | live_nwp_anchor | ARWF_STATION_METADATA | SPATIAL_CONTEXT | BUILD_CANONICAL_STATION_DIMENSION | 65 | Required for target station, gradients and station groups. |
| 09_hko_arwf_station_forecasts/hko_arwf_station_daily_forecasts.parquet | latitude | double | numeric | 530 | 530 | 0 | 0.0 | 22.182 | 22.529 | KEEP | live_nwp_anchor | ARWF_STATION_METADATA | SPATIAL_CONTEXT | BUILD_CANONICAL_STATION_DIMENSION | 65 | Required for target station, gradients and station groups. |
| 09_hko_arwf_station_forecasts/hko_arwf_station_daily_forecasts.parquet | longitude | double | numeric | 530 | 530 | 0 | 0.0 | 113.9 | 114.323 | KEEP | live_nwp_anchor | ARWF_STATION_METADATA | SPATIAL_CONTEXT | BUILD_CANONICAL_STATION_DIMENSION | 65 | Required for target station, gradients and station groups. |
| 09_hko_arwf_station_forecasts/hko_arwf_station_daily_forecasts.parquet | model_time | int64 | numeric | 530 | 530 | 0 | 0.0 | 2026061812.0 | 2026061812.0 | KEEP | live_nwp_anchor | MODEL_INIT_ISSUE_VALID_TIME | ELIGIBILITY_AND_STALENESS | PARSE_NUMERIC_OR_STRING_TO_TIMESTAMPTZ_AND_DERIVE_LEAD_HOURS | 100 | Essential to select the latest cycle available before T24. |
| 09_hko_arwf_station_forecasts/hko_arwf_station_daily_forecasts.parquet | last_modified | string | nominal_text | 530 | 530 | 0 | 0.0 |  |  | KEEP | live_nwp_anchor | MODEL_INIT_ISSUE_VALID_TIME | ELIGIBILITY_AND_STALENESS | PARSE_NUMERIC_OR_STRING_TO_TIMESTAMPTZ_AND_DERIVE_LEAD_HOURS | 100 | Essential to select the latest cycle available before T24. |
| 09_hko_arwf_station_forecasts/hko_arwf_station_daily_forecasts.parquet | forecast_date | string | datetime_or_date | 530 | 530 | 0 | 0.0 | 2026-06-19T00:00:00+00:00 | 2026-06-28T00:00:00+00:00 | KEEP | live_nwp_anchor | MODEL_INIT_ISSUE_VALID_TIME | ELIGIBILITY_AND_STALENESS | PARSE_NUMERIC_OR_STRING_TO_TIMESTAMPTZ_AND_DERIVE_LEAD_HOURS | 100 | Essential to select the latest cycle available before T24. |
| 09_hko_arwf_station_forecasts/hko_arwf_station_daily_forecasts.parquet | forecast_max_temperature_c | double | numeric | 530 | 380 | 150 | 0.283019 | 23.0 | 34.9 | KEEP | live_nwp_anchor | ARWF_STATION_TMAX_FORECAST | HIGH_POTENTIAL_FUTURE_PREDICTOR | COLLECT_LONG_HISTORY_AND_MAP_STATIONS | 95 | Potentially powerful independent NWP/station forecast anchor; current data cover one model cycle. |
| 09_hko_arwf_station_forecasts/hko_arwf_station_daily_forecasts.parquet | forecast_min_temperature_c | double | numeric | 530 | 380 | 150 | 0.283019 | 20.1 | 28.9 | KEEP | live_nwp_anchor | ARWF_STATION_TMIN_FORECAST | HIGH_POTENTIAL_FUTURE_PREDICTOR | COLLECT_LONG_HISTORY_AND_MAP_STATIONS | 70 | Useful for diurnal range and spatial model state. |
| 09_hko_arwf_station_forecasts/hko_arwf_station_daily_forecasts.parquet | availability_tier | string | nominal_text | 530 | 530 | 0 | 0.0 |  |  | KEEP | live_nwp_anchor | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 09_hko_arwf_station_forecasts/hko_arwf_station_daily_forecasts.parquet | operational_input_allowed | bool | boolean | 530 | 530 | 0 | 0.0 |  |  | KEEP | live_nwp_anchor | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 09_hko_arwf_station_forecasts/hko_arwf_station_daily_forecasts.parquet | source_time_policy | string | nominal_text | 530 | 530 | 0 | 0.0 |  |  | KEEP | live_nwp_anchor | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
