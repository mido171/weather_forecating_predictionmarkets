# NOAA ISD Regional Surface

## Dataset identity and use

| Field | Value |
| --- | --- |
| dataset_id | 04_noaa_isd_regional_surface |
| recommended_db_inclusion | YES-raw quarantine + rebuilt station schema |
| recommended_layer | diagnostic_station_network |
| current_operational_predictor_value_0_100 | 0 |
| diagnostic_research_value_0_100 | 98 |
| future_potential_0_100 | 92 |
| verdict | Highest-value non-forecast research source, but wind parser, coordinates and operational-vintage contract must be repaired. |
| source_tables_or_files | 2 |
| audited_attributes | 34 |
| profiled_rows_across_files | 4346780 |
| data_min | 1945-11-30T16:00:00+00:00 |
| data_max | 2025-08-25T00:00:00+00:00 |

## Source tables/files

| source_file | type | rows | attributes | data_min | data_max | db_action | db_layer | model_status | priority | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 04_noaa_isd_regional_surface/noaa_isd_core_observations.parquet | parquet | 4029291 | 21 | 1945-11-30T16:00:00+00:00 | 2025-08-24T21:30:00+00:00 | LOAD_RAW_QUARANTINE_AND_REBUILD_CLEAN | diagnostic_station_network | DIAGNOSTIC_ONLY_UNTIL_VINTAGE_PROOF | CRITICAL | Fix constant wind direction and station-coordinate metadata before feature derivation. |
| 04_noaa_isd_regional_surface/noaa_isd_station_day_cutoff_summary.parquet | parquet | 317489 | 13 | 1945-12-01T00:00:00+00:00 | 2025-08-25T00:00:00+00:00 | REBUILD_AFTER_RAW_FIX | diagnostic_station_network | DIAGNOSTIC_ONLY_UNTIL_VINTAGE_PROOF | CRITICAL | Summary inherits wind parser defect; verify all aggregations stop at cutoff. |

## Attribute nature summary

### Semantic classes

| semantic_class | attribute_count |
| --- | --- |
| boolean | 2 |
| datetime_or_date | 5 |
| hash_identifier | 1 |
| identifier_or_code | 3 |
| nominal_text | 7 |
| numeric | 16 |

### Model roles

| model_role | attribute_count |
| --- | --- |
| COVERAGE_AND_MISSINGNESS | 1 |
| LEAKAGE_POLICY_METADATA | 5 |
| OBSERVATION_TIME | 4 |
| PROVENANCE_ONLY | 1 |
| PROVENANCE_TIMESTAMP | 1 |
| QUALITY_CONTROL | 4 |
| SOURCE_OR_CATALOG_METADATA | 1 |
| STATION_KEY | 2 |
| STATION_METADATA | 3 |
| STATION_METEOROLOGY | 10 |
| WIND_REGIME | 2 |

### Operational status

| operational_status | attribute_count |
| --- | --- |
| CONDITIONAL_CATEGORICAL_ONLY | 1 |
| DIAGNOSTIC_ALIGNMENT_ONLY | 2 |
| DIAGNOSTIC_ONLY_UNTIL_VINTAGE_PROOF | 14 |
| DIAGNOSTIC_UNCERTAINTY_FEATURE | 1 |
| GATING_ONLY | 5 |
| GATING_OR_AUDIT_ONLY | 1 |
| NOT_A_PREDICTOR | 1 |
| POSSIBLE_MISSINGNESS_OR_TRUST_FEATURE_ONLY | 4 |
| REJECT_CURRENT_FIELD | 2 |
| STATIC_CONTEXT_ONLY | 3 |

### Storage decisions

| storage_decision | attribute_count |
| --- | --- |
| KEEP | 29 |
| KEEP_RAW_AND_CANONICAL_JOIN | 3 |
| KEEP_RAW_REBUILD | 2 |

## Dataset-specific quality issues

| severity | source_table | attributes | evidence | required_action |
| --- | --- | --- | --- | --- |
| CRITICAL | both tables | wind_direction_deg; wind_direction_deg_latest_before_1500 | Both fields are constant at 20 degrees across millions/hundreds of thousands of rows. | Treat current wind direction and all dependent u/v/directional features as invalid. Fix extractor and rebuild downstream matrices. |
| HIGH | noaa_isd_core_observations.parquet | latitude; longitude | Profile includes latitude 0 and longitude -114.283/144.2 despite a regional Hong Kong/South China network. | Use date-effective NOAA station history metadata, not row-level coordinates, and quarantine impossible station metadata. |
| HIGH | noaa_isd_station_day_cutoff_summary.parquet | daily_air_temperature_min_c; daily_air_temperature_max_c | Column names do not prove values were truncated at 15:00 HKT. | Reject as predictor until aggregation code proves no post-cutoff observations entered. |

## Complete audited attribute dictionary

This table includes every audited attribute found for this dataset. It records the physical source file, source type, inferred semantic class, missingness, storage decision, DB layer, model role, operational status, quality action, usefulness score, observed profile min/max, and the audit rationale.

| source_file | attribute | source_dtype | semantic_class | row_count | non_null_count | null_count | null_pct | profile_min | profile_max | storage_decision | db_layer | model_role | operational_status | quality_action | usefulness_score_0_100 | rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 04_noaa_isd_regional_surface/noaa_isd_core_observations.parquet | station_id | string | identifier_or_code | 4029291 | 4029291 | 0 | 0.0 |  |  | KEEP | diagnostic_station_network | STATION_KEY | DIAGNOSTIC_ALIGNMENT_ONLY | JOIN_DATE_EFFECTIVE_STATION_DOSSIER | 80 | Core key for spatial-network features. |
| 04_noaa_isd_regional_surface/noaa_isd_core_observations.parquet | observed_at_utc | string | datetime_or_date | 4029291 | 4029291 | 0 | 0.0 | 1945-11-30T16:00:00+00:00 | 2025-08-24T21:30:00+00:00 | KEEP | diagnostic_station_network | OBSERVATION_TIME | DIAGNOSTIC_ONLY_UNTIL_VINTAGE_PROOF | VERIFY_TIMEZONE_AND_STRICT_CUTOFF | 95 | Essential for T24 truncation; archive valid time is not availability proof. |
| 04_noaa_isd_regional_surface/noaa_isd_core_observations.parquet | observed_at_hkt | string | datetime_or_date | 4029291 | 4029291 | 0 | 0.0 | 1945-11-30T16:00:00+00:00 | 2025-08-24T21:30:00+00:00 | KEEP | diagnostic_station_network | OBSERVATION_TIME | DIAGNOSTIC_ONLY_UNTIL_VINTAGE_PROOF | VERIFY_TIMEZONE_AND_STRICT_CUTOFF | 95 | Essential for T24 truncation; archive valid time is not availability proof. |
| 04_noaa_isd_regional_surface/noaa_isd_core_observations.parquet | report_type | string | nominal_text | 4029291 | 4029291 | 0 | 0.0 |  |  | KEEP | diagnostic_station_network | QUALITY_CONTROL | POSSIBLE_MISSINGNESS_OR_TRUST_FEATURE_ONLY | STANDARDIZE_ENUMS_AND_QUARANTINE_BAD_ROWS | 35 | Quality/trust metadata; useful for filtering and possibly uncertainty, not primary physical signal. |
| 04_noaa_isd_regional_surface/noaa_isd_core_observations.parquet | latitude | double | numeric | 4029291 | 4029286 | 5 | 1e-06 | 0.0 | 24.55 | KEEP_RAW_AND_CANONICAL_JOIN | diagnostic_station_network | STATION_METADATA | STATIC_CONTEXT_ONLY | REPLACE_ROW_METADATA_WITH_DATE_EFFECTIVE_NOAA_STATION_HISTORY | 55 | Useful spatial context, but profile shows implausible row metadata values. |
| 04_noaa_isd_regional_surface/noaa_isd_core_observations.parquet | longitude | double | numeric | 4029291 | 4029281 | 10 | 2e-06 | -114.283 | 144.2 | KEEP_RAW_AND_CANONICAL_JOIN | diagnostic_station_network | STATION_METADATA | STATIC_CONTEXT_ONLY | REPLACE_ROW_METADATA_WITH_DATE_EFFECTIVE_NOAA_STATION_HISTORY | 55 | Useful spatial context, but profile shows implausible row metadata values. |
| 04_noaa_isd_regional_surface/noaa_isd_core_observations.parquet | elevation_m | double | numeric | 4029291 | 4023198 | 6093 | 0.001512 | 0.0 | 284.0 | KEEP_RAW_AND_CANONICAL_JOIN | diagnostic_station_network | STATION_METADATA | STATIC_CONTEXT_ONLY | REPLACE_ROW_METADATA_WITH_DATE_EFFECTIVE_NOAA_STATION_HISTORY | 55 | Useful spatial context, but profile shows implausible row metadata values. |
| 04_noaa_isd_regional_surface/noaa_isd_core_observations.parquet | wind_direction_deg | int64 | numeric | 4029291 | 4029291 | 0 | 0.0 | 20.0 | 20.0 | KEEP_RAW_REBUILD | diagnostic_station_network | WIND_REGIME | REJECT_CURRENT_FIELD | FIX_PARSER_CONSTANT_20_DEGREES | 95 | Extremely valuable in principle, but current field is constant at 20 degrees and unusable. |
| 04_noaa_isd_regional_surface/noaa_isd_core_observations.parquet | wind_speed_mps | double | numeric | 4029291 | 4007867 | 21424 | 0.005317 | 0.0 | 84.0 | KEEP | diagnostic_station_network | STATION_METEOROLOGY | DIAGNOSTIC_ONLY_UNTIL_VINTAGE_PROOF | NONE | 90 | High-value regional thermal/moisture/pressure/wind signal after QC and strict cutoff construction. |
| 04_noaa_isd_regional_surface/noaa_isd_core_observations.parquet | air_temperature_c | double | numeric | 4029291 | 3902701 | 126590 | 0.031417 | -13.6 | 54.4 | KEEP | diagnostic_station_network | STATION_METEOROLOGY | DIAGNOSTIC_ONLY_UNTIL_VINTAGE_PROOF | NONE | 90 | High-value regional thermal/moisture/pressure/wind signal after QC and strict cutoff construction. |
| 04_noaa_isd_regional_surface/noaa_isd_core_observations.parquet | dew_point_c | double | numeric | 4029291 | 3881634 | 147657 | 0.036646 | -65.6 | 36.0 | KEEP | diagnostic_station_network | STATION_METEOROLOGY | DIAGNOSTIC_ONLY_UNTIL_VINTAGE_PROOF | NONE | 90 | High-value regional thermal/moisture/pressure/wind signal after QC and strict cutoff construction. |
| 04_noaa_isd_regional_surface/noaa_isd_core_observations.parquet | sea_level_pressure_hpa | double | numeric | 4029291 | 2335942 | 1693349 | 0.42026 | 901.1 | 1088.2 | KEEP | diagnostic_station_network | STATION_METEOROLOGY | DIAGNOSTIC_ONLY_UNTIL_VINTAGE_PROOF | NONE | 75 | High-value regional thermal/moisture/pressure/wind signal after QC and strict cutoff construction. |
| 04_noaa_isd_regional_surface/noaa_isd_core_observations.parquet | temperature_quality_code | string | nominal_text | 4029291 | 4029291 | 0 | 0.0 |  |  | KEEP | diagnostic_station_network | QUALITY_CONTROL | POSSIBLE_MISSINGNESS_OR_TRUST_FEATURE_ONLY | STANDARDIZE_ENUMS_AND_QUARANTINE_BAD_ROWS | 35 | Quality/trust metadata; useful for filtering and possibly uncertainty, not primary physical signal. |
| 04_noaa_isd_regional_surface/noaa_isd_core_observations.parquet | dew_point_quality_code | string | nominal_text | 4029291 | 4029291 | 0 | 0.0 |  |  | KEEP | diagnostic_station_network | QUALITY_CONTROL | POSSIBLE_MISSINGNESS_OR_TRUST_FEATURE_ONLY | STANDARDIZE_ENUMS_AND_QUARANTINE_BAD_ROWS | 35 | Quality/trust metadata; useful for filtering and possibly uncertainty, not primary physical signal. |
| 04_noaa_isd_regional_surface/noaa_isd_core_observations.parquet | sea_level_pressure_quality_code | string | nominal_text | 4029291 | 4029291 | 0 | 0.0 |  |  | KEEP | diagnostic_station_network | QUALITY_CONTROL | POSSIBLE_MISSINGNESS_OR_TRUST_FEATURE_ONLY | STANDARDIZE_ENUMS_AND_QUARANTINE_BAD_ROWS | 35 | Quality/trust metadata; useful for filtering and possibly uncertainty, not primary physical signal. |
| 04_noaa_isd_regional_surface/noaa_isd_core_observations.parquet | source_id | string | identifier_or_code | 4029291 | 4029291 | 0 | 0.0 |  |  | KEEP | diagnostic_station_network | SOURCE_OR_CATALOG_METADATA | CONDITIONAL_CATEGORICAL_ONLY | NONE | 15 | Useful for lineage/source-era grouping, not meteorology by itself. |
| 04_noaa_isd_regional_surface/noaa_isd_core_observations.parquet | content_sha256 | string | hash_identifier | 4029291 | 4029291 | 0 | 0.0 |  |  | KEEP | diagnostic_station_network | PROVENANCE_ONLY | NOT_A_PREDICTOR | NONE | 0 | Integrity/deduplication key; essential for provenance, not forecasting. |
| 04_noaa_isd_regional_surface/noaa_isd_core_observations.parquet | raw_retrieved_at_utc | string | datetime_or_date | 4029291 | 4029291 | 0 | 0.0 | 2026-06-19T06:23:47.650000+00:00 | 2026-06-19T06:45:31.127456+00:00 | KEEP | diagnostic_station_network | PROVENANCE_TIMESTAMP | GATING_OR_AUDIT_ONLY | VALIDATE_TIMEZONE | 95 | Needed to prove retrieval/availability and reproduce vintages; not a weather variable. |
| 04_noaa_isd_regional_surface/noaa_isd_core_observations.parquet | availability_tier | string | nominal_text | 4029291 | 4029291 | 0 | 0.0 |  |  | KEEP | diagnostic_station_network | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 04_noaa_isd_regional_surface/noaa_isd_core_observations.parquet | operational_input_allowed | bool | boolean | 4029291 | 4029291 | 0 | 0.0 |  |  | KEEP | diagnostic_station_network | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 04_noaa_isd_regional_surface/noaa_isd_core_observations.parquet | source_time_policy | string | nominal_text | 4029291 | 4029291 | 0 | 0.0 |  |  | KEEP | diagnostic_station_network | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 04_noaa_isd_regional_surface/noaa_isd_station_day_cutoff_summary.parquet | station_id | string | identifier_or_code | 317489 | 317489 | 0 | 0.0 |  |  | KEEP | diagnostic_station_network | STATION_KEY | DIAGNOSTIC_ALIGNMENT_ONLY | JOIN_DATE_EFFECTIVE_STATION_DOSSIER | 80 | Core key for spatial-network features. |
| 04_noaa_isd_regional_surface/noaa_isd_station_day_cutoff_summary.parquet | local_date | string | datetime_or_date | 317489 | 317489 | 0 | 0.0 | 1945-12-01T00:00:00+00:00 | 2025-08-25T00:00:00+00:00 | KEEP | diagnostic_station_network | OBSERVATION_TIME | DIAGNOSTIC_ONLY_UNTIL_VINTAGE_PROOF | VERIFY_TIMEZONE_AND_STRICT_CUTOFF | 95 | Essential for T24 truncation; archive valid time is not availability proof. |
| 04_noaa_isd_regional_surface/noaa_isd_station_day_cutoff_summary.parquet | obs_count | int64 | numeric | 317489 | 317489 | 0 | 0.0 | 1.0 | 126.0 | KEEP | diagnostic_station_network | COVERAGE_AND_MISSINGNESS | DIAGNOSTIC_UNCERTAINTY_FEATURE | CONFIRM_COUNT_IS_PRE_CUTOFF_ONLY | 55 | Coverage may be both quality control and regime/uncertainty signal. |
| 04_noaa_isd_regional_surface/noaa_isd_station_day_cutoff_summary.parquet | latest_before_1500_hkt | string | datetime_or_date | 317489 | 317489 | 0 | 0.0 | 1945-12-01T07:00:00+00:00 | 2025-08-24T21:30:00+00:00 | KEEP | diagnostic_station_network | OBSERVATION_TIME | DIAGNOSTIC_ONLY_UNTIL_VINTAGE_PROOF | VERIFY_TIMEZONE_AND_STRICT_CUTOFF | 95 | Essential for T24 truncation; archive valid time is not availability proof. |
| 04_noaa_isd_regional_surface/noaa_isd_station_day_cutoff_summary.parquet | air_temperature_c_latest_before_1500 | double | numeric | 317489 | 308855 | 8634 | 0.027195 | -10.0 | 49.0 | KEEP | diagnostic_station_network | STATION_METEOROLOGY | DIAGNOSTIC_ONLY_UNTIL_VINTAGE_PROOF | NONE | 90 | High-value regional thermal/moisture/pressure/wind signal after QC and strict cutoff construction. |
| 04_noaa_isd_regional_surface/noaa_isd_station_day_cutoff_summary.parquet | dew_point_c_latest_before_1500 | double | numeric | 317489 | 305448 | 12041 | 0.037926 | -24.0 | 35.3 | KEEP | diagnostic_station_network | STATION_METEOROLOGY | DIAGNOSTIC_ONLY_UNTIL_VINTAGE_PROOF | NONE | 90 | High-value regional thermal/moisture/pressure/wind signal after QC and strict cutoff construction. |
| 04_noaa_isd_regional_surface/noaa_isd_station_day_cutoff_summary.parquet | sea_level_pressure_hpa_latest_before_1500 | double | numeric | 317489 | 256130 | 61359 | 0.193263 | 910.0 | 1077.0 | KEEP | diagnostic_station_network | STATION_METEOROLOGY | DIAGNOSTIC_ONLY_UNTIL_VINTAGE_PROOF | NONE | 90 | High-value regional thermal/moisture/pressure/wind signal after QC and strict cutoff construction. |
| 04_noaa_isd_regional_surface/noaa_isd_station_day_cutoff_summary.parquet | wind_direction_deg_latest_before_1500 | double | numeric | 317489 | 316755 | 734 | 0.002312 | 20.0 | 20.0 | KEEP_RAW_REBUILD | diagnostic_station_network | WIND_REGIME | REJECT_CURRENT_FIELD | FIX_PARSER_CONSTANT_20_DEGREES | 95 | Extremely valuable in principle, but current field is constant at 20 degrees and unusable. |
| 04_noaa_isd_regional_surface/noaa_isd_station_day_cutoff_summary.parquet | wind_speed_mps_latest_before_1500 | double | numeric | 317489 | 315120 | 2369 | 0.007462 | 0.0 | 84.0 | KEEP | diagnostic_station_network | STATION_METEOROLOGY | DIAGNOSTIC_ONLY_UNTIL_VINTAGE_PROOF | NONE | 90 | High-value regional thermal/moisture/pressure/wind signal after QC and strict cutoff construction. |
| 04_noaa_isd_regional_surface/noaa_isd_station_day_cutoff_summary.parquet | daily_air_temperature_min_c | double | numeric | 317489 | 310893 | 6596 | 0.020776 | -13.6 | 46.5 | KEEP | diagnostic_station_network | STATION_METEOROLOGY | DIAGNOSTIC_ONLY_UNTIL_VINTAGE_PROOF | NONE | 90 | High-value regional thermal/moisture/pressure/wind signal after QC and strict cutoff construction. |
| 04_noaa_isd_regional_surface/noaa_isd_station_day_cutoff_summary.parquet | daily_air_temperature_max_c | double | numeric | 317489 | 310893 | 6596 | 0.020776 | -10.0 | 54.4 | KEEP | diagnostic_station_network | STATION_METEOROLOGY | DIAGNOSTIC_ONLY_UNTIL_VINTAGE_PROOF | NONE | 90 | High-value regional thermal/moisture/pressure/wind signal after QC and strict cutoff construction. |
| 04_noaa_isd_regional_surface/noaa_isd_station_day_cutoff_summary.parquet | availability_tier | string | nominal_text | 317489 | 317489 | 0 | 0.0 |  |  | KEEP | diagnostic_station_network | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 04_noaa_isd_regional_surface/noaa_isd_station_day_cutoff_summary.parquet | operational_input_allowed | bool | boolean | 317489 | 317489 | 0 | 0.0 |  |  | KEEP | diagnostic_station_network | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
