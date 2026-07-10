# HKO Marine Tide Coastal Waters

## Dataset identity and use

| Field | Value |
| --- | --- |
| dataset_id | 08_hko_marine_tide_coastal_waters |
| recommended_db_inclusion | YES-live exact-vintage collection |
| recommended_layer | live_exact_vintage |
| current_operational_predictor_value_0_100 | 5 |
| diagnostic_research_value_0_100 | 40 |
| future_potential_0_100 | 60 |
| verdict | Secondary marine-regime source; current sample is tiny and tide likely indirect. |
| source_tables_or_files | 2 |
| audited_attributes | 18 |
| profiled_rows_across_files | 135 |
| data_min | 2026-06-19T04:30:00+00:00 |
| data_max | 2026-06-20T07:50:00+00:00 |

## Source tables/files

| source_file | type | rows | attributes | data_min | data_max | db_action | db_layer | model_status | priority | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 08_hko_marine_tide_coastal_waters/hko_latest_tidal_information.parquet | parquet | 105 | 8 | 2026-06-19T04:45:00+00:00 | 2026-06-20T07:50:00+00:00 | LOAD_LIVE_COLLECTION | live_exact_vintage | FUTURE_AFTER_HISTORY | MEDIUM | Keep exact-vintage feed; current history is insufficient. |
| 08_hko_marine_tide_coastal_waters/hko_south_china_coastal_waters_bulletin.parquet | parquet | 30 | 10 | 2026-06-19T04:30:00+00:00 | 2026-06-20T04:30:00+00:00 | LOAD_LIVE_COLLECTION | live_exact_vintage | FUTURE_AFTER_HISTORY | MEDIUM | Keep exact-vintage feed; current history is insufficient. |

## Attribute nature summary

### Semantic classes

| semantic_class | attribute_count |
| --- | --- |
| boolean | 2 |
| datetime_or_date | 4 |
| hash_identifier | 2 |
| identifier_or_code | 2 |
| nominal_text | 7 |
| numeric | 1 |

### Model roles

| model_role | attribute_count |
| --- | --- |
| COASTAL_WATER_LEVEL | 1 |
| LEAKAGE_POLICY_METADATA | 4 |
| LIVE_TIME | 2 |
| MARINE_AREA_KEY | 1 |
| MARINE_REGIME_TEXT | 3 |
| PROVENANCE_ONLY | 2 |
| PROVENANCE_TIMESTAMP | 2 |
| SOURCE_OR_CATALOG_METADATA | 2 |
| TIDE_STATION_KEY | 1 |

### Operational status

| operational_status | attribute_count |
| --- | --- |
| CONDITIONAL_CATEGORICAL_ONLY | 2 |
| FUTURE_LIVE_NLP_PREDICTOR | 3 |
| FUTURE_LIVE_PREDICTOR | 2 |
| GATING_ONLY | 4 |
| GATING_OR_AUDIT_ONLY | 2 |
| LOW_TO_MODERATE_FUTURE_PREDICTOR | 1 |
| NOT_A_PREDICTOR | 2 |
| SPATIAL_CONTEXT | 2 |

### Storage decisions

| storage_decision | attribute_count |
| --- | --- |
| KEEP | 15 |
| KEEP_FULL_TEXT | 3 |

## Dataset-specific quality issues

| severity | source_table | attributes | evidence | required_action |
| --- | --- | --- | --- | --- |
| MEDIUM | hko_latest_tidal_information.parquet | height_m | 14.3% of tide heights are missing. | Retain with station-specific availability flags; do not impute indiscriminately. |

## Complete audited attribute dictionary

This table includes every audited attribute found for this dataset. It records the physical source file, source type, inferred semantic class, missingness, storage decision, DB layer, model role, operational status, quality action, usefulness score, observed profile min/max, and the audit rationale.

| source_file | attribute | source_dtype | semantic_class | row_count | non_null_count | null_count | null_pct | profile_min | profile_max | storage_decision | db_layer | model_role | operational_status | quality_action | usefulness_score_0_100 | rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 08_hko_marine_tide_coastal_waters/hko_latest_tidal_information.parquet | source_id | string | identifier_or_code | 105 | 105 | 0 | 0.0 |  |  | KEEP | live_exact_vintage | SOURCE_OR_CATALOG_METADATA | CONDITIONAL_CATEGORICAL_ONLY | NONE | 15 | Useful for lineage/source-era grouping, not meteorology by itself. |
| 08_hko_marine_tide_coastal_waters/hko_latest_tidal_information.parquet | content_sha256 | string | hash_identifier | 105 | 105 | 0 | 0.0 |  |  | KEEP | live_exact_vintage | PROVENANCE_ONLY | NOT_A_PREDICTOR | NONE | 0 | Integrity/deduplication key; essential for provenance, not forecasting. |
| 08_hko_marine_tide_coastal_waters/hko_latest_tidal_information.parquet | raw_retrieved_at_utc | string | datetime_or_date | 105 | 105 | 0 | 0.0 | 2026-06-19T04:50:57.280298+00:00 | 2026-06-20T07:56:27.102158+00:00 | KEEP | live_exact_vintage | PROVENANCE_TIMESTAMP | GATING_OR_AUDIT_ONLY | VALIDATE_TIMEZONE | 95 | Needed to prove retrieval/availability and reproduce vintages; not a weather variable. |
| 08_hko_marine_tide_coastal_waters/hko_latest_tidal_information.parquet | tide_station | string | nominal_text | 105 | 105 | 0 | 0.0 |  |  | KEEP | live_exact_vintage | TIDE_STATION_KEY | SPATIAL_CONTEXT | JOIN_STATION_METADATA | 30 | Needed for spatial water-level features. |
| 08_hko_marine_tide_coastal_waters/hko_latest_tidal_information.parquet | observed_at_hkt | string | datetime_or_date | 105 | 105 | 0 | 0.0 | 2026-06-19T04:45:00+00:00 | 2026-06-20T07:50:00+00:00 | KEEP | live_exact_vintage | LIVE_TIME | FUTURE_LIVE_PREDICTOR | PARSE_TIMESTAMPTZ_AND_ENFORCE_CUTOFF | 90 | Exact-vintage time key. |
| 08_hko_marine_tide_coastal_waters/hko_latest_tidal_information.parquet | height_m | double | numeric | 105 | 90 | 15 | 0.142857 | 1.04 | 2.8 | KEEP | live_exact_vintage | COASTAL_WATER_LEVEL | LOW_TO_MODERATE_FUTURE_PREDICTOR | HANDLE_MISSING_AND_DERIVE_ANOMALY_VS_ASTRONOMICAL_TIDE | 45 | Indirect marine/pressure/wind regime proxy; unlikely to be a primary Tmax driver. |
| 08_hko_marine_tide_coastal_waters/hko_latest_tidal_information.parquet | availability_tier | string | nominal_text | 105 | 105 | 0 | 0.0 |  |  | KEEP | live_exact_vintage | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 08_hko_marine_tide_coastal_waters/hko_latest_tidal_information.parquet | operational_input_allowed | bool | boolean | 105 | 105 | 0 | 0.0 |  |  | KEEP | live_exact_vintage | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 08_hko_marine_tide_coastal_waters/hko_south_china_coastal_waters_bulletin.parquet | source_id | string | identifier_or_code | 30 | 30 | 0 | 0.0 |  |  | KEEP | live_exact_vintage | SOURCE_OR_CATALOG_METADATA | CONDITIONAL_CATEGORICAL_ONLY | NONE | 15 | Useful for lineage/source-era grouping, not meteorology by itself. |
| 08_hko_marine_tide_coastal_waters/hko_south_china_coastal_waters_bulletin.parquet | content_sha256 | string | hash_identifier | 30 | 30 | 0 | 0.0 |  |  | KEEP | live_exact_vintage | PROVENANCE_ONLY | NOT_A_PREDICTOR | NONE | 0 | Integrity/deduplication key; essential for provenance, not forecasting. |
| 08_hko_marine_tide_coastal_waters/hko_south_china_coastal_waters_bulletin.parquet | raw_retrieved_at_utc | string | datetime_or_date | 30 | 30 | 0 | 0.0 | 2026-06-19T04:50:55.669615+00:00 | 2026-06-20T07:56:24.707400+00:00 | KEEP | live_exact_vintage | PROVENANCE_TIMESTAMP | GATING_OR_AUDIT_ONLY | VALIDATE_TIMEZONE | 95 | Needed to prove retrieval/availability and reproduce vintages; not a weather variable. |
| 08_hko_marine_tide_coastal_waters/hko_south_china_coastal_waters_bulletin.parquet | update_time_hkt | string | datetime_or_date | 30 | 30 | 0 | 0.0 | 2026-06-19T04:30:00+00:00 | 2026-06-20T04:30:00+00:00 | KEEP | live_exact_vintage | LIVE_TIME | FUTURE_LIVE_PREDICTOR | PARSE_TIMESTAMPTZ_AND_ENFORCE_CUTOFF | 90 | Exact-vintage time key. |
| 08_hko_marine_tide_coastal_waters/hko_south_china_coastal_waters_bulletin.parquet | location_name | string | nominal_text | 30 | 30 | 0 | 0.0 |  |  | KEEP | live_exact_vintage | MARINE_AREA_KEY | SPATIAL_CONTEXT | NORMALIZE_AREAS | 35 | Needed to distinguish adjacent waters and remote coastal sectors. |
| 08_hko_marine_tide_coastal_waters/hko_south_china_coastal_waters_bulletin.parquet | wind_info | string | nominal_text | 30 | 30 | 0 | 0.0 |  |  | KEEP_FULL_TEXT | live_exact_vintage | MARINE_REGIME_TEXT | FUTURE_LIVE_NLP_PREDICTOR | PARSE_DIRECTION_FORCE_CONVECTION_AND_SEA_STATE | 65 | Potential marine-flow and convection context; current sample is tiny. |
| 08_hko_marine_tide_coastal_waters/hko_south_china_coastal_waters_bulletin.parquet | weather_description | string | nominal_text | 30 | 30 | 0 | 0.0 |  |  | KEEP_FULL_TEXT | live_exact_vintage | MARINE_REGIME_TEXT | FUTURE_LIVE_NLP_PREDICTOR | PARSE_DIRECTION_FORCE_CONVECTION_AND_SEA_STATE | 65 | Potential marine-flow and convection context; current sample is tiny. |
| 08_hko_marine_tide_coastal_waters/hko_south_china_coastal_waters_bulletin.parquet | sea_situation | string | nominal_text | 30 | 30 | 0 | 0.0 |  |  | KEEP_FULL_TEXT | live_exact_vintage | MARINE_REGIME_TEXT | FUTURE_LIVE_NLP_PREDICTOR | PARSE_DIRECTION_FORCE_CONVECTION_AND_SEA_STATE | 65 | Potential marine-flow and convection context; current sample is tiny. |
| 08_hko_marine_tide_coastal_waters/hko_south_china_coastal_waters_bulletin.parquet | availability_tier | string | nominal_text | 30 | 30 | 0 | 0.0 |  |  | KEEP | live_exact_vintage | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 08_hko_marine_tide_coastal_waters/hko_south_china_coastal_waters_bulletin.parquet | operational_input_allowed | bool | boolean | 30 | 30 | 0 | 0.0 |  |  | KEEP | live_exact_vintage | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
