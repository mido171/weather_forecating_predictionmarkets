# HKO Daily Climate All Elements

## Dataset identity and use

| Field | Value |
| --- | --- |
| dataset_id | 02_hko_daily_climate_all_elements |
| recommended_db_inclusion | YES-diagnostic schema |
| recommended_layer | diagnostic_physics |
| current_operational_predictor_value_0_100 | 0 |
| diagnostic_research_value_0_100 | 90 |
| future_potential_0_100 | 75 |
| verdict | Store all 21 variables, but do not promote as operational predictors until publication timing is proven. |
| source_tables_or_files | 1 |
| audited_attributes | 17 |
| profiled_rows_across_files | 556399 |
| data_min | 1884-01-01T00:00:00+00:00 |
| data_max | 2026-05-31T00:00:00+00:00 |

## Source tables/files

| source_file | type | rows | attributes | data_min | data_max | db_action | db_layer | model_status | priority | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 02_hko_daily_climate_all_elements/hko_daily_climate_elements.parquet | parquet | 556399 | 17 | 1884-01-01T00:00:00+00:00 | 2026-05-31T00:00:00+00:00 | LOAD_DIAGNOSTIC | diagnostic_physics | BLOCKED_UNTIL_PUBLICATION_PROOF | HIGH | High physical value, but finalized daily table lacks first-publication proof. |

## Attribute nature summary

### Semantic classes

| semantic_class | attribute_count |
| --- | --- |
| boolean | 1 |
| datetime_or_date | 2 |
| hash_identifier | 1 |
| identifier_or_code | 1 |
| nominal_text | 7 |
| numeric | 5 |

### Model roles

| model_role | attribute_count |
| --- | --- |
| DERIVED_CALENDAR | 3 |
| DIAGNOSTIC_METEOROLOGICAL_VALUE | 1 |
| LEAKAGE_POLICY_METADATA | 3 |
| MEASUREMENT_DIMENSION | 3 |
| PROVENANCE_ONLY | 1 |
| PROVENANCE_TIMESTAMP | 1 |
| QUALITY_CONTROL | 3 |
| SOURCE_OR_CATALOG_METADATA | 1 |
| VALID_DATE | 1 |

### Operational status

| operational_status | attribute_count |
| --- | --- |
| BLOCKED_UNTIL_FIRST_PUBLICATION_PROOF | 1 |
| CONDITIONAL_CATEGORICAL_ONLY | 1 |
| DIAGNOSTIC_ALIGNMENT_ONLY | 1 |
| GATING_ONLY | 3 |
| GATING_OR_AUDIT_ONLY | 1 |
| JOIN_AND_PIVOT_KEY | 3 |
| NOT_A_PREDICTOR | 1 |
| NOT_DIRECT | 3 |
| POSSIBLE_MISSINGNESS_OR_TRUST_FEATURE_ONLY | 3 |

### Storage decisions

| storage_decision | attribute_count |
| --- | --- |
| DERIVE_NOT_STORE_DUPLICATE | 3 |
| KEEP | 14 |

## Dataset-specific quality issues

| severity | source_table | attributes | evidence | required_action |
| --- | --- | --- | --- | --- |
| HIGH | hko_daily_climate_elements.parquet | value; parse_issue | 6,916 'Trace' values are currently nonnumeric; 7,389 are missing; 5 invalid dates exist. | Preserve trace_flag separately and apply variable-specific trace policy; quarantine invalid dates. |

## Complete audited attribute dictionary

This table includes every audited attribute found for this dataset. It records the physical source file, source type, inferred semantic class, missingness, storage decision, DB layer, model role, operational status, quality action, usefulness score, observed profile min/max, and the audit rationale.

| source_file | attribute | source_dtype | semantic_class | row_count | non_null_count | null_count | null_pct | profile_min | profile_max | storage_decision | db_layer | model_role | operational_status | quality_action | usefulness_score_0_100 | rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 02_hko_daily_climate_all_elements/hko_daily_climate_elements.parquet | source_id | string | identifier_or_code | 556399 | 556399 | 0 | 0.0 |  |  | KEEP | diagnostic_physics | SOURCE_OR_CATALOG_METADATA | CONDITIONAL_CATEGORICAL_ONLY | NONE | 15 | Useful for lineage/source-era grouping, not meteorology by itself. |
| 02_hko_daily_climate_all_elements/hko_daily_climate_elements.parquet | content_sha256 | string | hash_identifier | 556399 | 556399 | 0 | 0.0 |  |  | KEEP | diagnostic_physics | PROVENANCE_ONLY | NOT_A_PREDICTOR | NONE | 0 | Integrity/deduplication key; essential for provenance, not forecasting. |
| 02_hko_daily_climate_all_elements/hko_daily_climate_elements.parquet | raw_retrieved_at_utc | string | datetime_or_date | 556399 | 556399 | 0 | 0.0 | 2026-06-18T22:51:28.588110+00:00 | 2026-06-18T22:52:35.306583+00:00 | KEEP | diagnostic_physics | PROVENANCE_TIMESTAMP | GATING_OR_AUDIT_ONLY | VALIDATE_TIMEZONE | 95 | Needed to prove retrieval/availability and reproduce vintages; not a weather variable. |
| 02_hko_daily_climate_all_elements/hko_daily_climate_elements.parquet | station_or_domain | string | nominal_text | 556399 | 556399 | 0 | 0.0 |  |  | KEEP | diagnostic_physics | MEASUREMENT_DIMENSION | JOIN_AND_PIVOT_KEY | NORMALIZE_CONTROLLED_VOCABULARY | 70 | Essential to interpret the long-form value. |
| 02_hko_daily_climate_all_elements/hko_daily_climate_elements.parquet | variable | string | nominal_text | 556399 | 556399 | 0 | 0.0 |  |  | KEEP | diagnostic_physics | MEASUREMENT_DIMENSION | JOIN_AND_PIVOT_KEY | NORMALIZE_CONTROLLED_VOCABULARY | 70 | Essential to interpret the long-form value. |
| 02_hko_daily_climate_all_elements/hko_daily_climate_elements.parquet | unit | string | nominal_text | 556399 | 556399 | 0 | 0.0 |  |  | KEEP | diagnostic_physics | MEASUREMENT_DIMENSION | JOIN_AND_PIVOT_KEY | NORMALIZE_CONTROLLED_VOCABULARY | 70 | Essential to interpret the long-form value. |
| 02_hko_daily_climate_all_elements/hko_daily_climate_elements.parquet | local_date | string | datetime_or_date | 556399 | 556399 | 0 | 0.0 | 1884-01-01T00:00:00+00:00 | 2026-05-31T00:00:00+00:00 | KEEP | diagnostic_physics | VALID_DATE | DIAGNOSTIC_ALIGNMENT_ONLY | ENSURE_HKT_LOCAL_DATE | 80 | Needed for causal lagging and joins; same-day values remain blocked. |
| 02_hko_daily_climate_all_elements/hko_daily_climate_elements.parquet | year | int64 | numeric | 556399 | 556399 | 0 | 0.0 | 1884.0 | 2026.0 | DERIVE_NOT_STORE_DUPLICATE | diagnostic_physics | DERIVED_CALENDAR | NOT_DIRECT | NONE | 5 | Redundant with local_date. |
| 02_hko_daily_climate_all_elements/hko_daily_climate_elements.parquet | month | int64 | numeric | 556399 | 556399 | 0 | 0.0 | 1.0 | 12.0 | DERIVE_NOT_STORE_DUPLICATE | diagnostic_physics | DERIVED_CALENDAR | NOT_DIRECT | NONE | 5 | Redundant with local_date. |
| 02_hko_daily_climate_all_elements/hko_daily_climate_elements.parquet | day | int64 | numeric | 556399 | 556399 | 0 | 0.0 | 1.0 | 31.0 | DERIVE_NOT_STORE_DUPLICATE | diagnostic_physics | DERIVED_CALENDAR | NOT_DIRECT | NONE | 5 | Redundant with local_date. |
| 02_hko_daily_climate_all_elements/hko_daily_climate_elements.parquet | value | double | numeric | 556399 | 542089 | 14310 | 0.025719 | -8.6 | 34029.0 | KEEP | diagnostic_physics | DIAGNOSTIC_METEOROLOGICAL_VALUE | BLOCKED_UNTIL_FIRST_PUBLICATION_PROOF | VARIABLE_SPECIFIC_RANGE_AND_TRACE_CLEANING | 85 | High physical value conditional on variable/station, but current finalized table is not operationally timestamp-proven. |
| 02_hko_daily_climate_all_elements/hko_daily_climate_elements.parquet | value_precision | double | numeric | 556399 | 542089 | 14310 | 0.025719 | 0.01 | 1.0 | KEEP | diagnostic_physics | QUALITY_CONTROL | POSSIBLE_MISSINGNESS_OR_TRUST_FEATURE_ONLY | STANDARDIZE_ENUMS_AND_QUARANTINE_BAD_ROWS | 35 | Quality/trust metadata; useful for filtering and possibly uncertainty, not primary physical signal. |
| 02_hko_daily_climate_all_elements/hko_daily_climate_elements.parquet | completeness | string | nominal_text | 556399 | 549005 | 7394 | 0.013289 |  |  | KEEP | diagnostic_physics | QUALITY_CONTROL | POSSIBLE_MISSINGNESS_OR_TRUST_FEATURE_ONLY | STANDARDIZE_ENUMS_AND_QUARANTINE_BAD_ROWS | 35 | Quality/trust metadata; useful for filtering and possibly uncertainty, not primary physical signal. |
| 02_hko_daily_climate_all_elements/hko_daily_climate_elements.parquet | parse_issue | string | nominal_text | 556399 | 14310 | 542089 | 0.974281 |  |  | KEEP | diagnostic_physics | QUALITY_CONTROL | POSSIBLE_MISSINGNESS_OR_TRUST_FEATURE_ONLY | STANDARDIZE_ENUMS_AND_QUARANTINE_BAD_ROWS | 35 | Quality/trust metadata; useful for filtering and possibly uncertainty, not primary physical signal. |
| 02_hko_daily_climate_all_elements/hko_daily_climate_elements.parquet | availability_tier | string | nominal_text | 556399 | 556399 | 0 | 0.0 |  |  | KEEP | diagnostic_physics | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 02_hko_daily_climate_all_elements/hko_daily_climate_elements.parquet | operational_input_allowed | bool | boolean | 556399 | 556399 | 0 | 0.0 |  |  | KEEP | diagnostic_physics | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 02_hko_daily_climate_all_elements/hko_daily_climate_elements.parquet | source_time_policy | string | nominal_text | 556399 | 556399 | 0 | 0.0 |  |  | KEEP | diagnostic_physics | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
