# HKO Daily Tmax Target

## Dataset identity and use

| Field | Value |
| --- | --- |
| dataset_id | 01_hko_daily_tmax_target |
| recommended_db_inclusion | YES-canonical labels plus audit raw |
| recommended_layer | label_core / raw_audit |
| current_operational_predictor_value_0_100 | 95 |
| diagnostic_research_value_0_100 | 100 |
| future_potential_0_100 | 100 |
| verdict | Use canonical labels; derive only prior-day/older target-memory features. Never use target T. |
| source_tables_or_files | 2 |
| audited_attributes | 22 |
| profiled_rows_across_files | 99087 |
| data_min | 1884-01-01T00:00:00+00:00 |
| data_max | 2026-06-17T00:00:00+00:00 |

## Source tables/files

| source_file | type | rows | attributes | data_min | data_max | db_action | db_layer | model_status | priority | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01_hko_daily_tmax_target/hko_daily_extract_tmax_payload_rows.parquet | parquet | 49628 | 14 | 1884-01-01T00:00:00+00:00 | 2026-06-17T00:00:00+00:00 | LOAD_PROVENANCE | raw_audit | LABEL_AUDIT_ONLY | HIGH | Retain parser/source audit; do not use as predictor. |
| 01_hko_daily_tmax_target/hko_daily_tmax_target_labels.parquet | parquet | 49459 | 8 | 1884-01-01T00:00:00+00:00 | 2026-05-31T00:00:00+00:00 | LOAD_CANONICAL | label_core | LABEL_ONLY | CRITICAL | Canonical target label table; never expose target T as predictor. |

## Attribute nature summary

### Semantic classes

| semantic_class | attribute_count |
| --- | --- |
| boolean | 1 |
| datetime_or_date | 4 |
| free_natural_text | 1 |
| hash_identifier | 2 |
| identifier_or_code | 2 |
| nominal_text | 6 |
| numeric | 6 |

### Model roles

| model_role | attribute_count |
| --- | --- |
| DERIVED_CALENDAR | 3 |
| LABEL_ONLY | 2 |
| LEAKAGE_POLICY_METADATA | 5 |
| PRIMARY_KEY_DATE | 2 |
| PROVENANCE_ONLY | 2 |
| PROVENANCE_TIMESTAMP | 2 |
| QUALITY_CONTROL | 3 |
| SOURCE_OR_CATALOG_METADATA | 2 |
| STATION_IDENTITY | 1 |

### Operational status

| operational_status | attribute_count |
| --- | --- |
| CONDITIONAL_CATEGORICAL_ONLY | 2 |
| FORBIDDEN_AT_INFERENCE_FOR_TARGET_T | 2 |
| GATING_ONLY | 5 |
| GATING_OR_AUDIT_ONLY | 2 |
| JOIN_ONLY | 3 |
| NOT_A_PREDICTOR | 2 |
| POSSIBLE_MISSINGNESS_OR_TRUST_FEATURE_ONLY | 3 |
| SAFE_IF_DERIVED_FROM_TARGET_DATE | 3 |

### Storage decisions

| storage_decision | attribute_count |
| --- | --- |
| DERIVE_NOT_STORE_DUPLICATE | 3 |
| KEEP | 19 |

## Dataset-specific quality issues

| severity | source_table | attributes | evidence | required_action |
| --- | --- | --- | --- | --- |
| HIGH | two tables | local_date; target values | Payload table has 49,628 rows versus 49,459 canonical labels and includes monthly/yearly sources plus one parse failure. | Use labels table as canonical. Deduplicate/reconcile payload dates and source overlap only for audit. |

## Complete audited attribute dictionary

This table includes every audited attribute found for this dataset. It records the physical source file, source type, inferred semantic class, missingness, storage decision, DB layer, model role, operational status, quality action, usefulness score, observed profile min/max, and the audit rationale.

| source_file | attribute | source_dtype | semantic_class | row_count | non_null_count | null_count | null_pct | profile_min | profile_max | storage_decision | db_layer | model_role | operational_status | quality_action | usefulness_score_0_100 | rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01_hko_daily_tmax_target/hko_daily_extract_tmax_payload_rows.parquet | source_id | string | identifier_or_code | 49628 | 49628 | 0 | 0.0 |  |  | KEEP | raw_audit | SOURCE_OR_CATALOG_METADATA | CONDITIONAL_CATEGORICAL_ONLY | NONE | 15 | Useful for lineage/source-era grouping, not meteorology by itself. |
| 01_hko_daily_tmax_target/hko_daily_extract_tmax_payload_rows.parquet | content_sha256 | string | hash_identifier | 49628 | 49628 | 0 | 0.0 |  |  | KEEP | raw_audit | PROVENANCE_ONLY | NOT_A_PREDICTOR | NONE | 0 | Integrity/deduplication key; essential for provenance, not forecasting. |
| 01_hko_daily_tmax_target/hko_daily_extract_tmax_payload_rows.parquet | raw_retrieved_at_utc | string | datetime_or_date | 49628 | 49628 | 0 | 0.0 | 2026-06-18T22:52:48.528263+00:00 | 2026-06-18T22:58:13.112613+00:00 | KEEP | raw_audit | PROVENANCE_TIMESTAMP | GATING_OR_AUDIT_ONLY | VALIDATE_TIMEZONE | 95 | Needed to prove retrieval/availability and reproduce vintages; not a weather variable. |
| 01_hko_daily_tmax_target/hko_daily_extract_tmax_payload_rows.parquet | local_date | string | datetime_or_date | 49628 | 49627 | 1 | 2e-05 | 1884-01-01T00:00:00+00:00 | 2026-06-17T00:00:00+00:00 | KEEP | raw_audit | PRIMARY_KEY_DATE | JOIN_ONLY | ENSURE_HKT_LOCAL_DATE | 100 | Primary label date and time-series key. |
| 01_hko_daily_tmax_target/hko_daily_extract_tmax_payload_rows.parquet | year | int64 | numeric | 49628 | 49628 | 0 | 0.0 | 1884.0 | 2026.0 | DERIVE_NOT_STORE_DUPLICATE | raw_audit | DERIVED_CALENDAR | SAFE_IF_DERIVED_FROM_TARGET_DATE | NONE | 10 | Redundant with local_date; generate in feature views. |
| 01_hko_daily_tmax_target/hko_daily_extract_tmax_payload_rows.parquet | month | double | numeric | 49628 | 49627 | 1 | 2e-05 | 1.0 | 12.0 | DERIVE_NOT_STORE_DUPLICATE | raw_audit | DERIVED_CALENDAR | SAFE_IF_DERIVED_FROM_TARGET_DATE | NONE | 10 | Redundant with local_date; generate in feature views. |
| 01_hko_daily_tmax_target/hko_daily_extract_tmax_payload_rows.parquet | day | double | numeric | 49628 | 49627 | 1 | 2e-05 | 1.0 | 31.0 | DERIVE_NOT_STORE_DUPLICATE | raw_audit | DERIVED_CALENDAR | SAFE_IF_DERIVED_FROM_TARGET_DATE | NONE | 10 | Redundant with local_date; generate in feature views. |
| 01_hko_daily_tmax_target/hko_daily_extract_tmax_payload_rows.parquet | absolute_daily_max_c | double | numeric | 49628 | 49627 | 1 | 2e-05 | 3.2 | 36.6 | KEEP | raw_audit | LABEL_ONLY | FORBIDDEN_AT_INFERENCE_FOR_TARGET_T | RANGE_CHECK_3_TO_40C | 100 | Canonical outcome; use only after T is complete and for causal lag creation. |
| 01_hko_daily_tmax_target/hko_daily_extract_tmax_payload_rows.parquet | value_precision | double | numeric | 49628 | 49627 | 1 | 2e-05 | 0.1 | 0.1 | KEEP | raw_audit | QUALITY_CONTROL | POSSIBLE_MISSINGNESS_OR_TRUST_FEATURE_ONLY | STANDARDIZE_ENUMS_AND_QUARANTINE_BAD_ROWS | 35 | Quality/trust metadata; useful for filtering and possibly uncertainty, not primary physical signal. |
| 01_hko_daily_tmax_target/hko_daily_extract_tmax_payload_rows.parquet | completeness | string | nominal_text | 49628 | 49627 | 1 | 2e-05 |  |  | KEEP | raw_audit | QUALITY_CONTROL | POSSIBLE_MISSINGNESS_OR_TRUST_FEATURE_ONLY | STANDARDIZE_ENUMS_AND_QUARANTINE_BAD_ROWS | 35 | Quality/trust metadata; useful for filtering and possibly uncertainty, not primary physical signal. |
| 01_hko_daily_tmax_target/hko_daily_extract_tmax_payload_rows.parquet | parse_issue | string | nominal_text | 49628 | 1 | 49627 | 0.99998 |  |  | KEEP | raw_audit | QUALITY_CONTROL | POSSIBLE_MISSINGNESS_OR_TRUST_FEATURE_ONLY | STANDARDIZE_ENUMS_AND_QUARANTINE_BAD_ROWS | 35 | Quality/trust metadata; useful for filtering and possibly uncertainty, not primary physical signal. |
| 01_hko_daily_tmax_target/hko_daily_extract_tmax_payload_rows.parquet | availability_tier | string | nominal_text | 49628 | 49627 | 1 | 2e-05 |  |  | KEEP | raw_audit | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 01_hko_daily_tmax_target/hko_daily_extract_tmax_payload_rows.parquet | operational_input_allowed | bool | nominal_text | 49628 | 49627 | 1 | 2e-05 |  |  | KEEP | raw_audit | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 01_hko_daily_tmax_target/hko_daily_extract_tmax_payload_rows.parquet | source_time_policy | string | free_natural_text | 49628 | 49627 | 1 | 2e-05 |  |  | KEEP | raw_audit | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 01_hko_daily_tmax_target/hko_daily_tmax_target_labels.parquet | local_date | string | datetime_or_date | 49459 | 49459 | 0 | 0.0 | 1884-01-01T00:00:00+00:00 | 2026-05-31T00:00:00+00:00 | KEEP | label_core | PRIMARY_KEY_DATE | JOIN_ONLY | ENSURE_HKT_LOCAL_DATE | 100 | Primary label date and time-series key. |
| 01_hko_daily_tmax_target/hko_daily_tmax_target_labels.parquet | target_tmax_c | double | numeric | 49459 | 49459 | 0 | 0.0 | 3.2 | 36.6 | KEEP | label_core | LABEL_ONLY | FORBIDDEN_AT_INFERENCE_FOR_TARGET_T | RANGE_CHECK_3_TO_40C | 100 | Canonical outcome; use only after T is complete and for causal lag creation. |
| 01_hko_daily_tmax_target/hko_daily_tmax_target_labels.parquet | target_station | string | nominal_text | 49459 | 49459 | 0 | 0.0 |  |  | KEEP | label_core | STATION_IDENTITY | JOIN_ONLY | NORMALIZE_TO_STATION_DIMENSION | 20 | Target identity; not variable because only one station. |
| 01_hko_daily_tmax_target/hko_daily_tmax_target_labels.parquet | target_source_id | string | identifier_or_code | 49459 | 49459 | 0 | 0.0 |  |  | KEEP | label_core | SOURCE_OR_CATALOG_METADATA | CONDITIONAL_CATEGORICAL_ONLY | NONE | 15 | Useful for lineage/source-era grouping, not meteorology by itself. |
| 01_hko_daily_tmax_target/hko_daily_tmax_target_labels.parquet | content_sha256 | string | hash_identifier | 49459 | 49459 | 0 | 0.0 |  |  | KEEP | label_core | PROVENANCE_ONLY | NOT_A_PREDICTOR | NONE | 0 | Integrity/deduplication key; essential for provenance, not forecasting. |
| 01_hko_daily_tmax_target/hko_daily_tmax_target_labels.parquet | raw_retrieved_at_utc | string | datetime_or_date | 49459 | 49459 | 0 | 0.0 | 2026-06-18T22:51:53.536120+00:00 | 2026-06-18T22:51:53.536120+00:00 | KEEP | label_core | PROVENANCE_TIMESTAMP | GATING_OR_AUDIT_ONLY | VALIDATE_TIMEZONE | 95 | Needed to prove retrieval/availability and reproduce vintages; not a weather variable. |
| 01_hko_daily_tmax_target/hko_daily_tmax_target_labels.parquet | availability_tier | string | nominal_text | 49459 | 49459 | 0 | 0.0 |  |  | KEEP | label_core | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 01_hko_daily_tmax_target/hko_daily_tmax_target_labels.parquet | operational_input_allowed | bool | boolean | 49459 | 49459 | 0 | 0.0 |  |  | KEEP | label_core | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
