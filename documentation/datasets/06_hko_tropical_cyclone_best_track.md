# HKO Tropical Cyclone Best Track

## Dataset identity and use

| Field | Value |
| --- | --- |
| dataset_id | 06_hko_tropical_cyclone_best_track |
| recommended_db_inclusion | YES-diagnostic only |
| recommended_layer | diagnostic_regime_labels |
| current_operational_predictor_value_0_100 | 0 |
| diagnostic_research_value_0_100 | 75 |
| future_potential_0_100 | 65 |
| verdict | Useful for TC mechanism labels and proxy discovery; retrospective best track is forbidden live. |
| source_tables_or_files | 1 |
| audited_attributes | 16 |
| profiled_rows_across_files | 26189 |
| data_min | 1985-01-06T06:00:00+00:00 |
| data_max | 2024-12-25T06:00:00+00:00 |

## Source tables/files

| source_file | type | rows | attributes | data_min | data_max | db_action | db_layer | model_status | priority | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 06_hko_tropical_cyclone_best_track/hko_tropical_cyclone_best_track.parquet | parquet | 26189 | 16 | 1985-01-06T06:00:00+00:00 | 2024-12-25T06:00:00+00:00 | LOAD_DIAGNOSTIC | diagnostic_regime_labels | RETROSPECTIVE_ONLY | MEDIUM | Use for mechanism discovery/autopsy, never as operational predictor. |

## Attribute nature summary

### Semantic classes

| semantic_class | attribute_count |
| --- | --- |
| boolean | 1 |
| datetime_or_date | 3 |
| free_natural_text | 1 |
| hash_identifier | 1 |
| identifier_or_code | 3 |
| nominal_text | 3 |
| numeric | 4 |

### Model roles

| model_role | attribute_count |
| --- | --- |
| LEAKAGE_POLICY_METADATA | 3 |
| PROVENANCE_ONLY | 1 |
| PROVENANCE_TIMESTAMP | 1 |
| RETROSPECTIVE_VALID_TIME | 2 |
| SOURCE_OR_CATALOG_METADATA | 1 |
| TC_IDENTIFIER_OR_PROVENANCE | 2 |
| TC_MECHANISM_SIGNAL | 6 |

### Operational status

| operational_status | attribute_count |
| --- | --- |
| CONDITIONAL_CATEGORICAL_ONLY | 1 |
| DIAGNOSTIC_ONLY | 2 |
| GATING_ONLY | 3 |
| GATING_OR_AUDIT_ONLY | 1 |
| NOT_A_PREDICTOR | 1 |
| NOT_DIRECT | 2 |
| RETROSPECTIVE_DIAGNOSTIC_ONLY | 6 |

### Storage decisions

| storage_decision | attribute_count |
| --- | --- |
| KEEP | 16 |

## Dataset-specific quality issues

No dataset-specific audit issue is recorded in the 2026-06-23 audit bundle.

## Complete audited attribute dictionary

This table includes every audited attribute found for this dataset. It records the physical source file, source type, inferred semantic class, missingness, storage decision, DB layer, model role, operational status, quality action, usefulness score, observed profile min/max, and the audit rationale.

| source_file | attribute | source_dtype | semantic_class | row_count | non_null_count | null_count | null_pct | profile_min | profile_max | storage_decision | db_layer | model_role | operational_status | quality_action | usefulness_score_0_100 | rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 06_hko_tropical_cyclone_best_track/hko_tropical_cyclone_best_track.parquet | source_id | string | identifier_or_code | 26189 | 26189 | 0 | 0.0 |  |  | KEEP | diagnostic_regime_labels | SOURCE_OR_CATALOG_METADATA | CONDITIONAL_CATEGORICAL_ONLY | NONE | 15 | Useful for lineage/source-era grouping, not meteorology by itself. |
| 06_hko_tropical_cyclone_best_track/hko_tropical_cyclone_best_track.parquet | content_sha256 | string | hash_identifier | 26189 | 26189 | 0 | 0.0 |  |  | KEEP | diagnostic_regime_labels | PROVENANCE_ONLY | NOT_A_PREDICTOR | NONE | 0 | Integrity/deduplication key; essential for provenance, not forecasting. |
| 06_hko_tropical_cyclone_best_track/hko_tropical_cyclone_best_track.parquet | raw_retrieved_at_utc | string | datetime_or_date | 26189 | 26189 | 0 | 0.0 | 2026-06-19T04:51:07.648318+00:00 | 2026-06-19T04:52:38.219460+00:00 | KEEP | diagnostic_regime_labels | PROVENANCE_TIMESTAMP | GATING_OR_AUDIT_ONLY | VALIDATE_TIMEZONE | 95 | Needed to prove retrieval/availability and reproduce vintages; not a weather variable. |
| 06_hko_tropical_cyclone_best_track/hko_tropical_cyclone_best_track.parquet | cyclone_name | string | nominal_text | 26189 | 26189 | 0 | 0.0 |  |  | KEEP | diagnostic_regime_labels | TC_MECHANISM_SIGNAL | RETROSPECTIVE_DIAGNOSTIC_ONLY | NORMALIZE_LONGITUDE_AND_DERIVE_DISTANCE_MOTION_QUADRANT | 70 | Useful to discover TC-related Tmax mechanisms and build safe operational proxies. |
| 06_hko_tropical_cyclone_best_track/hko_tropical_cyclone_best_track.parquet | valid_at_utc | string | datetime_or_date | 26189 | 26189 | 0 | 0.0 | 1985-01-06T06:00:00+00:00 | 2024-12-25T06:00:00+00:00 | KEEP | diagnostic_regime_labels | RETROSPECTIVE_VALID_TIME | DIAGNOSTIC_ONLY | VERIFY_TIMEZONE | 60 | Aligns post-analyzed cyclone state to target dates; not operational availability. |
| 06_hko_tropical_cyclone_best_track/hko_tropical_cyclone_best_track.parquet | valid_at_hkt | string | datetime_or_date | 26189 | 26189 | 0 | 0.0 | 1985-01-06T06:00:00+00:00 | 2024-12-25T06:00:00+00:00 | KEEP | diagnostic_regime_labels | RETROSPECTIVE_VALID_TIME | DIAGNOSTIC_ONLY | VERIFY_TIMEZONE | 60 | Aligns post-analyzed cyclone state to target dates; not operational availability. |
| 06_hko_tropical_cyclone_best_track/hko_tropical_cyclone_best_track.parquet | intensity | string | nominal_text | 26189 | 26189 | 0 | 0.0 |  |  | KEEP | diagnostic_regime_labels | TC_MECHANISM_SIGNAL | RETROSPECTIVE_DIAGNOSTIC_ONLY | NORMALIZE_LONGITUDE_AND_DERIVE_DISTANCE_MOTION_QUADRANT | 70 | Useful to discover TC-related Tmax mechanisms and build safe operational proxies. |
| 06_hko_tropical_cyclone_best_track/hko_tropical_cyclone_best_track.parquet | latitude | double | numeric | 26189 | 26189 | 0 | 0.0 | 1.5 | 48.5 | KEEP | diagnostic_regime_labels | TC_MECHANISM_SIGNAL | RETROSPECTIVE_DIAGNOSTIC_ONLY | NORMALIZE_LONGITUDE_AND_DERIVE_DISTANCE_MOTION_QUADRANT | 70 | Useful to discover TC-related Tmax mechanisms and build safe operational proxies. |
| 06_hko_tropical_cyclone_best_track/hko_tropical_cyclone_best_track.parquet | longitude | double | numeric | 26189 | 26189 | 0 | 0.0 | 78.0 | 188.5 | KEEP | diagnostic_regime_labels | TC_MECHANISM_SIGNAL | RETROSPECTIVE_DIAGNOSTIC_ONLY | NORMALIZE_LONGITUDE_AND_DERIVE_DISTANCE_MOTION_QUADRANT | 70 | Useful to discover TC-related Tmax mechanisms and build safe operational proxies. |
| 06_hko_tropical_cyclone_best_track/hko_tropical_cyclone_best_track.parquet | minimum_central_pressure_hpa | double | numeric | 26189 | 26189 | 0 | 0.0 | 890.0 | 1010.0 | KEEP | diagnostic_regime_labels | TC_MECHANISM_SIGNAL | RETROSPECTIVE_DIAGNOSTIC_ONLY | NORMALIZE_LONGITUDE_AND_DERIVE_DISTANCE_MOTION_QUADRANT | 70 | Useful to discover TC-related Tmax mechanisms and build safe operational proxies. |
| 06_hko_tropical_cyclone_best_track/hko_tropical_cyclone_best_track.parquet | maximum_surface_wind_kt | double | numeric | 26189 | 26189 | 0 | 0.0 | 20.0 | 155.0 | KEEP | diagnostic_regime_labels | TC_MECHANISM_SIGNAL | RETROSPECTIVE_DIAGNOSTIC_ONLY | NORMALIZE_LONGITUDE_AND_DERIVE_DISTANCE_MOTION_QUADRANT | 70 | Useful to discover TC-related Tmax mechanisms and build safe operational proxies. |
| 06_hko_tropical_cyclone_best_track/hko_tropical_cyclone_best_track.parquet | jma_code | string | identifier_or_code | 26189 | 26189 | 0 | 0.0 |  |  | KEEP | diagnostic_regime_labels | TC_IDENTIFIER_OR_PROVENANCE | NOT_DIRECT | NONE | 10 | Retain for event identity and provenance. |
| 06_hko_tropical_cyclone_best_track/hko_tropical_cyclone_best_track.parquet | hko_code | string | identifier_or_code | 26189 | 26189 | 0 | 0.0 |  |  | KEEP | diagnostic_regime_labels | TC_IDENTIFIER_OR_PROVENANCE | NOT_DIRECT | NONE | 10 | Retain for event identity and provenance. |
| 06_hko_tropical_cyclone_best_track/hko_tropical_cyclone_best_track.parquet | availability_tier | string | nominal_text | 26189 | 26189 | 0 | 0.0 |  |  | KEEP | diagnostic_regime_labels | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 06_hko_tropical_cyclone_best_track/hko_tropical_cyclone_best_track.parquet | operational_input_allowed | bool | boolean | 26189 | 26189 | 0 | 0.0 |  |  | KEEP | diagnostic_regime_labels | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 06_hko_tropical_cyclone_best_track/hko_tropical_cyclone_best_track.parquet | source_time_policy | string | free_natural_text | 26189 | 26189 | 0 | 0.0 |  |  | KEEP | diagnostic_regime_labels | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
