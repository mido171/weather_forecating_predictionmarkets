# HKO Radar Satellite Lightning Nowcast

## Dataset identity and use

| Field | Value |
| --- | --- |
| dataset_id | 07_hko_radar_satellite_lightning_nowcast |
| recommended_db_inclusion | YES-live exact-vintage collection/object catalog |
| recommended_layer | live_exact_vintage / live_object_catalog |
| current_operational_predictor_value_0_100 | 5 |
| diagnostic_research_value_0_100 | 45 |
| future_potential_0_100 | 92 |
| verdict | Potentially valuable cloud/rain/convection layer, but present history is only days and imagery is not feature-extracted. |
| source_tables_or_files | 4 |
| audited_attributes | 40 |
| profiled_rows_across_files | 4812 |
| data_min | 2026-06-17T00:15:00+00:00 |
| data_max | 2026-06-20T09:36:00+00:00 |

## Source tables/files

| source_file | type | rows | attributes | data_min | data_max | db_action | db_layer | model_status | priority | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 07_hko_radar_satellite_lightning_nowcast/hko_gridded_rainfall_nowcast_summary.parquet | parquet | 41 | 13 | 2026-06-19T04:36:00+00:00 | 2026-06-20T09:36:00+00:00 | LOAD_LIVE_COLLECTION | live_exact_vintage | FUTURE_AFTER_HISTORY | HIGH | Exact-vintage but current sample is far too short for training; keep collecting. |
| 07_hko_radar_satellite_lightning_nowcast/hko_lightning_counts_latest.parquet | parquet | 102 | 9 |  |  | LOAD_LIVE_COLLECTION | live_exact_vintage | FUTURE_AFTER_HISTORY | HIGH | Exact-vintage but current sample is far too short for training; keep collecting. |
| 07_hko_radar_satellite_lightning_nowcast/hko_radar_manifest_frames.parquet | parquet | 80 | 8 | 2026-06-19T02:54:00+00:00 | 2026-06-19T04:48:00+00:00 | REGISTER_OBJECT_METADATA | live_object_catalog | FUTURE_AFTER_HISTORY | HIGH | Exact-vintage but current sample is far too short for training; keep collecting. Binary imagery remains in object storage; derive pixel/geospatial features later. |
| 07_hko_radar_satellite_lightning_nowcast/hko_satellite_image_inventory.parquet | parquet | 4589 | 10 | 2026-06-17T00:15:00+00:00 | 2026-06-19T23:30:00+00:00 | REGISTER_OBJECT_METADATA | live_object_catalog | FUTURE_AFTER_HISTORY | HIGH | Exact-vintage but current sample is far too short for training; keep collecting. Binary imagery remains in object storage; derive pixel/geospatial features later. |

## Attribute nature summary

### Semantic classes

| semantic_class | attribute_count |
| --- | --- |
| boolean | 4 |
| datetime_or_date | 8 |
| free_natural_text | 1 |
| hash_identifier | 4 |
| high_cardinality_text | 1 |
| identifier_or_code | 4 |
| nominal_text | 8 |
| numeric | 8 |
| path_or_file_reference | 2 |

### Model roles

| model_role | attribute_count |
| --- | --- |
| CONVECTION_SIGNAL | 1 |
| FILE_OR_GRID_QC | 2 |
| LEAKAGE_POLICY_METADATA | 9 |
| LIVE_VALID_OR_ISSUE_TIME | 5 |
| NOWCAST_PRECIPITATION_STATE | 5 |
| OBJECT_OR_SOURCE_REFERENCE | 3 |
| PROVENANCE_ONLY | 4 |
| PROVENANCE_TIMESTAMP | 4 |
| SOURCE_OR_CATALOG_METADATA | 4 |
| SPATIAL_OR_TYPE_DIMENSION | 3 |

### Operational status

| operational_status | attribute_count |
| --- | --- |
| CONDITIONAL_CATEGORICAL_ONLY | 4 |
| FUTURE_LIVE_CONTEXT | 3 |
| FUTURE_LIVE_PREDICTOR | 11 |
| GATING_ONLY | 9 |
| GATING_OR_AUDIT_ONLY | 4 |
| NOT_A_PREDICTOR | 7 |
| QUALITY_ONLY | 2 |

### Storage decisions

| storage_decision | attribute_count |
| --- | --- |
| KEEP | 37 |
| KEEP_REFERENCE_ONLY | 3 |

## Dataset-specific quality issues

| severity | source_table | attributes | evidence | required_action |
| --- | --- | --- | --- | --- |
| HIGH | hko_gridded_rainfall_nowcast_summary.parquet | rainfall_max_mm | Maximum is 544.88 mm while sample covers only 41 snapshots. | Verify accumulation interval, grid fill values and units before use. |
| HIGH | hko_satellite_image_inventory.parquet | image_time_hkt | 132 of 4,589 non-null image times fail datetime parsing. | Quarantine malformed entries and distinguish page/JS manifests from actual images. |
| MEDIUM | hko_lightning_counts_latest.parquet | lightning_count | All 102 values are zero. | Retain feed but current sample has no information gain; continue collection and monitor variance. |

## Complete audited attribute dictionary

This table includes every audited attribute found for this dataset. It records the physical source file, source type, inferred semantic class, missingness, storage decision, DB layer, model role, operational status, quality action, usefulness score, observed profile min/max, and the audit rationale.

| source_file | attribute | source_dtype | semantic_class | row_count | non_null_count | null_count | null_pct | profile_min | profile_max | storage_decision | db_layer | model_role | operational_status | quality_action | usefulness_score_0_100 | rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 07_hko_radar_satellite_lightning_nowcast/hko_gridded_rainfall_nowcast_summary.parquet | source_id | string | identifier_or_code | 41 | 41 | 0 | 0.0 |  |  | KEEP | live_exact_vintage | SOURCE_OR_CATALOG_METADATA | CONDITIONAL_CATEGORICAL_ONLY | NONE | 15 | Useful for lineage/source-era grouping, not meteorology by itself. |
| 07_hko_radar_satellite_lightning_nowcast/hko_gridded_rainfall_nowcast_summary.parquet | content_sha256 | string | hash_identifier | 41 | 41 | 0 | 0.0 |  |  | KEEP | live_exact_vintage | PROVENANCE_ONLY | NOT_A_PREDICTOR | NONE | 0 | Integrity/deduplication key; essential for provenance, not forecasting. |
| 07_hko_radar_satellite_lightning_nowcast/hko_gridded_rainfall_nowcast_summary.parquet | raw_retrieved_at_utc | string | datetime_or_date | 41 | 41 | 0 | 0.0 | 2026-06-19T04:50:52.482529+00:00 | 2026-06-20T07:56:22.144891+00:00 | KEEP | live_exact_vintage | PROVENANCE_TIMESTAMP | GATING_OR_AUDIT_ONLY | VALIDATE_TIMEZONE | 95 | Needed to prove retrieval/availability and reproduce vintages; not a weather variable. |
| 07_hko_radar_satellite_lightning_nowcast/hko_gridded_rainfall_nowcast_summary.parquet | issue_time_hkt | string | datetime_or_date | 41 | 41 | 0 | 0.0 | 2026-06-19T04:36:00+00:00 | 2026-06-20T07:36:00+00:00 | KEEP | live_exact_vintage | LIVE_VALID_OR_ISSUE_TIME | FUTURE_LIVE_PREDICTOR | PARSE_TO_TIMESTAMPTZ_AND_ENFORCE_CUTOFF | 90 | Exact-vintage temporal key; current history is too short. |
| 07_hko_radar_satellite_lightning_nowcast/hko_gridded_rainfall_nowcast_summary.parquet | ending_time_hkt | string | datetime_or_date | 41 | 41 | 0 | 0.0 | 2026-06-19T06:36:00+00:00 | 2026-06-20T09:36:00+00:00 | KEEP | live_exact_vintage | LIVE_VALID_OR_ISSUE_TIME | FUTURE_LIVE_PREDICTOR | PARSE_TO_TIMESTAMPTZ_AND_ENFORCE_CUTOFF | 90 | Exact-vintage temporal key; current history is too short. |
| 07_hko_radar_satellite_lightning_nowcast/hko_gridded_rainfall_nowcast_summary.parquet | grid_cell_count | int64 | numeric | 41 | 41 | 0 | 0.0 | 58564.0 | 58564.0 | KEEP | live_exact_vintage | FILE_OR_GRID_QC | QUALITY_ONLY | CHECK_CONSTANTS_AND_FILE_COMPLETENESS | 15 | Useful for integrity, not weather prediction. |
| 07_hko_radar_satellite_lightning_nowcast/hko_gridded_rainfall_nowcast_summary.parquet | rainfall_mean_mm | double | numeric | 41 | 41 | 0 | 0.0 | 0.00792756642305845 | 0.2087000546410883 | KEEP | live_exact_vintage | NOWCAST_PRECIPITATION_STATE | FUTURE_LIVE_PREDICTOR | VERIFY_UNITS_FILL_VALUES_AND_SPATIAL_DOMAIN | 80 | Potential cloud/rain suppression signal, but only 41 rows and rainfall maximum is suspicious. |
| 07_hko_radar_satellite_lightning_nowcast/hko_gridded_rainfall_nowcast_summary.parquet | rainfall_max_mm | double | numeric | 41 | 41 | 0 | 0.0 | 6.03 | 544.88 | KEEP | live_exact_vintage | NOWCAST_PRECIPITATION_STATE | FUTURE_LIVE_PREDICTOR | VERIFY_UNITS_FILL_VALUES_AND_SPATIAL_DOMAIN | 80 | Potential cloud/rain suppression signal, but only 41 rows and rainfall maximum is suspicious. |
| 07_hko_radar_satellite_lightning_nowcast/hko_gridded_rainfall_nowcast_summary.parquet | rainfall_p95_mm | double | numeric | 41 | 41 | 0 | 0.0 | 0.0 | 0.22 | KEEP | live_exact_vintage | NOWCAST_PRECIPITATION_STATE | FUTURE_LIVE_PREDICTOR | VERIFY_UNITS_FILL_VALUES_AND_SPATIAL_DOMAIN | 80 | Potential cloud/rain suppression signal, but only 41 rows and rainfall maximum is suspicious. |
| 07_hko_radar_satellite_lightning_nowcast/hko_gridded_rainfall_nowcast_summary.parquet | rain_area_fraction_gt_0mm | double | numeric | 41 | 41 | 0 | 0.0 | 0.030974660200805956 | 0.1576907315074107 | KEEP | live_exact_vintage | NOWCAST_PRECIPITATION_STATE | FUTURE_LIVE_PREDICTOR | VERIFY_UNITS_FILL_VALUES_AND_SPATIAL_DOMAIN | 80 | Potential cloud/rain suppression signal, but only 41 rows and rainfall maximum is suspicious. |
| 07_hko_radar_satellite_lightning_nowcast/hko_gridded_rainfall_nowcast_summary.parquet | rain_area_fraction_ge_1mm | double | numeric | 41 | 41 | 0 | 0.0 | 0.0014855542654190288 | 0.02593743596748856 | KEEP | live_exact_vintage | NOWCAST_PRECIPITATION_STATE | FUTURE_LIVE_PREDICTOR | VERIFY_UNITS_FILL_VALUES_AND_SPATIAL_DOMAIN | 80 | Potential cloud/rain suppression signal, but only 41 rows and rainfall maximum is suspicious. |
| 07_hko_radar_satellite_lightning_nowcast/hko_gridded_rainfall_nowcast_summary.parquet | availability_tier | string | nominal_text | 41 | 41 | 0 | 0.0 |  |  | KEEP | live_exact_vintage | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 07_hko_radar_satellite_lightning_nowcast/hko_gridded_rainfall_nowcast_summary.parquet | operational_input_allowed | bool | boolean | 41 | 41 | 0 | 0.0 |  |  | KEEP | live_exact_vintage | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 07_hko_radar_satellite_lightning_nowcast/hko_lightning_counts_latest.parquet | source_id | string | identifier_or_code | 102 | 102 | 0 | 0.0 |  |  | KEEP | live_exact_vintage | SOURCE_OR_CATALOG_METADATA | CONDITIONAL_CATEGORICAL_ONLY | NONE | 15 | Useful for lineage/source-era grouping, not meteorology by itself. |
| 07_hko_radar_satellite_lightning_nowcast/hko_lightning_counts_latest.parquet | content_sha256 | string | hash_identifier | 102 | 102 | 0 | 0.0 |  |  | KEEP | live_exact_vintage | PROVENANCE_ONLY | NOT_A_PREDICTOR | NONE | 0 | Integrity/deduplication key; essential for provenance, not forecasting. |
| 07_hko_radar_satellite_lightning_nowcast/hko_lightning_counts_latest.parquet | raw_retrieved_at_utc | string | datetime_or_date | 102 | 102 | 0 | 0.0 | 2026-06-19T04:57:47.135172+00:00 | 2026-06-20T08:03:02.341258+00:00 | KEEP | live_exact_vintage | PROVENANCE_TIMESTAMP | GATING_OR_AUDIT_ONLY | VALIDATE_TIMEZONE | 95 | Needed to prove retrieval/availability and reproduce vintages; not a weather variable. |
| 07_hko_radar_satellite_lightning_nowcast/hko_lightning_counts_latest.parquet | period | string | nominal_text | 102 | 102 | 0 | 0.0 |  |  | KEEP | live_exact_vintage | LIVE_VALID_OR_ISSUE_TIME | FUTURE_LIVE_PREDICTOR | PARSE_TO_TIMESTAMPTZ_AND_ENFORCE_CUTOFF | 90 | Exact-vintage temporal key; current history is too short. |
| 07_hko_radar_satellite_lightning_nowcast/hko_lightning_counts_latest.parquet | lightning_type | string | nominal_text | 102 | 102 | 0 | 0.0 |  |  | KEEP | live_exact_vintage | SPATIAL_OR_TYPE_DIMENSION | FUTURE_LIVE_CONTEXT | NORMALIZE_ENUMS | 45 | Needed to localize live convective/radar signals. |
| 07_hko_radar_satellite_lightning_nowcast/hko_lightning_counts_latest.parquet | region | string | nominal_text | 102 | 102 | 0 | 0.0 |  |  | KEEP | live_exact_vintage | SPATIAL_OR_TYPE_DIMENSION | FUTURE_LIVE_CONTEXT | NORMALIZE_ENUMS | 45 | Needed to localize live convective/radar signals. |
| 07_hko_radar_satellite_lightning_nowcast/hko_lightning_counts_latest.parquet | lightning_count | int64 | numeric | 102 | 102 | 0 | 0.0 | 0.0 | 0.0 | KEEP | live_exact_vintage | CONVECTION_SIGNAL | FUTURE_LIVE_PREDICTOR | CONTINUE_COLLECTION_AND_CHECK_ZERO_VARIANCE | 55 | Potential convective regime signal; current sample is entirely zero. |
| 07_hko_radar_satellite_lightning_nowcast/hko_lightning_counts_latest.parquet | availability_tier | string | nominal_text | 102 | 102 | 0 | 0.0 |  |  | KEEP | live_exact_vintage | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 07_hko_radar_satellite_lightning_nowcast/hko_lightning_counts_latest.parquet | operational_input_allowed | bool | boolean | 102 | 102 | 0 | 0.0 |  |  | KEEP | live_exact_vintage | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 07_hko_radar_satellite_lightning_nowcast/hko_radar_manifest_frames.parquet | source_id | string | identifier_or_code | 80 | 80 | 0 | 0.0 |  |  | KEEP | live_object_catalog | SOURCE_OR_CATALOG_METADATA | CONDITIONAL_CATEGORICAL_ONLY | NONE | 15 | Useful for lineage/source-era grouping, not meteorology by itself. |
| 07_hko_radar_satellite_lightning_nowcast/hko_radar_manifest_frames.parquet | content_sha256 | string | hash_identifier | 80 | 80 | 0 | 0.0 |  |  | KEEP | live_object_catalog | PROVENANCE_ONLY | NOT_A_PREDICTOR | NONE | 0 | Integrity/deduplication key; essential for provenance, not forecasting. |
| 07_hko_radar_satellite_lightning_nowcast/hko_radar_manifest_frames.parquet | raw_retrieved_at_utc | string | datetime_or_date | 80 | 80 | 0 | 0.0 | 2026-06-19T04:57:42.275051+00:00 | 2026-06-19T04:57:42.275051+00:00 | KEEP | live_object_catalog | PROVENANCE_TIMESTAMP | GATING_OR_AUDIT_ONLY | VALIDATE_TIMEZONE | 95 | Needed to prove retrieval/availability and reproduce vintages; not a weather variable. |
| 07_hko_radar_satellite_lightning_nowcast/hko_radar_manifest_frames.parquet | range_key | string | nominal_text | 80 | 80 | 0 | 0.0 |  |  | KEEP | live_object_catalog | SPATIAL_OR_TYPE_DIMENSION | FUTURE_LIVE_CONTEXT | NORMALIZE_ENUMS | 45 | Needed to localize live convective/radar signals. |
| 07_hko_radar_satellite_lightning_nowcast/hko_radar_manifest_frames.parquet | frame_relative_path | string | path_or_file_reference | 80 | 80 | 0 | 0.0 |  |  | KEEP_REFERENCE_ONLY | live_object_catalog | OBJECT_OR_SOURCE_REFERENCE | NOT_A_PREDICTOR | REPLACE_MACHINE_LOCAL_PATH_WITH_OBJECT_URI | 5 | Keep as lineage/object reference; avoid machine-specific absolute paths in canonical DB. |
| 07_hko_radar_satellite_lightning_nowcast/hko_radar_manifest_frames.parquet | frame_time_hkt | string | datetime_or_date | 80 | 80 | 0 | 0.0 | 2026-06-19T02:54:00+00:00 | 2026-06-19T04:48:00+00:00 | KEEP | live_object_catalog | LIVE_VALID_OR_ISSUE_TIME | FUTURE_LIVE_PREDICTOR | PARSE_TO_TIMESTAMPTZ_AND_ENFORCE_CUTOFF | 90 | Exact-vintage temporal key; current history is too short. |
| 07_hko_radar_satellite_lightning_nowcast/hko_radar_manifest_frames.parquet | availability_tier | string | nominal_text | 80 | 80 | 0 | 0.0 |  |  | KEEP | live_object_catalog | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 07_hko_radar_satellite_lightning_nowcast/hko_radar_manifest_frames.parquet | operational_input_allowed | bool | boolean | 80 | 80 | 0 | 0.0 |  |  | KEEP | live_object_catalog | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 07_hko_radar_satellite_lightning_nowcast/hko_satellite_image_inventory.parquet | source_id | string | identifier_or_code | 4589 | 4589 | 0 | 0.0 |  |  | KEEP | live_object_catalog | SOURCE_OR_CATALOG_METADATA | CONDITIONAL_CATEGORICAL_ONLY | NONE | 15 | Useful for lineage/source-era grouping, not meteorology by itself. |
| 07_hko_radar_satellite_lightning_nowcast/hko_satellite_image_inventory.parquet | content_sha256 | string | hash_identifier | 4589 | 4589 | 0 | 0.0 |  |  | KEEP | live_object_catalog | PROVENANCE_ONLY | NOT_A_PREDICTOR | NONE | 0 | Integrity/deduplication key; essential for provenance, not forecasting. |
| 07_hko_radar_satellite_lightning_nowcast/hko_satellite_image_inventory.parquet | raw_retrieved_at_utc | string | datetime_or_date | 4589 | 4589 | 0 | 0.0 | 2026-06-19T05:01:57.332497+00:00 | 2026-06-20T07:55:49.583579+00:00 | KEEP | live_object_catalog | PROVENANCE_TIMESTAMP | GATING_OR_AUDIT_ONLY | VALIDATE_TIMEZONE | 95 | Needed to prove retrieval/availability and reproduce vintages; not a weather variable. |
| 07_hko_radar_satellite_lightning_nowcast/hko_satellite_image_inventory.parquet | filename | string | high_cardinality_text | 4589 | 4589 | 0 | 0.0 |  |  | KEEP_REFERENCE_ONLY | live_object_catalog | OBJECT_OR_SOURCE_REFERENCE | NOT_A_PREDICTOR | REPLACE_MACHINE_LOCAL_PATH_WITH_OBJECT_URI | 5 | Keep as lineage/object reference; avoid machine-specific absolute paths in canonical DB. |
| 07_hko_radar_satellite_lightning_nowcast/hko_satellite_image_inventory.parquet | image_time_hkt | string | datetime_or_date | 4589 | 4589 | 0 | 0.0 | 2026-06-17T00:15:00+00:00 | 2026-06-19T23:30:00+00:00 | KEEP | live_object_catalog | LIVE_VALID_OR_ISSUE_TIME | FUTURE_LIVE_PREDICTOR | PARSE_TO_TIMESTAMPTZ_AND_ENFORCE_CUTOFF | 90 | Exact-vintage temporal key; current history is too short. |
| 07_hko_radar_satellite_lightning_nowcast/hko_satellite_image_inventory.parquet | content_length | int64 | numeric | 4589 | 4589 | 0 | 0.0 | 123.0 | 1133077.0 | KEEP | live_object_catalog | FILE_OR_GRID_QC | QUALITY_ONLY | CHECK_CONSTANTS_AND_FILE_COMPLETENESS | 15 | Useful for integrity, not weather prediction. |
| 07_hko_radar_satellite_lightning_nowcast/hko_satellite_image_inventory.parquet | content_path | string | path_or_file_reference | 4589 | 4589 | 0 | 0.0 |  |  | KEEP_REFERENCE_ONLY | live_object_catalog | OBJECT_OR_SOURCE_REFERENCE | NOT_A_PREDICTOR | REPLACE_MACHINE_LOCAL_PATH_WITH_OBJECT_URI | 5 | Keep as lineage/object reference; avoid machine-specific absolute paths in canonical DB. |
| 07_hko_radar_satellite_lightning_nowcast/hko_satellite_image_inventory.parquet | availability_tier | string | nominal_text | 4589 | 4589 | 0 | 0.0 |  |  | KEEP | live_object_catalog | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 07_hko_radar_satellite_lightning_nowcast/hko_satellite_image_inventory.parquet | operational_input_allowed | bool | boolean | 4589 | 4589 | 0 | 0.0 |  |  | KEEP | live_object_catalog | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 07_hko_radar_satellite_lightning_nowcast/hko_satellite_image_inventory.parquet | source_time_policy | string | free_natural_text | 4589 | 4589 | 0 | 0.0 |  |  | KEEP | live_object_catalog | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
