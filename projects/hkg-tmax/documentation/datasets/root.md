# [root]

## Dataset identity and use

| Field | Value |
| --- | --- |
| dataset_id | [root] |
| recommended_db_inclusion | YES-catalog/provenance only; archive bytes outside DB |
| recommended_layer | catalog / acquisition_provenance / object_catalog |
| current_operational_predictor_value_0_100 | 0 |
| diagnostic_research_value_0_100 | 45 |
| future_potential_0_100 | 50 |
| verdict | Manifests support reproducibility; ZIP bytes and local paths are not model inputs. |
| source_tables_or_files | 3 |
| audited_attributes | 19 |
| profiled_rows_across_files | 561 |
| data_min |  |
| data_max |  |

## Source tables/files

| source_file | type | rows | attributes | data_min | data_max | db_action | db_layer | model_status | priority | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hko_forecast_rss_archives_20200601_20260619.zip | zip | 0 | 0 |  |  | REGISTER_OBJECT_ONLY | object_catalog | NOT_DIRECT | MEDIUM | Do not store archive bytes in relational DB. |
| hko_forecast_rss_archives_20200601_20260619_manifest.csv | csv | 522 | 13 |  |  | LOAD_METADATA | acquisition_provenance | NOT_DIRECT | HIGH | Critical provenance and completeness manifest. |
| MANIFEST.csv | csv | 39 | 6 |  |  | LOAD_METADATA | catalog | NOT_DIRECT | MEDIUM | Repository catalog; not a weather predictor. |

## Attribute nature summary

### Semantic classes

| semantic_class | attribute_count |
| --- | --- |
| boolean | 1 |
| datetime_or_date | 1 |
| hash_identifier | 1 |
| nominal_text | 8 |
| numeric | 5 |
| path_or_file_reference | 2 |
| url | 1 |

### Model roles

| model_role | attribute_count |
| --- | --- |
| CATALOG_OR_ACQUISITION_PROVENANCE | 5 |
| OBJECT_OR_SOURCE_REFERENCE | 5 |
| PROVENANCE_ONLY | 1 |
| PROVENANCE_TIMESTAMP | 1 |
| QUALITY_CONTROL | 2 |
| SOURCE_OR_CATALOG_METADATA | 5 |

### Operational status

| operational_status | attribute_count |
| --- | --- |
| CONDITIONAL_CATEGORICAL_ONLY | 5 |
| GATING_OR_AUDIT_ONLY | 1 |
| NOT_A_PREDICTOR | 11 |
| POSSIBLE_MISSINGNESS_OR_TRUST_FEATURE_ONLY | 2 |

### Storage decisions

| storage_decision | attribute_count |
| --- | --- |
| KEEP | 9 |
| KEEP_METADATA_ONLY | 5 |
| KEEP_REFERENCE_ONLY | 5 |

## Dataset-specific quality issues

No dataset-specific audit issue is recorded in the 2026-06-23 audit bundle.

## Complete audited attribute dictionary

This table includes every audited attribute found for this dataset. It records the physical source file, source type, inferred semantic class, missingness, storage decision, DB layer, model role, operational status, quality action, usefulness score, observed profile min/max, and the audit rationale.

| source_file | attribute | source_dtype | semantic_class | row_count | non_null_count | null_count | null_pct | profile_min | profile_max | storage_decision | db_layer | model_role | operational_status | quality_action | usefulness_score_0_100 | rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hko_forecast_rss_archives_20200601_20260619_manifest.csv | feed | object | nominal_text | 522 | 522 | 0 | 0.0 |  |  | KEEP | acquisition_provenance | SOURCE_OR_CATALOG_METADATA | CONDITIONAL_CATEGORICAL_ONLY | NONE | 15 | Useful for lineage/source-era grouping, not meteorology by itself. |
| hko_forecast_rss_archives_20200601_20260619_manifest.csv | kind | object | nominal_text | 522 | 522 | 0 | 0.0 |  |  | KEEP | acquisition_provenance | SOURCE_OR_CATALOG_METADATA | CONDITIONAL_CATEGORICAL_ONLY | NONE | 15 | Useful for lineage/source-era grouping, not meteorology by itself. |
| hko_forecast_rss_archives_20200601_20260619_manifest.csv | timestamp | float64 | numeric | 522 | 516 | 6 | 0.011494 | 20200601.0 | 20260619.0 | KEEP_METADATA_ONLY | acquisition_provenance | CATALOG_OR_ACQUISITION_PROVENANCE | NOT_A_PREDICTOR | NORMALIZE_PATHS_AND_TIMESTAMPS | 10 | Repository/acquisition metadata, not weather data. |
| hko_forecast_rss_archives_20200601_20260619_manifest.csv | url | object | url | 522 | 522 | 0 | 0.0 |  |  | KEEP_REFERENCE_ONLY | acquisition_provenance | OBJECT_OR_SOURCE_REFERENCE | NOT_A_PREDICTOR | REPLACE_MACHINE_LOCAL_PATH_WITH_OBJECT_URI | 5 | Keep as lineage/object reference; avoid machine-specific absolute paths in canonical DB. |
| hko_forecast_rss_archives_20200601_20260619_manifest.csv | status_code | int64 | numeric | 522 | 522 | 0 | 0.0 | 200.0 | 200.0 | KEEP | acquisition_provenance | QUALITY_CONTROL | POSSIBLE_MISSINGNESS_OR_TRUST_FEATURE_ONLY | STANDARDIZE_ENUMS_AND_QUARANTINE_BAD_ROWS | 35 | Quality/trust metadata; useful for filtering and possibly uncertainty, not primary physical signal. |
| hko_forecast_rss_archives_20200601_20260619_manifest.csv | content_sha256 | object | hash_identifier | 522 | 522 | 0 | 0.0 |  |  | KEEP | acquisition_provenance | PROVENANCE_ONLY | NOT_A_PREDICTOR | NONE | 0 | Integrity/deduplication key; essential for provenance, not forecasting. |
| hko_forecast_rss_archives_20200601_20260619_manifest.csv | content_length | int64 | numeric | 522 | 522 | 0 | 0.0 | 7170.0 | 1041350.0 | KEEP_METADATA_ONLY | acquisition_provenance | CATALOG_OR_ACQUISITION_PROVENANCE | NOT_A_PREDICTOR | NORMALIZE_PATHS_AND_TIMESTAMPS | 10 | Repository/acquisition metadata, not weather data. |
| hko_forecast_rss_archives_20200601_20260619_manifest.csv | path | object | path_or_file_reference | 522 | 522 | 0 | 0.0 |  |  | KEEP_REFERENCE_ONLY | acquisition_provenance | OBJECT_OR_SOURCE_REFERENCE | NOT_A_PREDICTOR | REPLACE_MACHINE_LOCAL_PATH_WITH_OBJECT_URI | 5 | Keep as lineage/object reference; avoid machine-specific absolute paths in canonical DB. |
| hko_forecast_rss_archives_20200601_20260619_manifest.csv | retrieved_at_utc | object | datetime_or_date | 522 | 522 | 0 | 0.0 | 2026-06-20T14:23:43.965980+00:00 | 2026-06-20T14:31:20.142217+00:00 | KEEP | acquisition_provenance | PROVENANCE_TIMESTAMP | GATING_OR_AUDIT_ONLY | VALIDATE_TIMEZONE | 95 | Needed to prove retrieval/availability and reproduce vintages; not a weather variable. |
| hko_forecast_rss_archives_20200601_20260619_manifest.csv | data_gov_filename | object | nominal_text | 522 | 516 | 6 | 0.011494 |  |  | KEEP_REFERENCE_ONLY | acquisition_provenance | OBJECT_OR_SOURCE_REFERENCE | NOT_A_PREDICTOR | REPLACE_MACHINE_LOCAL_PATH_WITH_OBJECT_URI | 5 | Keep as lineage/object reference; avoid machine-specific absolute paths in canonical DB. |
| hko_forecast_rss_archives_20200601_20260619_manifest.csv | data_gov_period | object | nominal_text | 522 | 516 | 6 | 0.011494 |  |  | KEEP_METADATA_ONLY | acquisition_provenance | CATALOG_OR_ACQUISITION_PROVENANCE | NOT_A_PREDICTOR | NORMALIZE_PATHS_AND_TIMESTAMPS | 10 | Repository/acquisition metadata, not weather data. |
| hko_forecast_rss_archives_20200601_20260619_manifest.csv | data_gov_expected_size | float64 | numeric | 522 | 516 | 6 | 0.011494 | 7170.0 | 1041350.0 | KEEP_METADATA_ONLY | acquisition_provenance | CATALOG_OR_ACQUISITION_PROVENANCE | NOT_A_PREDICTOR | NORMALIZE_PATHS_AND_TIMESTAMPS | 10 | Repository/acquisition metadata, not weather data. |
| hko_forecast_rss_archives_20200601_20260619_manifest.csv | skipped_existing | bool | boolean | 522 | 522 | 0 | 0.0 |  |  | KEEP | acquisition_provenance | QUALITY_CONTROL | POSSIBLE_MISSINGNESS_OR_TRUST_FEATURE_ONLY | STANDARDIZE_ENUMS_AND_QUARANTINE_BAD_ROWS | 35 | Quality/trust metadata; useful for filtering and possibly uncertainty, not primary physical signal. |
| MANIFEST.csv | dataset_folder | object | nominal_text | 39 | 39 | 0 | 0.0 |  |  | KEEP | catalog | SOURCE_OR_CATALOG_METADATA | CONDITIONAL_CATEGORICAL_ONLY | NONE | 15 | Useful for lineage/source-era grouping, not meteorology by itself. |
| MANIFEST.csv | dataset_title | object | nominal_text | 39 | 39 | 0 | 0.0 |  |  | KEEP | catalog | SOURCE_OR_CATALOG_METADATA | CONDITIONAL_CATEGORICAL_ONLY | NONE | 15 | Useful for lineage/source-era grouping, not meteorology by itself. |
| MANIFEST.csv | file_name | object | nominal_text | 39 | 39 | 0 | 0.0 |  |  | KEEP_REFERENCE_ONLY | catalog | OBJECT_OR_SOURCE_REFERENCE | NOT_A_PREDICTOR | REPLACE_MACHINE_LOCAL_PATH_WITH_OBJECT_URI | 5 | Keep as lineage/object reference; avoid machine-specific absolute paths in canonical DB. |
| MANIFEST.csv | organized_path | object | path_or_file_reference | 39 | 39 | 0 | 0.0 |  |  | KEEP_REFERENCE_ONLY | catalog | OBJECT_OR_SOURCE_REFERENCE | NOT_A_PREDICTOR | REPLACE_MACHINE_LOCAL_PATH_WITH_OBJECT_URI | 5 | Keep as lineage/object reference; avoid machine-specific absolute paths in canonical DB. |
| MANIFEST.csv | bytes | int64 | numeric | 39 | 39 | 0 | 0.0 | 6736.0 | 54883488.0 | KEEP_METADATA_ONLY | catalog | CATALOG_OR_ACQUISITION_PROVENANCE | NOT_A_PREDICTOR | NORMALIZE_PATHS_AND_TIMESTAMPS | 10 | Repository/acquisition metadata, not weather data. |
| MANIFEST.csv | storage | object | nominal_text | 39 | 39 | 0 | 0.0 |  |  | KEEP | catalog | SOURCE_OR_CATALOG_METADATA | CONDITIONAL_CATEGORICAL_ONLY | NONE | 15 | Useful for lineage/source-era grouping, not meteorology by itself. |
