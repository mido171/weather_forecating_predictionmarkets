# Static Geospatial Inventory

## Dataset identity and use

| Field | Value |
| --- | --- |
| dataset_id | 11_static_geospatial_inventory |
| recommended_db_inclusion | YES-object metadata, then derived feature dimension |
| recommended_layer | static_object_catalog / station_context |
| current_operational_predictor_value_0_100 | 20 |
| diagnostic_research_value_0_100 | 70 |
| future_potential_0_100 | 85 |
| verdict | Terrain/coast/exposure features are valuable; inventory rows themselves are not predictors. |
| source_tables_or_files | 1 |
| audited_attributes | 11 |
| profiled_rows_across_files | 60 |
| data_min |  |
| data_max |  |

## Source tables/files

| source_file | type | rows | attributes | data_min | data_max | db_action | db_layer | model_status | priority | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11_static_geospatial_inventory/static_geospatial_package_inventory.parquet | parquet | 60 | 11 |  |  | REGISTER_OBJECT_METADATA | static_object_catalog | DERIVE_STATIC_FEATURES | HIGH | Inventory itself is not a predictor; derive station/site terrain/coast/land-use features. |

## Attribute nature summary

### Semantic classes

| semantic_class | attribute_count |
| --- | --- |
| boolean | 1 |
| datetime_or_date | 1 |
| free_natural_text | 1 |
| hash_identifier | 1 |
| identifier_or_code | 1 |
| nominal_text | 2 |
| numeric | 2 |
| path_or_file_reference | 2 |

### Model roles

| model_role | attribute_count |
| --- | --- |
| LEAKAGE_POLICY_METADATA | 3 |
| OBJECT_OR_SOURCE_REFERENCE | 1 |
| PROVENANCE_ONLY | 1 |
| PROVENANCE_TIMESTAMP | 1 |
| SOURCE_OR_CATALOG_METADATA | 1 |
| STATIC_ASSET_QC | 3 |
| STATIC_OBJECT_REFERENCE | 1 |

### Operational status

| operational_status | attribute_count |
| --- | --- |
| CONDITIONAL_CATEGORICAL_ONLY | 1 |
| DERIVATION_INPUT_ONLY | 1 |
| GATING_ONLY | 3 |
| GATING_OR_AUDIT_ONLY | 1 |
| NOT_A_PREDICTOR | 2 |
| NOT_DIRECT | 3 |

### Storage decisions

| storage_decision | attribute_count |
| --- | --- |
| KEEP | 9 |
| KEEP_REFERENCE_ONLY | 2 |

## Dataset-specific quality issues

| severity | source_table | attributes | evidence | required_action |
| --- | --- | --- | --- | --- |
| MEDIUM | land-use assets | 2018-2024 land-use rasters | Using recent land-use maps for early historical rows creates temporal context leakage/misrepresentation. | Use terrain/coastline as static; make land use date-effective or reserve it for modern-era models. |

## Complete audited attribute dictionary

This table includes every audited attribute found for this dataset. It records the physical source file, source type, inferred semantic class, missingness, storage decision, DB layer, model role, operational status, quality action, usefulness score, observed profile min/max, and the audit rationale.

| source_file | attribute | source_dtype | semantic_class | row_count | non_null_count | null_count | null_pct | profile_min | profile_max | storage_decision | db_layer | model_role | operational_status | quality_action | usefulness_score_0_100 | rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11_static_geospatial_inventory/static_geospatial_package_inventory.parquet | source_id | string | identifier_or_code | 60 | 60 | 0 | 0.0 |  |  | KEEP | static_object_catalog | SOURCE_OR_CATALOG_METADATA | CONDITIONAL_CATEGORICAL_ONLY | NONE | 15 | Useful for lineage/source-era grouping, not meteorology by itself. |
| 11_static_geospatial_inventory/static_geospatial_package_inventory.parquet | content_sha256 | string | hash_identifier | 60 | 60 | 0 | 0.0 |  |  | KEEP | static_object_catalog | PROVENANCE_ONLY | NOT_A_PREDICTOR | NONE | 0 | Integrity/deduplication key; essential for provenance, not forecasting. |
| 11_static_geospatial_inventory/static_geospatial_package_inventory.parquet | raw_retrieved_at_utc | string | datetime_or_date | 60 | 60 | 0 | 0.0 | 2026-06-19T09:00:42.747347+00:00 | 2026-06-19T09:06:17.419473+00:00 | KEEP | static_object_catalog | PROVENANCE_TIMESTAMP | GATING_OR_AUDIT_ONLY | VALIDATE_TIMEZONE | 95 | Needed to prove retrieval/availability and reproduce vintages; not a weather variable. |
| 11_static_geospatial_inventory/static_geospatial_package_inventory.parquet | content_length | int64 | numeric | 60 | 60 | 0 | 0.0 | 55.0 | 77733805.0 | KEEP | static_object_catalog | STATIC_ASSET_QC | NOT_DIRECT | VERIFY_CRS_RESOLUTION_LICENSE_AND_COMPLETENESS | 20 | Useful for ingestion quality, not prediction. |
| 11_static_geospatial_inventory/static_geospatial_package_inventory.parquet | content_path | string | path_or_file_reference | 60 | 60 | 0 | 0.0 |  |  | KEEP_REFERENCE_ONLY | static_object_catalog | OBJECT_OR_SOURCE_REFERENCE | NOT_A_PREDICTOR | REPLACE_MACHINE_LOCAL_PATH_WITH_OBJECT_URI | 5 | Keep as lineage/object reference; avoid machine-specific absolute paths in canonical DB. |
| 11_static_geospatial_inventory/static_geospatial_package_inventory.parquet | extension | string | nominal_text | 60 | 60 | 0 | 0.0 |  |  | KEEP | static_object_catalog | STATIC_ASSET_QC | NOT_DIRECT | VERIFY_CRS_RESOLUTION_LICENSE_AND_COMPLETENESS | 20 | Useful for ingestion quality, not prediction. |
| 11_static_geospatial_inventory/static_geospatial_package_inventory.parquet | zip_member_count_sampled | int64 | numeric | 60 | 60 | 0 | 0.0 | 0.0 | 46.0 | KEEP | static_object_catalog | STATIC_ASSET_QC | NOT_DIRECT | VERIFY_CRS_RESOLUTION_LICENSE_AND_COMPLETENESS | 20 | Useful for ingestion quality, not prediction. |
| 11_static_geospatial_inventory/static_geospatial_package_inventory.parquet | zip_members_sample_json | string | path_or_file_reference | 60 | 60 | 0 | 0.0 |  |  | KEEP_REFERENCE_ONLY | static_object_catalog | STATIC_OBJECT_REFERENCE | DERIVATION_INPUT_ONLY | USE_PORTABLE_OBJECT_URI_AND_FULL_MEMBER_MANIFEST | 30 | Object metadata; actual raster/vector extraction is required. |
| 11_static_geospatial_inventory/static_geospatial_package_inventory.parquet | availability_tier | string | nominal_text | 60 | 60 | 0 | 0.0 |  |  | KEEP | static_object_catalog | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 11_static_geospatial_inventory/static_geospatial_package_inventory.parquet | operational_input_allowed | bool | boolean | 60 | 60 | 0 | 0.0 |  |  | KEEP | static_object_catalog | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 11_static_geospatial_inventory/static_geospatial_package_inventory.parquet | source_time_policy | string | free_natural_text | 60 | 60 | 0 | 0.0 |  |  | KEEP | static_object_catalog | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
